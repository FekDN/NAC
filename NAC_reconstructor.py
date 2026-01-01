# Copyright (c) 2025 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

import os
import sys
import json
import struct
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
import torch
import traceback

try:
    from NAC_kernels import NAC_OPS
except ImportError:
    print("Warning: NAC_kernels.py not found. Standard NAC operations will not be recognized.")
    NAC_OPS = {}


class Reconstructor:
    """Reconstructs pseudo-code from a single, self-contained NAC file."""
    def __init__(self):
        print("\n--- Initializing Reconstructor ---")
        self.id_to_canonical: Dict[int, str] = {}
        self.constants: Dict[int, Any] = {}
        self.permutations: Dict[int, Tuple[str, ...]] = {}
        self.parsed_nodes: List[Dict[str, Any]] = []
        self.loaded_param_data: Dict[int, Union[torch.Tensor, Tuple[torch.Tensor, Dict]]] = {}
        self.global_var_map: Dict[int, str] = {}
        self.quantization_method: str = "none"
        self.weights_stored_internally: bool = True
        self.io_counts: Tuple[int, int] = (0, 0)
        self.d_model: int = 0
        self.param_id_to_name: Dict[int, str] = {}
        self.input_node_idx_to_name: Dict[int, str] = {}
        self.memory_map: Dict[int, List[Dict[str, Any]]] = {}

        self._numpy_dtype_map = {
            torch.float32: np.float32, torch.float64: np.float64, torch.float16: np.float16,
            torch.bfloat16: np.float32, # np doesn't have bfloat16, convert to float32 for reading
            torch.int32: np.int32, torch.int64: np.int64, torch.int16: np.int16,
            torch.int8: np.int8, torch.uint8: np.uint8, torch.bool: np.bool_
        }

        self.CODE_TO_CATEGORY: Dict[str, str] = {
            'Q': 'offset', 'K': 'offset', 'V': 'offset', 'M': 'offset',
            'B': 'offset', 'W': 'offset', 'T': 'offset', 'P': 'offset',
            'S': 'const', 'A': 'const', 'f': 'const', 'i': 'const',
            'b': 'const', 's': 'const', 'c': 'const',
        }

        self.id_to_canonical = { 2: "<INPUT>", 3: "<OUTPUT>", 6: "<CONTROL_FLOW>", 7: "<CONVERGENCE>" }
        self.id_to_canonical.update(NAC_OPS)
        
        # Для чтения секции MMAP
        self.code_to_action: Dict[int, str] = {10: 'SAVE_RESULT', 20: 'FREE', 30: 'FORWARD', 40: 'PRELOAD'}


    def _map_enum_to_dtype(self, enum: int) -> Optional[torch.dtype]:
        return {
            0: torch.float32, 1: torch.float64, 2: torch.float16, 3: torch.bfloat16,
            4: torch.int32, 5: torch.int64, 6: torch.int16, 7: torch.int8, 8: torch.uint8, 9: torch.bool
        }.get(enum)

    def _infer_cd_lengths(self, A: int, B: int) -> Tuple[int, int]:
        op_name = self.id_to_canonical.get(A)
        if op_name == "<INPUT>": return (2 if B in (1, 2, 3) else 0), 0
        if op_name == "<OUTPUT>": num_outputs = self.io_counts[1]; return num_outputs + 1, num_outputs
        if op_name == "<CONTROL_FLOW>": return 3, 1
        if op_name == "<CONVERGENCE>": return -1, -1 
        if B > 0:
            perm = self.permutations.get(B)
            if not perm: raise ValueError(f"Invalid permutation ID {B} for op {op_name} (A={A}).")
            nD = len(perm)
            num_consts = sum(1 for p in perm if self.CODE_TO_CATEGORY.get(p) == 'const')
            nC = num_consts + 1 if num_consts > 0 else 0
            return nC, nD
        return 0, 0

    def _load_nac_file(self, nac_path: str):
        print(f"Loading self-contained binary NAC file: {nac_path}")
        with open(nac_path, 'rb') as f:
            # --- HEADER ---
            if f.read(3) != b'NAC': raise ValueError("'NAC' magic bytes not found.")
            version = struct.unpack('<B', f.read(1))[0]
            if version != 1: raise ValueError(f"Unsupported NAC version {version}.")
            quant_byte = struct.unpack('<B', f.read(1))[0]
            self.weights_stored_internally = (quant_byte & 0x80) != 0
            quant_map = {0: 'none', 1: 'FP16', 2: 'INT8_TENSOR', 3: 'INT8_CHANNEL'}
            self.quantization_method = quant_map.get(quant_byte & 0x7F, 'unknown')
            num_inputs, num_outputs, _ = struct.unpack('<HHB', f.read(5))
            self.io_counts = (num_inputs, num_outputs)
            
            offsets_header_format = '<H9Q6x' 
            header_bytes = f.read(struct.calcsize(offsets_header_format))
            unpacked_header = struct.unpack(offsets_header_format, header_bytes)
            self.d_model = unpacked_header[0]
            offsets = unpacked_header[1:]
        
            mmap_off, ops_off, cmap_off, cnst_off, perm_off, data_off, \
            proc_off, meta_off, rsrc_off = offsets
            
            print(f"NAC v{version}, d_model: {self.d_model}, Quant: '{self.quantization_method}', IO: {self.io_counts}, Weights: {'Internal' if self.weights_stored_internally else 'External'}")

            # --- MMAP Section (ID-Length-[Action-Target]...) ---
            if mmap_off > 0:
                f.seek(mmap_off); f.read(4) # 'MMAP'
                num_mmap_records = struct.unpack('<I', f.read(4))[0]
                for _ in range(num_mmap_records):
                    instr_id, num_commands = struct.unpack('<HB', f.read(3))
                    commands_for_tick = []
                    for _ in range(num_commands):
                        action_code, target_id = struct.unpack('<BH', f.read(3))
                        action_name = self.code_to_action.get(action_code, f'UNKNOWN_ACTION_{action_code}')
                        commands_for_tick.append({'action': action_name, 'target_id': target_id})
                    self.memory_map[instr_id] = commands_for_tick
            
            # --- CMAP ---
            f.seek(cmap_off); f.read(4) 
            num_custom_ops = struct.unpack('<I', f.read(4))[0]
            for _ in range(num_custom_ops):
                op_id, name_len = struct.unpack('<HB', f.read(3))
                op_name = f.read(name_len).decode('utf-8')
                if op_id not in self.id_to_canonical: self.id_to_canonical[op_id] = op_name

            # --- CNST ---
            f.seek(cnst_off); f.read(4)
            num_consts = struct.unpack('<I', f.read(4))[0]
            for _ in range(num_consts):
                const_id, type_code, length = struct.unpack('<HBH', f.read(5))
                val = None
                if type_code == 0: val = None
                elif type_code == 1: val = struct.unpack('<?', f.read(length))[0]
                elif type_code == 2: val = struct.unpack('<q', f.read(length))[0]
                elif type_code == 3: val = struct.unpack('<d', f.read(length))[0]
                elif type_code == 4: val = f.read(length).decode('utf-8')
                elif type_code == 5: val = list(struct.unpack(f'<{length}i', f.read(length * 4))) if length > 0 else []
                elif type_code == 6: val = list(struct.unpack(f'<{length}f', f.read(length * 4))) if length > 0 else []
                self.constants[const_id] = val

            # --- PERM ---
            f.seek(perm_off); f.read(4)
            for _ in range(struct.unpack('<I', f.read(4))[0]):
                p_id, p_len = struct.unpack('<HB', f.read(3))
                self.permutations[p_id] = tuple(f.read(p_len).decode('utf-8'))

            # --- OPS ---
            f.seek(ops_off); f.read(4)
            op_count = struct.unpack('<I', f.read(4))[0]
            for i in range(op_count):
                A, B = struct.unpack('<BB', f.read(2))
                op_name_peek = self.id_to_canonical.get(A)
                c_vals, d_vals = [], []
                nC, nD = self._infer_cd_lengths(A, B)
                if nC > 0: c_vals = list(struct.unpack(f'<{nC}h', f.read(nC * 2)))
                if nD > 0: d_vals = list(struct.unpack(f'<{nD}h', f.read(nD * 2)))
                self.parsed_nodes.append({'A': A, 'B': B, 'C': c_vals, 'D': d_vals})

            # --- DATA ---
            f.seek(data_off); f.read(4)
            for _ in range(struct.unpack('<I', f.read(4))[0]):
                p_id, name_len = struct.unpack('<HH', f.read(4))
                self.param_id_to_name[p_id] = f.read(name_len).decode('utf-8')
            for _ in range(struct.unpack('<I', f.read(4))[0]):
                i_idx, name_len = struct.unpack('<HH', f.read(4))
                self.input_node_idx_to_name[i_idx] = f.read(name_len).decode('utf-8')
            
            if self.weights_stored_internally:
                param_count = struct.unpack('<I', f.read(4))[0]
                for _ in range(param_count):
                    p_id, meta_len, data_len = struct.unpack('<HIQ', f.read(14))
                    metadata = json.loads(f.read(meta_len).decode('utf-8'))
                    f.seek(data_len, 1) # Skip tensor data
                    
                    torch_dtype = self._map_enum_to_dtype(metadata.get('dtype'))
                    shape = metadata.get('shape', [])
                    
                    tensor_info = {'shape': shape, 'dtype': str(torch_dtype)}
                    if 'shape' in metadata: del metadata['shape']
                    if 'dtype' in metadata: del metadata['dtype']
                    
                    dummy_tensor = torch.empty(0)
                    self.loaded_param_data[p_id] = (dummy_tensor, {**tensor_info, **metadata})

        print(f"Successfully loaded {len(self.parsed_nodes)} ops, {len(self.memory_map)} MMAP records, and metadata for {len(self.loaded_param_data)} tensors.")


    def reconstruct_from_nac_file(self, nac_path: str, show_mmap: bool = False) -> str:
        try:
            self._load_nac_file(nac_path)
        except (ValueError, IOError, struct.error, json.JSONDecodeError) as e:
            print(f"FATAL ERROR loading or parsing NAC file: {e}"); traceback.print_exc(); return ""

        print("\n--- Reconstructing pseudo-code from loaded NAC data ---")
        lines = []
        action_priority = {'FORWARD': 0, 'FREE': 1, 'SAVE_RESULT': 2, 'PRELOAD': 3}

        for i, node in enumerate(self.parsed_nodes):
            indent_str = "  "
            var_name = f"v{i}"; self.global_var_map[i] = var_name
            A, B, C, D = node['A'], node['B'], node['C'], node['D']
            op_name = self.id_to_canonical.get(A, f"<UNKNOWN_OP_{A}>")
            line = ""

            if op_name == "<INPUT>":
                if B == 0:
                    name_str = f", name='{self.input_node_idx_to_name.get(i)}'" if i in self.input_node_idx_to_name else ""
                    line = f"{var_name} = data_input(type=DATA{name_str})"
                elif B == 1:
                    param_id = C[1]; param_name = self.param_id_to_name.get(param_id, '<UNKNOWN>')
                    param_info = f"name='{param_name}', id={param_id}"
                    loaded_param = self.loaded_param_data.get(param_id)
                    if isinstance(loaded_param, tuple):
                        _, metadata = loaded_param
                        # В метаданных shape уже есть, поэтому не дублируем
                        param_info += f", {', '.join(f'{k}={v}' for k, v in metadata.items())}"
                    line = f"{var_name} = load_param({param_info})"
                elif B == 3:
                    line = f"{var_name} = constant(value={repr(self.constants.get(C[1]))})"
            elif op_name == "<OUTPUT>":
                output_deps = [self.global_var_map.get(i + offset, f"v{i+offset}_<ERR>") for offset in D]
                line = f"return {', '.join(output_deps)}"
            else: 
                final_args = []
                perm = self.permutations.get(B)
                if perm:
                    c_iter, d_iter = iter(C[1:] if C and C[0] > 0 else []), iter(D)
                    for p_code in perm:
                        category = self.CODE_TO_CATEGORY.get(p_code, '?')
                        if category == 'offset':
                            offset = next(d_iter)
                            final_args.append(self.global_var_map.get(i + offset, f"v{i+offset}_<ERR>"))
                        elif category == 'const':
                            const_id = next(c_iter)
                            final_args.append(repr(self.constants.get(const_id, f"<CONST_ERR_{const_id}>")))
                args_str = ", ".join(final_args)
                line = f"{var_name} = {op_name}({args_str})"

            # Добавляем запись управления памятью, если флаг -m активен
            if show_mmap and i in self.memory_map:
                commands = self.memory_map[i]
                commands.sort(key=lambda cmd: (action_priority.get(cmd['action'], 99), cmd['target_id']))
                mmap_str_parts = [f"{cmd['action']} -> {cmd['target_id']}" for cmd in commands]
                if mmap_str_parts:
                    line += " " + ", ".join(mmap_str_parts)

            lines.append(indent_str + line)
            
        return "\n".join(lines)

def reconstruct_from_file(nac_filepath: str, show_mmap: bool = False):
    print("\n" + "="*20 + f" RECONSTRUCTION OF {os.path.basename(nac_filepath)} " + "="*20)
    if not os.path.exists(nac_filepath):
        print(f"Error: File not found: {os.path.abspath(nac_filepath)}")
        return
    reconstructor = Reconstructor()
    pseudo_code = reconstructor.reconstruct_from_nac_file(nac_filepath, show_mmap=show_mmap)
    if pseudo_code:
        print("\n--- Reconstructed Pseudo-code ---")
        print(f"File: {os.path.abspath(nac_filepath)}")
        print(f"Model Dimension (d_model): {reconstructor.d_model}")
        print(f"Quantization Method: {reconstructor.quantization_method}")
        print(f"Memory Management Records: {'Shown' if show_mmap else 'Hidden'}")
        print("-" * 30)
        print(pseudo_code)
        print("-" * 30)

if __name__ == "__main__":
    args = sys.argv[1:]
    show_mmap_flag = '-m' in args or '--mmap' in args
    if show_mmap_flag:
        args = [arg for arg in args if arg not in ('-m', '--mmap')]

    if not args:
        print("Usage: python NAC_reconstructor.py [-m|--mmap] <path_to_model.nac>")
        sys.exit(1)
        
    model_to_reconstruct = args[0]
    reconstruct_from_file(nac_filepath=model_to_reconstruct, show_mmap=show_mmap_flag)