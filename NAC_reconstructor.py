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

        self.CODE_TO_CATEGORY: Dict[str, str] = {
            'Q': 'offset', 'K': 'offset', 'V': 'offset', 'M': 'offset',
            'B': 'offset', 'W': 'offset', 'T': 'offset', 'P': 'offset',
            'S': 'const', 'A': 'const', 'f': 'const', 'i': 'const',
            'b': 'const', 's': 'const', 'c': 'const',
        }

        self.id_to_canonical = { 2: "<INPUT>", 3: "<OUTPUT>", 6: "<CONTROL_FLOW>", 7: "<CONVERGENCE>" }
        self.id_to_canonical.update(NAC_OPS)
        
        self.code_to_action: Dict[int, str] = {10: 'SAVE_RESULT', 20: 'FREE', 30: 'FORWARD', 40: 'PRELOAD'}

    def _map_enum_to_dtype_str(self, enum: int) -> str:
        return {
            0: 'float32', 1: 'float64', 2: 'float16', 3: 'bfloat16',
            4: 'int32', 5: 'int64', 6: 'int16', 7: 'int8', 8: 'uint8', 9: 'bool'
        }.get(enum, 'unknown')

    def _infer_special_cd_lengths(self, A: int, B: int) -> Tuple[int, int]:
        op_name = self.id_to_canonical.get(A)
        if op_name == "<INPUT>": return (2 if B in (1, 2, 3) else 0), 0
        if op_name == "<OUTPUT>": num_outputs = self.io_counts[1]; return num_outputs + 1, num_outputs
        if op_name == "<CONTROL_FLOW>": return 3, 1
        if op_name == "<CONVERGENCE>": return (-1, -1) # Dynamic length
        return 0, 0

    def _read_op(self, f: Any) -> Dict:
        A, B = struct.unpack('<BB', f.read(2))
        C, D = [], []
        
        if A < 10:  # Special ops
            nC, nD = self._infer_special_cd_lengths(A, B)
            if nC > 0: C = list(struct.unpack(f'<{nC}h', f.read(nC * 2)))
            if nD > 0: D = list(struct.unpack(f'<{nD}h', f.read(nD * 2)))
        else:  # Regular ops
            perm = self.permutations.get(B)
            if perm:
                num_consts_in_perm = sum(1 for p in perm if self.CODE_TO_CATEGORY.get(p) == 'const')
                if num_consts_in_perm > 0:
                    num_consts_from_c, = struct.unpack('<h', f.read(2))
                    if num_consts_from_c > 0:
                        C = [num_consts_from_c] + list(struct.unpack(f'<{num_consts_from_c}h', f.read(num_consts_from_c * 2)))
                    else: C = [0]
                
                nD = len(perm)
                if nD > 0: D = list(struct.unpack(f'<{nD}h', f.read(nD * 2)))
        return {'A': A, 'B': B, 'C': C, 'D': D}

    def _load_nac_file(self, nac_path: str):
        print(f"Loading self-contained binary NAC file: {nac_path}")
        with open(nac_path, 'rb') as f:
            # --- HEADER ---
            if f.read(3) != b'NAC': raise ValueError("'NAC' magic bytes not found.")
            version, quant_byte = struct.unpack('<BB', f.read(2))
            if version != 1: raise ValueError(f"Unsupported NAC version {version}.")
            
            self.weights_stored_internally = (quant_byte & 0x80) != 0
            quant_map = {0: 'none', 1: 'FP16', 2: 'INT8_TENSOR', 3: 'INT8_CHANNEL'}
            self.quantization_method = quant_map.get(quant_byte & 0x7F, 'unknown')
            
            num_inputs, num_outputs, _ = struct.unpack('<HHB', f.read(5))
            self.io_counts = (num_inputs, num_outputs)
            
            offsets_header_format = '<H9Q4x' 
            header_bytes = f.read(struct.calcsize(offsets_header_format))
            self.d_model, *offsets = struct.unpack(offsets_header_format, header_bytes)
        
            mmap_off, ops_off, cmap_off, cnst_off, perm_off, data_off, proc_off, meta_off, rsrc_off = offsets
            print(f"NAC v{version}, d_model: {self.d_model}, Quant: '{self.quantization_method}', IO: {self.io_counts}, Weights: {'Internal' if self.weights_stored_internally else 'External'}")

            # --- MMAP ---
            if mmap_off > 0:
                f.seek(mmap_off); f.read(4)
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    instr_id, num_commands = struct.unpack('<HB', f.read(3))
                    self.memory_map[instr_id] = [{'action': self.code_to_action.get(c, f'UK_{c}'), 'target_id': t} for c, t in [struct.unpack('<BH', f.read(3)) for _ in range(num_commands)]]

            # --- CMAP, CNST, PERM ---
            if cmap_off > 0:
                f.seek(cmap_off); f.read(4)
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    op_id, name_len = struct.unpack('<HB', f.read(3)); self.id_to_canonical[op_id] = f.read(name_len).decode('utf-8')
            if cnst_off > 0:
                f.seek(cnst_off); f.read(4)
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    const_id, type_code, length = struct.unpack('<HBH', f.read(5)); val = None
                    if type_code == 1: val = struct.unpack('<?', f.read(length))[0]
                    elif type_code == 2: val = struct.unpack('<q', f.read(length))[0]
                    elif type_code == 3: val = struct.unpack('<d', f.read(length))[0]
                    elif type_code == 4: val = f.read(length).decode('utf-8')
                    elif type_code == 5: val = list(struct.unpack(f'<{length}i', f.read(length * 4)))
                    elif type_code == 6: val = list(struct.unpack(f'<{length}f', f.read(length * 4)))
                    self.constants[const_id] = val
            if perm_off > 0:
                f.seek(perm_off); f.read(4)
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    p_id, p_len = struct.unpack('<HB', f.read(3)); self.permutations[p_id] = tuple(f.read(p_len).decode('utf-8'))

            # --- OPS ---
            if ops_off > 0:
                f.seek(ops_off); f.read(4)
                self.parsed_nodes = [self._read_op(f) for _ in range(struct.unpack('<I', f.read(4))[0])]

            # --- DATA ---
            if data_off > 0:
                f.seek(data_off); f.read(4)
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    p_id, name_len = struct.unpack('<HH', f.read(4)); self.param_id_to_name[p_id] = f.read(name_len).decode('utf-8')
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    i_idx, name_len = struct.unpack('<HH', f.read(4)); self.input_node_idx_to_name[i_idx] = f.read(name_len).decode('utf-8')
                
                if self.weights_stored_internally:
                    num_tensors = struct.unpack('<I', f.read(4))[0]
                    quant_code_to_str = {0: 'none', 1: 'INT8_TENSOR', 2: 'INT8_CHANNEL'}
                    for _ in range(num_tensors):
                        p_id, meta_len, data_len = struct.unpack('<HIQ', f.read(14))
                        meta_bytes = f.read(meta_len)
                        f.seek(data_len, 1) # Skip the tensor data

                        meta_offset = 0; metadata = {}
                        dtype_id, rank = struct.unpack_from('<BB', meta_bytes, meta_offset); meta_offset += 2
                        metadata['dtype'] = self._map_enum_to_dtype_str(dtype_id)
                        
                        shape = list(struct.unpack_from(f'<{rank}I', meta_bytes, meta_offset)) if rank > 0 else []; meta_offset += rank * 4
                        metadata['shape'] = shape
                        
                        quant_type_code, = struct.unpack_from('<B', meta_bytes, meta_offset); meta_offset += 1
                        quant_type_str = quant_code_to_str.get(quant_type_code, 'none')
                        if quant_type_str != 'none': metadata['quant_type'] = quant_type_str

                        if quant_type_str == 'INT8_TENSOR':
                            metadata['scale'], = struct.unpack_from('<f', meta_bytes, meta_offset); meta_offset += 4
                        elif quant_type_str == 'INT8_CHANNEL':
                            axis, num_scales = struct.unpack_from('<BI', meta_bytes, meta_offset); meta_offset += 5
                            metadata['axis'] = axis
                            metadata['scales'] = list(struct.unpack_from(f'<{num_scales}f', meta_bytes, meta_offset))
                        
                        dummy_tensor = torch.empty(0)
                        self.loaded_param_data[p_id] = (dummy_tensor, metadata)

        print(f"Successfully loaded {len(self.parsed_nodes)} ops, {len(self.memory_map)} MMAP records, and metadata for {len(self.loaded_param_data)} tensors.")


    def reconstruct_from_nac_file(self, nac_path: str, show_mmap: bool = False) -> Tuple[str, str]:
        try:
            self._load_nac_file(nac_path)
        except Exception as e:
            print(f"FATAL ERROR loading or parsing NAC file: {e}"); traceback.print_exc(); return "", ""

        print("\n--- Reconstructing from loaded NAC data ---")
        lines = []
        action_priority = {'FORWARD': 0, 'FREE': 1, 'SAVE_RESULT': 2, 'PRELOAD': 3}
        user_inputs_info = []

        for i, node in enumerate(self.parsed_nodes):
            var_name = f"v{i}"; self.global_var_map[i] = var_name
            A, B, C, D = node['A'], node['B'], node['C'], node['D']
            op_name = self.id_to_canonical.get(A, f"<UNKNOWN_OP_{A}>")
            line = ""

            if op_name == "<INPUT>":
                if B == 0:
                    name = self.input_node_idx_to_name.get(i, f"<unnamed_input_{i}>")
                    line = f"{var_name} = user_input(name='{name}')"
                    user_inputs_info.append(f"  - Input index {i}: name='{name}'")
                elif B == 1:
                    param_id = C[1]; param_name = self.param_id_to_name.get(param_id, '<UNKNOWN>')
                    param_info = f"name='{param_name}'"
                    if param_id in self.loaded_param_data:
                        _, metadata = self.loaded_param_data[param_id]
                        param_info += f", {', '.join(f'{k}={v}' for k, v in metadata.items())}"
                    line = f"{var_name} = load_param({param_info})"
                elif B == 3:
                    const_val = self.constants.get(C[1])
                    line = f"{var_name} = lifted_constant(value={repr(const_val)})"
                    user_inputs_info.append(f"  - Input index {i}: name='lifted_constant', value={repr(const_val)}")
            elif op_name == "<OUTPUT>":
                output_deps = [self.global_var_map.get(i + offset, f"v{i+offset}_<ERR>") for offset in D]
                line = f"return {', '.join(output_deps)}"
            else: 
                final_args = []
                perm = self.permutations.get(B)
                if perm:
                    c_iter = iter(C[1:] if C and C[0] > 0 else [])
                    
                    const_markers_in_D = [idx for idx, val in enumerate(D) if val == 0]
                    const_marker_iter = iter(const_markers_in_D)
                    
                    for d_idx, d_val in enumerate(D):
                        if d_val != 0:
                            final_args.append(self.global_var_map.get(i + d_val, f"v{i+d_val}_<ERR>"))
                        else:
                            const_id = next(c_iter)
                            final_args.append(repr(self.constants.get(const_id, f"<CONST_ERR_{const_id}>")))
                
                args_str = ", ".join(final_args)
                line = f"{var_name} = {op_name}({args_str})"

            if show_mmap and i in self.memory_map:
                commands = sorted(self.memory_map[i], key=lambda cmd: (action_priority.get(cmd['action'], 99), cmd['target_id']))
                mmap_str = ", ".join([f"{cmd['action']} -> {cmd['target_id']}" for cmd in commands])
                if mmap_str: line += " " + mmap_str
            lines.append("  " + line)
            
        return "\n".join(lines), "\n".join(sorted(user_inputs_info))

def reconstruct_from_file(nac_filepath: str, show_mmap: bool = False):
    print("\n" + "="*20 + f" RECONSTRUCTION OF {os.path.basename(nac_filepath)} " + "="*20)
    if not os.path.exists(nac_filepath):
        print(f"Error: File not found: {os.path.abspath(nac_filepath)}"); return
        
    reconstructor = Reconstructor()
    pseudo_code, input_summary = reconstructor.reconstruct_from_nac_file(nac_filepath, show_mmap=show_mmap)
    
    if pseudo_code:
        print("\n--- Reconstructed Model Info ---")
        print(f"File: {os.path.abspath(nac_filepath)}")
        print(f"Model Dimension (d_model): {reconstructor.d_model}")
        print(f"Quantization Method: {reconstructor.quantization_method}")
        print(f"Total User Inputs: {reconstructor.io_counts[0]}")

        if input_summary:
            print("\n--- User Input Summary (in order of execution) ---")
            print(input_summary)

        print("\n--- Reconstructed Pseudo-code ---")
        print(pseudo_code)
        print("-" * 30)

if __name__ == "__main__":
    args = sys.argv[1:]
    show_mmap_flag = '-m' in args or '--mmap' in args
    if show_mmap_flag: args = [a for a in args if a not in ('-m', '--mmap')]

    if not args: print("Usage: python NAC_reconstructor.py [-m|--mmap] <path_to_model.nac>"); sys.exit(1)
        
    reconstruct_from_file(nac_filepath=args[0], show_mmap=show_mmap_flag)