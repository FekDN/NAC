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

        # --- Initializing special and standard operations ---
        self.id_to_canonical = {
            2: "<INPUT>",
            3: "<OUTPUT>",
            6: "<CONTROL_FLOW>",
            7: "<CONVERGENCE>",
        }
        # Loading standard operations from NAC_kernels.py
        self.id_to_canonical.update(NAC_OPS)


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
            self.d_model, *offsets = struct.unpack(offsets_header_format, header_bytes)
            
            ops_off, cmap_off, cnst_off, perm_off, data_off, \
            proc_off, meta_off, rsrc_off, reserved2_off = offsets
            
            print(f"NAC v{version}, d_model: {self.d_model}, Quant: '{self.quantization_method}', IO: {self.io_counts}, Weights: {'Internal' if self.weights_stored_internally else 'External'}")

            # --- CMAP (ID-Length-Value) ---
            f.seek(cmap_off); f.read(4) 
            num_custom_ops = struct.unpack('<I', f.read(4))[0]
            for _ in range(num_custom_ops):
                op_id, name_len = struct.unpack('<HB', f.read(3))
                op_name = f.read(name_len).decode('utf-8')
                if op_id not in self.id_to_canonical:
                    self.id_to_canonical[op_id] = op_name
                else:
                    print(f"Warning: CMAP contains standard op ID {op_id} ('{op_name}'). Ignoring, using predefined '{self.id_to_canonical[op_id]}'.")

            # --- CNST (ID-Length-Value) ---
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

            # --- PERM (ID-Length-Value) ---
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
                if op_name_peek == "<CONVERGENCE>":
                    num_c, = struct.unpack('<h', f.read(2))
                    c_vals = [num_c] + list(struct.unpack(f'<{num_c}h', f.read(num_c * 2)))
                    num_d, = struct.unpack('<h', f.read(2))
                    d_vals = [num_d] + list(struct.unpack(f'<{num_d}h', f.read(num_d * 2)))
                else:
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
                    # [ID: u16][meta_len: u32][data_len: u64]
                    p_id, meta_len, data_len = struct.unpack('<HIQ', f.read(14))
                    
                    # Read the entire metadata block and parse it as JSON.
                    meta_bytes = f.read(meta_len)
                    metadata = json.loads(meta_bytes.decode('utf-8'))

                    # Skip the tensor data, we don't need it for code reconstruction
                    raw_data = f.read(data_len)
                    
                    torch_dtype = self._map_enum_to_dtype(metadata.get('dtype'))
                    shape = metadata.get('shape')
                    if torch_dtype and shape:
                        if (np_dtype := self._numpy_dtype_map.get(torch_dtype)):
                            # We don't create a tensor, we just store its metadata for output.
                            tensor_info = {'shape': shape, 'dtype': str(torch_dtype)}
                            # Deleting already used keys
                            if 'shape' in metadata: del metadata['shape']
                            if 'dtype' in metadata: del metadata['dtype']
                            
                            # Save the dummy tensor and the remaining metadata
                            dummy_tensor = torch.empty(0) # Plug
                            self.loaded_param_data[p_id] = (dummy_tensor, {**tensor_info, **metadata})
            else:
                print("External weights are not loaded, but parameter names are mapped.")

        print(f"Successfully loaded {len(self.parsed_nodes)} ops and metadata for {len(self.loaded_param_data)} tensors.")


    def reconstruct_from_nac_file(self, nac_path: str) -> str:
        try:
            self._load_nac_file(nac_path)
        except (ValueError, IOError, struct.error, json.JSONDecodeError) as e:
            print(f"FATAL ERROR loading or parsing NAC file: {e}"); traceback.print_exc(); return ""

        print("\n--- Reconstructing pseudo-code from loaded NAC data ---")
        lines = []
        indent_level = 0
        indent_control_points = {}

        for i, node in enumerate(self.parsed_nodes):
            op_name = self.id_to_canonical.get(node['A'])
            if op_name == "<CONVERGENCE>":
                D = node['D']
                jump_after, num_branches = D[0], D[1]
                end_of_block = i + jump_after
                indent_control_points[i + 1] = ('START_BRANCH_BLOCK',)
                indent_control_points[end_of_block] = ('END_BRANCH_BLOCK',)
                for offset in D[2:]:
                    indent_control_points[i + offset] = ('START_BRANCH',)

        for i, node in enumerate(self.parsed_nodes):
            if i in indent_control_points:
                if indent_control_points[i][0] == 'END_BRANCH_BLOCK': indent_level -= 1
            indent_str = "  " * indent_level
            if i in indent_control_points:
                if indent_control_points[i][0] == 'START_BRANCH': lines.append(f"{indent_str}# --- Branch ---")

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
                        tensor, metadata = loaded_param; param_info += f", shape={list(tensor.shape)}, quant_meta={metadata}"
                    elif isinstance(loaded_param, torch.Tensor):
                        param_info += f", shape={list(loaded_param.shape)}"
                    line = f"{var_name} = load_param({param_info})"
                elif B == 2:
                    line = f"{var_name} = state_input(maps_to_output_idx={C[1]})"
                elif B == 3:
                    line = f"{var_name} = constant(value={repr(self.constants.get(C[1]))})"
            elif op_name == "<OUTPUT>":
                output_deps = [self.global_var_map.get(i + offset, f"v{i+offset}_<ERR>") for offset in D]
                line = f"return {', '.join(output_deps)}"
            elif op_name == "<CONTROL_FLOW>":
                pred_var = self.global_var_map.get(i + D[0], f"v{i+D[0]}_<ERR>")
                line = f"# CONTROL_FLOW on predicate: {pred_var}, true_len={C[1]}, false_len={C[2]}"
            elif op_name == "<CONVERGENCE>":
                input_var = self.global_var_map.get(i + C[0], f"v{i+C[0]}_<ERR>")
                coherence = float(B) / 100.0 if B > 0 else 0.5
                line = f"{var_name} = <CONVERGENCE>(input={input_var}, coherence={coherence})"
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
                            # d_iter should still be advanced even if its value is not used for consts
                            next(d_iter, None) # Use None as the default value to avoid StopIteration
                            const_id = next(c_iter)
                            final_args.append(repr(self.constants.get(const_id, f"<CONST_ERR_{const_id}>")))
                args_str = ", ".join(final_args)
                line = f"{var_name} = {op_name}({args_str})"

            lines.append(indent_str + line)
            
            if i + 1 in indent_control_points:
                if indent_control_points[i + 1][0] == 'START_BRANCH_BLOCK':
                    indent_level += 1
        return "\n".join(lines)

def reconstruct_from_file(nac_filepath: str):
    print("\n" + "="*20 + f" RECONSTRUCTION OF {os.path.basename(nac_filepath)} " + "="*20)
    if not os.path.exists(nac_filepath):
        print(f"Error: File not found: {os.path.abspath(nac_filepath)}")
        return
    reconstructor = Reconstructor()
    pseudo_code = reconstructor.reconstruct_from_nac_file(nac_filepath)
    if pseudo_code:
        print("\n--- Reconstructed Pseudo-code ---")
        print(f"File: {os.path.abspath(nac_filepath)}")
        print(f"Model Dimension (d_model): {reconstructor.d_model}")
        print(f"Quantization Method: {reconstructor.quantization_method}")
        print("-" * 30)
        print(pseudo_code)
        print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python NAC_reconstructor.py <path_to_model.nac>")
        sys.exit(1)
    model_to_reconstruct = sys.argv[1]
    reconstruct_from_file(nac_filepath=model_to_reconstruct)