# Copyright (c) 2025 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

import os
import struct
import sys
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Callable, Optional, Union
from safetensors.numpy import load_file
import time
import json
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.simplefilter("ignore", FutureWarning)

from NAC_kernels import NacKernelBase, softmax, NAC_OPS
from TISA_tokenizer import TISAVM

class NacRuntime(NacKernelBase):
    CODE_TO_CATEGORY: Dict[str, str] = {
        'Q': 'offset', 'K': 'offset', 'V': 'offset', 'M': 'offset', 'B': 'offset',
        'W': 'offset', 'T': 'offset', 'P': 'offset', 'S': 'const', 'A': 'const',
        'f': 'const', 'i': 'const', 'b': 'const', 's': 'const', 'c': 'const',
    }
    
    def __init__(self, nac_path: str):
        print("--- Initializing NAC Runtime ---")
        self._load_nac_file(nac_path)
        self._register_ops()
        self.results: List[Any] = []
        self._auto_init_state_map()
        self.perf_stats: Dict[str, Tuple[float, int]] = {}
        print("Runtime ready.")

    def _infer_special_cd_lengths(self, A: int, B: int) -> Tuple[int, int]:
        op_name = self.id_to_canonical.get(A)
        if op_name == "<INPUT>": return (2 if B in (1, 2, 3) else 0), 0
        if op_name == "<OUTPUT>": num_outputs = self.io_counts[1]; return num_outputs + 1, num_outputs
        if op_name == "<CONTROL_FLOW>": return 3, 1
        if op_name == "<CONVERGENCE>": return (-1, 1) 
        return 0, 0

    def _read_op(self, f: Any) -> Dict:
        A, B = struct.unpack('<BB', f.read(2))
        C, D = [], []
        op_name = self.id_to_canonical.get(A)

        if A < 10:  # Special (System) Operations
            nC, nD = self._infer_special_cd_lengths(A, B)
            if op_name == "<CONVERGENCE>":
                num_elements_c, = struct.unpack('<h', f.read(2))
                C = [num_elements_c] + list(struct.unpack(f'<{num_elements_c}h', f.read(num_elements_c * 2)))
            elif nC > 0:
                C = list(struct.unpack(f'<{nC}h', f.read(nC * 2)))
            if nD > 0:
                D = list(struct.unpack(f'<{nD}h', f.read(nD * 2)))
        else:  # Regular Operations (A >= 10)
            perm = self.permutations.get(B)
            if perm:
                # Read C if it is expected
                num_consts_in_perm = sum(1 for p in perm if self.CODE_TO_CATEGORY.get(p) == 'const')
                if num_consts_in_perm > 0:
                    try:
                        num_consts_from_c, = struct.unpack('<h', f.read(2))
                        if num_consts_from_c > 0:
                            C = [num_consts_from_c] + list(struct.unpack(f'<{num_consts_from_c}h', f.read(num_consts_from_c * 2)))
                        else: C = [0]
                    except struct.error: C = []
                
                # Read D, its length is equal to the permutation length
                nD = len(perm)
                if nD > 0:
                    try:
                        D = list(struct.unpack(f'<{nD}h', f.read(nD * 2)))
                    except struct.error: D = []
        return {'A': A, 'B': B, 'C': C, 'D': D}

    def encode(self, text: str) -> List[int]:
        if not self.tokenizer: raise RuntimeError("Tokenizer not available.")
        return self.tokenizer.run(self.tokenizer.manifest, text)

    def decode(self, ids: Union[List[int], np.ndarray], skip_special_tokens: bool = True) -> str:
        if not self.tokenizer:
            raise RuntimeError("Tokenizer is not available for this NAC model.")
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list): 
            ids = ids[0] # Take the first element of the batch
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def _load_nac_file(self, nac_path: str):
        print(f"Loading NAC file (Robust Binary Format): {nac_path}")
        
        self.id_to_canonical = { 2: "<INPUT>", 3: "<OUTPUT>", 6: "<CONTROL_FLOW>", 7: "<CONVERGENCE>",}
        self.id_to_canonical.update(NAC_OPS)
        
        self.constants, self.permutations, self.parameters = {}, {}, {}
        self.operations, self.param_id_to_name, self.input_node_idx_to_name = [], {}, {}
        self.d_model = 0 
        self.tokenizer = None
        
        with open(nac_path, 'rb') as f:
            # 1. Reading and parsing the header
            if f.read(3) != b'NAC': raise ValueError("'NAC' magic bytes not found.")
            version, raw_quant_id = struct.unpack('<BB', f.read(2))
            if version != 1: raise ValueError(f"Unsupported NAC version: {version}")
            
            num_inputs, num_outputs, _ = struct.unpack('<HHB', f.read(5))
            self.io_counts = (num_inputs, num_outputs)
            
            self.weights_stored_internally = (raw_quant_id & 0x80) != 0
            quant_map = {0:'none', 1:'FP16', 2:'INT8_TENSOR', 3:'INT8_CHANNEL'}
            self.quantization_method = quant_map.get(raw_quant_id & 0x7F, 'unknown')
            
            offsets_header_format = '<H9Q4x'
            header_bytes = f.read(struct.calcsize(offsets_header_format))
            self.d_model, *offsets = struct.unpack(offsets_header_format, header_bytes)
            mmap_off, ops_off, cmap_off, cnst_off, perm_off, data_off, proc_off, meta_off, rsrc_off = offsets
            
            print(f"NAC v{version}, d_model: {self.d_model}, Quant: '{self.quantization_method}', IO: {self.io_counts}, Weights: {'Internal' if self.weights_stored_internally else 'External'}")

            # 2. Loading metadata (CMAP, CNST, PERM)
            if cmap_off > 0:
                f.seek(cmap_off); f.read(4) # Skipping the tag
                num_entries = struct.unpack('<I', f.read(4))[0]
                for _ in range(num_entries):
                    op_id, name_len = struct.unpack('<HB', f.read(3))
                    self.id_to_canonical[op_id] = f.read(name_len).decode('utf-8')
            
            if cnst_off > 0:
                f.seek(cnst_off); f.read(4)
                num_entries = struct.unpack('<I', f.read(4))[0]
                for _ in range(num_entries):
                    const_id, type_code, length = struct.unpack('<HBH', f.read(5)); val = None
                    if type_code == 1: val = struct.unpack('<?', f.read(length))[0]
                    elif type_code == 2: val = struct.unpack('<q', f.read(length))[0]
                    elif type_code == 3: val = struct.unpack('<d', f.read(length))[0]
                    elif type_code == 4: val = f.read(length).decode('utf-8')
                    elif type_code == 5: val = list(struct.unpack(f'<{length}i', f.read(length * 4))) if length > 0 else []
                    elif type_code == 6: val = list(struct.unpack(f'<{length}f', f.read(length * 4))) if length > 0 else []
                    self.constants[const_id] = val
            
            if perm_off > 0:
                f.seek(perm_off); f.read(4)
                num_entries = struct.unpack('<I', f.read(4))[0]
                for _ in range(num_entries):
                    p_id, p_len = struct.unpack('<HB', f.read(3))
                    self.permutations[p_id] = tuple(f.read(p_len).decode('utf-8'))
            
            if ops_off > 0:
                f.seek(ops_off); f.read(4)
                num_ops = struct.unpack('<I', f.read(4))[0]
                self.operations = [self._read_op(f) for _ in range(num_ops)]

            # 3. Loading data (DATA)
            if data_off > 0:
                f.seek(data_off); f.read(4)
                # Reading name mappings
                num_params = struct.unpack('<I', f.read(4))[0]
                for _ in range(num_params):
                    p_id, name_len = struct.unpack('<HH', f.read(4))
                    self.param_id_to_name[p_id] = f.read(name_len).decode('utf-8')
                
                num_inputs = struct.unpack('<I', f.read(4))[0]
                for _ in range(num_inputs):
                    i_idx, name_len = struct.unpack('<HH', f.read(4))
                    self.input_node_idx_to_name[i_idx] = f.read(name_len).decode('utf-8')
                
                # Loading weight tensors
                if self.weights_stored_internally:
                    num_tensors = struct.unpack('<I', f.read(4))[0]
                    numpy_dtype_map = {0:np.float32, 1:np.float64, 2:np.float16, 3:"<bf16>", 4:np.int32, 5:np.int64, 6:np.int16, 7:np.int8, 8:np.uint8, 9:np.bool_}
                    quant_code_to_str = {0: 'none', 1: 'INT8_TENSOR', 2: 'INT8_CHANNEL'}

                    for _ in range(num_tensors):
                        p_id, meta_len, data_len = struct.unpack('<HIQ', f.read(14))
                        meta_bytes = f.read(meta_len)
                        data_bytes = f.read(data_len)

                        meta_offset = 0
                        meta = {}
                        
                        # 1. Dtype and Rank
                        dtype_id, rank = struct.unpack_from('<BB', meta_bytes, meta_offset)
                        meta_offset += 2
                        
                        # 2. Shape
                        shape = []
                        if rank > 0:
                            shape = list(struct.unpack_from(f'<{rank}I', meta_bytes, meta_offset))
                            meta_offset += rank * 4
                        meta['shape'] = shape
                        
                        # 3. Quantization Info
                        quant_type_code, = struct.unpack_from('<B', meta_bytes, meta_offset)
                        meta_offset += 1
                        quant_type_str = quant_code_to_str.get(quant_type_code, 'none')
                        meta['quant_type'] = quant_type_str

                        if quant_type_str == 'INT8_TENSOR':
                            scale, = struct.unpack_from('<f', meta_bytes, meta_offset)
                            meta_offset += 4
                            meta['scale'] = scale
                        elif quant_type_str == 'INT8_CHANNEL':
                            axis, num_scales = struct.unpack_from('<BI', meta_bytes, meta_offset)
                            meta_offset += 5
                            scales = list(struct.unpack_from(f'<{num_scales}f', meta_bytes, meta_offset))
                            meta['axis'], meta['scales'] = axis, scales
                        
                        dtype = numpy_dtype_map.get(dtype_id, np.float32)
                        if dtype == "<bf16>":
                            u16_arr = np.frombuffer(data_bytes, dtype=np.uint16).reshape(shape)
                            arr = (u16_arr.astype(np.uint32) << 16).view(np.float32).copy()
                        else:
                            arr = np.frombuffer(data_bytes, dtype=dtype).reshape(shape).copy()
                        
                        self.parameters[p_id] = self._dequantize(arr, meta)
                else: # Loading from an external .safetensors file
                    safetensors_path = os.path.splitext(nac_path)[0] + '.safetensors'
                    if not os.path.exists(safetensors_path): raise FileNotFoundError(f"External weights file not found: {safetensors_path}")
                    
                    from safetensors import safe_open
                    
                    tensors = {}
                    per_tensor_metadata = {}
                    with safe_open(safetensors_path, framework="np", device="cpu") as st_f:
                        # 1. Loading general file metadata
                        file_metadata = st_f.metadata() or {}
                        # 2. Extract the 'metadata' field, which stores metadata for each tensor.
                        if "metadata" in file_metadata:
                            # This is a string containing a JSON dictionary where the key is the name of the tensor,
                            # and the value is ANOTHER JSON ROW with the metadata of this tensor.
                            metadata_of_metadata = json.loads(file_metadata["metadata"])
                            for tensor_name, meta_json_str in metadata_of_metadata.items():
                                per_tensor_metadata[tensor_name] = json.loads(meta_json_str)

                        # 3. Loading the tensors
                        for key in st_f.keys():
                            tensors[key] = st_f.get_tensor(key)

                    # 4. Compare tensors, their metadata and dequantize
                    for p_id, p_name in self.param_id_to_name.items():
                        if p_name in tensors:
                            meta = per_tensor_metadata.get(p_name, {}) # Get the metadata dictionary for this tensor
                            self.parameters[p_id] = self._dequantize(tensors[p_name], meta)
            
            # 4. Loading tokenizer resources (RSRC)
            tokenizer_resources_raw = {}
            if rsrc_off > 0: # Resources inside .nac
                f.seek(rsrc_off); f.read(4)
                num_files = struct.unpack('<I', f.read(4))[0]
                print(f"Reading {num_files} internal resource files from RSRC section...")
                for _ in range(num_files):
                    name_len = struct.unpack('<H', f.read(2))[0]
                    filename = f.read(name_len).decode('utf-8')
                    data_len = struct.unpack('<I', f.read(4))[0]
                    tokenizer_resources_raw[filename] = f.read(data_len)
            else: # Resources outside
                tokenizer_dir = os.path.splitext(nac_path)[0] + "-tokenizer"
                if os.path.exists(tokenizer_dir):
                    print(f"Reading external tokenizer resources from: {tokenizer_dir}")
                    for fname in os.listdir(tokenizer_dir):
                        fpath = os.path.join(tokenizer_dir, fname)
                        if os.path.isfile(fpath):
                            with open(fpath, 'rb') as rf: 
                                tokenizer_resources_raw[fname] = rf.read()

            # 5. Initializing TISAVM based on the manifest (PROC) and resources
            if proc_off > 0:
                f.seek(proc_off); f.read(4)
                manifest_bytes = f.read(struct.unpack('<I', f.read(4))[0])

                vm_resources = {}
                print("Processing tokenizer resources for TISAVM...")

                tok_json = None
                if "tokenizer.json" in tokenizer_resources_raw:
                    try: tok_json = json.loads(tokenizer_resources_raw["tokenizer.json"].decode('utf-8'))
                    except json.JSONDecodeError: print("  - WARNING: Could not parse 'tokenizer.json'.")
                
                if tok_json and tok_json.get('model', {}).get('type') == 'Unigram':
                    unigram_vocab_list = tok_json.get('model', {}).get('vocab', [])
                    vm_resources['vocab'] = {token: i for i, (token, score) in enumerate(unigram_vocab_list)}
                    vm_resources['unigram_scores'] = {token: score for token, score in unigram_vocab_list}
                    print("  - Vocab and Unigram scores loaded from 'tokenizer.json'.")
                elif tok_json and 'vocab' in tok_json.get('model', {}):
                    vm_resources['vocab'] = tok_json.get('model', {}).get('vocab')
                    print("  - Vocab loaded from 'tokenizer.json'.")
                elif "vocab.json" in tokenizer_resources_raw:
                    vm_resources['vocab'] = json.loads(tokenizer_resources_raw["vocab.json"].decode('utf-8'))
                    print("  - Vocab loaded from 'vocab.json'.")
                else: print("  - WARNING: No vocab file found ('tokenizer.json' or 'vocab.json').")

                if tok_json and "merges" in tok_json.get("model", {}):
                    merges_lines = tok_json["model"]["merges"]
                    vm_resources['ranks'] = {tuple(line.split()): i for i, line in enumerate(merges_lines)}
                    print("  - Merges loaded from 'tokenizer.json'.")
                elif "merges.txt" in tokenizer_resources_raw:
                    merges_content = tokenizer_resources_raw["merges.txt"].decode('utf-8')
                    merges_lines = [line for line in merges_content.splitlines() if line and not line.startswith("#")]
                    vm_resources['ranks'] = {tuple(line.split()): i for i, line in enumerate(merges_lines)}
                    print("  - Merges loaded from 'merges.txt'.")

                if vm_resources.get('vocab'):
                    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
                    cs = bs[:]; n = 0
                    for b in range(256):
                        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
                    vm_resources['byte_map'] = dict(zip(bs, [chr(c) for c in cs]))
                    
                    vm_resources.setdefault('ranks', {})
                    vm_resources.setdefault('unigram_scores', {})

                    self.tokenizer = TISAVM(vm_resources)
                    self.tokenizer.manifest = manifest_bytes
                    print("TISAVM initialized successfully.")
                else: print("Warning: Tokenizer manifest found, but vocab/resource files are missing or empty.")
            
            print(f"Loaded {len(self.operations)} ops and {len(self.parameters)} parameters.")

    def _dequantize(self, arr: np.ndarray, metadata: Dict) -> np.ndarray:
        if metadata.get('quant_type') == 'INT8_TENSOR': return arr.astype(np.float32) * metadata.get('scale', 1.0)
        if metadata.get('quant_type') == 'INT8_CHANNEL':
            scales = np.array(metadata['scales'], dtype=np.float32); shape = [1]*arr.ndim; shape[metadata.get('axis',0)]=-1
            return arr.astype(np.float32) * scales.reshape(shape)
        if arr.dtype == np.float16: return arr.astype(np.float32)
        return arr

    def _auto_init_state_map(self):
        self.state_map: Dict[int, np.ndarray] = {}
        for op in self.operations:
            if self.id_to_canonical.get(op['A']) == "<INPUT>" and op.get('B') == 2:
                output_idx = op['C'][1]
                self.state_map[output_idx] = np.zeros((1, 12, 0, 64), dtype=np.float32)

    def _register_ops(self):
        self.op_kernels: Dict[str, Callable] = {name: getattr(self, name) for name in dir(self) if name.startswith('op_')}
        self.op_kernels['op_branch'] = self._kernel_branch

    def _print_perf_stats(self):
        if not self.perf_stats: return
        print("\n--- Operator Performance ---")
        sorted_stats = sorted(self.perf_stats.items(), key=lambda item: item[1][0], reverse=True)
        print(f"{'Operation':<50} | {'Avg Time (ms)':>15} | {'Total Time (ms)':>15} | {'Calls':>7}")
        print("-" * 100)
        for op_name, (total_time, count) in sorted_stats:
            if count == 0: continue
            avg_time_ms = (total_time / count) * 1000
            total_time_ms = total_time * 1000
            print(f"{op_name:<50} | {avg_time_ms:15.4f} | {total_time_ms:15.2f} | {count:>7}")
        print("-" * 100)

    def run(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        is_first_run = not hasattr(self, '_is_initialized')
        if is_first_run:
            print("\n--- Starting Model Execution ---")
            self._is_initialized = True
            self.perf_stats = {} 
        self.results = [None] * len(self.operations)
        import itertools
        self.input_stream = itertools.cycle(inputs)
        res = self._execute_block(0, len(self.operations), verbose=is_first_run)
        if is_first_run:
            print("--- Execution Finished ---")
            self._print_perf_stats()
        return [res] if not isinstance(res, list) else res
    
    def _execute_block(self, start_idx: int, num_ops: int, verbose: bool = False) -> Any:
        ip = start_idx
        last_res = None
        end_idx = start_idx + num_ops
        while ip < end_idx:
            op = self.operations[ip]
            op_name = self.id_to_canonical.get(op['A'], "")
            if op_name == "<OUTPUT>" and op['B'] == 1:
                output_deps = [self.results[ip + offset] for offset in op['D']]
                return output_deps[0] if len(output_deps) == 1 else tuple(output_deps)
            if op_name == "<INPUT>":
                B, C = op['B'], op['C']
                if B == 0: self.results[ip] = next(self.input_stream)
                elif B == 1: self.results[ip] = self.parameters[C[1]]
                elif B == 2: self.results[ip] = self.state_map.get(C[1])
                elif B == 3: self.results[ip] = self.constants.get(C[1])
            elif op_name == "<OUTPUT>" and op['B'] == 0:
                output_deps = [self.results[ip + offset] for offset in op['D']]
                last_res = output_deps[0] if len(output_deps) == 1 else tuple(output_deps)
                break
            elif op_name != "<NONE>":
                start_time = time.time()
                kernel_name = "op_branch" if op_name == "<CONVERGENCE>" else "op_" + op_name.replace('.', '_')
                kernel = self.op_kernels.get(kernel_name)
                if not kernel: raise NotImplementedError(f"Op '{op_name}' ({kernel_name}) not implemented.")
                if kernel_name == "op_branch":
                    self.results[ip] = kernel(op, ip)
                else:
                    args = self._gather_args(op, ip)
                    import inspect
                    sig = inspect.signature(kernel)
                    if "_perm" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                        self.results[ip] = kernel(*args, _perm=self.permutations.get(op.get('B', 0)))
                    else:
                        self.results[ip] = kernel(*args)
                duration = time.time() - start_time
                total_time, count = self.perf_stats.get(op_name, (0.0, 0))
                self.perf_stats[op_name] = (total_time + duration, count + 1)
                if op_name == "<CONVERGENCE>":
                    ip += op['D'][0]
                    continue
            last_res = self.results[ip]
            ip += 1
        return last_res

    def _kernel_branch(self, op: Dict, current_ip: int) -> Any:
        input_tensor = self.results[current_ip + op['C'][0]]
        coherence_threshold = float(op['B']) / 100.0 if op['B'] > 0 else 0.5
        jump_offset_after_branches, num_branches, *branch_offsets = op['D']
        if num_branches != len(branch_offsets): raise ValueError(f"BRANCH op at {current_ip} corrupted.")
        all_offsets = sorted(branch_offsets + [jump_offset_after_branches])
        branch_definitions = []
        for i in range(num_branches):
            start_offset = branch_offsets[i]
            end_offset = all_offsets[all_offsets.index(start_offset) + 1]
            branch_definitions.append({"start": current_ip + start_offset, "len": end_offset - start_offset})
        print(f"--- Executing <CONVERGENCE> with {len(branch_definitions)} branches (coherence: {coherence_threshold}) ---")
        branch_results = [res for branch in branch_definitions if (res := self._execute_block(branch['start'], branch['len'])) is not None]
        print(f"  -> {len(branch_results)} branch results collected. Starting EMM fusion.")
        if not branch_results: return input_tensor
        if len(branch_results) == 1: return branch_results[0]
        main_tensors = [np.asarray(res[0] if isinstance(res, tuple) else res) for res in branch_results]
        max_len = max(t.shape[1] for t in main_tensors if t.ndim > 1)
        padded_main_tensors = []
        for t in main_tensors:
            if t.ndim > 1 and t.shape[1] < max_len:
                pad_width = [(0, 0)] * t.ndim; pad_width[1] = (0, max_len - t.shape[1])
                padded_main_tensors.append(np.pad(t, pad_width, 'constant'))
            else: padded_main_tensors.append(t)
        coherence_scores = []
        for i in range(len(padded_main_tensors)):
            others = padded_main_tensors[:i] + padded_main_tensors[i+1:]
            if not others: coherence_scores.append(1.0); continue
            external_center = np.mean(np.stack(others, axis=0), axis=0)
            sim_flat = np.sum(padded_main_tensors[i].reshape(padded_main_tensors[i].shape[0], -1) * external_center.reshape(external_center.shape[0], -1), axis=1)
            coherence_scores.append(np.mean(sim_flat))
        coherent_indices = [i for i, score in enumerate(coherence_scores) if score >= coherence_threshold]
        if not coherent_indices:
            print("  -> Fusion Decision: No coherent streams. Selecting closest to mean.")
            center_of_mass = np.mean(np.stack(padded_main_tensors, axis=0), axis=0)
            distances = [np.linalg.norm(t - center_of_mass) for t in padded_main_tensors]
            return branch_results[np.argmin(distances)]
        print(f"  -> Fusion Decision: Averaging {len(coherent_indices)} coherent streams.")
        coherent_results = [branch_results[i] for i in coherent_indices]
        if isinstance(coherent_results[0], tuple):
            final_tuple = []
            for i in range(len(coherent_results[0])):
                elements_to_avg = [res[i] for res in coherent_results if i < len(res) and res[i] is not None and isinstance(res[i], np.ndarray)]
                if not elements_to_avg: final_tuple.append(None); continue
                if elements_to_avg[0].ndim > 1:
                    max_len_elem = max(e.shape[1] for e in elements_to_avg if e.ndim > 1)
                    padded_elements = []
                    for e in elements_to_avg:
                        if e.ndim > 1 and e.shape[1] < max_len_elem:
                            pad_width = [(0, 0)] * e.ndim; pad_width[1] = (0, max_len_elem - e.shape[1])
                            padded_elements.append(np.pad(e, pad_width, 'constant'))
                        else: padded_elements.append(e)
                    final_tuple.append(np.mean(np.stack(padded_elements, axis=0), axis=0))
                else: final_tuple.append(np.mean(np.stack(elements_to_avg, axis=0), axis=0))
            return tuple(final_tuple)
        else:
            coherent_main_tensors = [padded_main_tensors[i] for i in coherent_indices]
            return np.mean(np.stack(coherent_main_tensors, axis=0), axis=0)

    def _gather_args(self, op: Dict[str, Any], current_idx: int) -> List[Any]:
        args = []
        perm = self.permutations.get(op.get('B'))
        if not perm: return []

        c_ids = op['C'][1:] if op.get('C') and op['C'][0] > 0 else []
        c_iter = iter(c_ids)
        d_values = op.get('D', [])

        if len(d_values) != len(perm):
             raise ValueError(f"Instruction {current_idx}: Mismatch between perm length ({len(perm)}) and D field length ({len(d_values)}).")

        for i in range(len(perm)):
            d_val = d_values[i]
            if d_val != 0:
                ancestor_idx = current_idx + d_val
                args.append(self.results[ancestor_idx])
            else:
                try:
                    const_id = next(c_iter)
                    args.append(self.constants.get(const_id))
                except StopIteration:
                    raise ValueError(f"Instruction {current_idx}: D field expects a constant, but C field is exhausted.")
        return args

    def _gather_args1(self, op: Dict[str, Any], current_idx: int) -> List[Any]:
        # This version uses a different principle - it will work, but does not guarantee correctness
        args = []
        c_ids = op['C'][1:] if op.get('C') and op['C'][0] > 0 else []
        c_iter, d_iter = iter(c_ids), iter(op.get('D', []))
        perm = self.permutations.get(op.get('B'))
        if not perm: return []
        
        for p_code in perm:
            category = self.CODE_TO_CATEGORY.get(p_code, '?')
            dep_entry = next(d_iter)
            if category == 'offset': args.append(self.results[current_idx + dep_entry])
            elif category == 'const':
                try: args.append(self.constants.get(next(c_iter)))
                except StopIteration: args.append(None)
        return args