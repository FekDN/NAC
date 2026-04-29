# --- START OF FILE NAC_run.py ---
# Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

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
        self._is_initialized = False
        print("Runtime ready.")

    def _infer_special_cd_lengths(self, A: int, B: int) -> Tuple[int, int]:
        op_name = self.id_to_canonical.get(A)
        if op_name == "<INPUT>": return (2 if B in (1, 2, 3, 4, 5) else 0), 0
        if op_name == "<OUTPUT>":
            if B == 3: return 2, 2
            if B == 4: return 0, 1
            num_outputs = self.io_counts[1]
            return num_outputs + 1, num_outputs
        if op_name == "<CONTROL_FLOW>": return 3, 1
        if op_name == "<CONVERGENCE>": return (-1, 1)
        return 0, 0

    def _read_op(self, f: Any) -> Dict:
        A, B = struct.unpack('<BB', f.read(2))
        C, D = [], []
        op_name = self.id_to_canonical.get(A)

        if A < 10:
            nC, nD = self._infer_special_cd_lengths(A, B)
            if op_name == "<CONVERGENCE>":
                num_elements_c, = struct.unpack('<h', f.read(2))
                C = [num_elements_c] + list(struct.unpack(f'<{num_elements_c}h', f.read(num_elements_c * 2)))
            elif nC > 0: C = list(struct.unpack(f'<{nC}h', f.read(nC * 2)))
            if nD > 0: D = list(struct.unpack(f'<{nD}h', f.read(nD * 2)))
        else:
            perm = self.permutations.get(B)
            # getitem / ops with no permutation entry: B-field may encode the literal
            # index, but the C array (constant indices) can still be present.
            # Detect by checking if the op name is a known perm-less call.
            _is_getitem = op_name in ("getitem", "operator.getitem", "aten.select.int")

            if perm or _is_getitem:
                num_consts_in_perm = (sum(1 for p in perm if self.CODE_TO_CATEGORY.get(p) == 'const')
                                      if perm else 1)  # getitem always has 1 constant (the index)
                if num_consts_in_perm > 0:
                    try:
                        num_consts_from_c, = struct.unpack('<h', f.read(2))
                        C = ([num_consts_from_c] + list(struct.unpack(f'<{num_consts_from_c}h',
                             f.read(num_consts_from_c * 2))) if num_consts_from_c > 0 else [0])
                    except struct.error: C = []
                nD = len(perm) if perm else 1  # getitem has exactly 1 input dep
                if nD > 0:
                    try: D = list(struct.unpack(f'<{nD}h', f.read(nD * 2)))
                    except struct.error: D = []
        return {'A': A, 'B': B, 'C': C, 'D': D}

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.run(self.tokenizer.manifest, text)

    def decode(self, ids: Union[List[int], np.ndarray], skip_special_tokens: bool = True) -> str:
        if isinstance(ids, np.ndarray): ids = ids.tolist()
        if ids and isinstance(ids[0], list): ids = ids[0]
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @staticmethod
    def preprocess_image_for_imagenet(image_path: str) -> np.ndarray:
        from PIL import Image
        img = Image.open(image_path).convert('RGB').resize((256, 256))
        left, top = (256 - 224) // 2, (256 - 224) // 2
        arr = np.array(img.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255.0
        return np.expand_dims(((arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])).transpose(2, 0, 1), axis=0)

    @staticmethod
    def load_dynamic_file(path: str, file_type: int) -> np.ndarray:
        if file_type == 2: return np.load(path)
        elif file_type == 3: return NacRuntime.preprocess_image_for_imagenet(path)
        raise ValueError(f"Unknown file_type={file_type}")

    @staticmethod
    def load_mep_from_nac(nac_path: str):
        with open(nac_path, 'rb') as f:
            f.seek(10)
            orch_off = struct.unpack('<H10Q', f.read(82))[8] 
            if orch_off == 0: return None, None
            f.seek(orch_off)
            if f.read(4) != b'ORCH': raise ValueError("Bad ORCH magic")
            bytecode_len, constants_count = struct.unpack('<II', f.read(8))
            if bytecode_len == 0: return None, None
            bytecode = f.read(bytecode_len)
            constants = {}
            for _ in range(constants_count):
                const_id, type_code, length = struct.unpack('<HBH', f.read(5))
                if type_code == 1: val = struct.unpack('<?', f.read(length))[0]
                elif type_code == 2: val = struct.unpack('<q', f.read(length))[0]
                elif type_code == 3: val = struct.unpack('<d', f.read(length))[0]
                elif type_code == 4: val = f.read(length).decode('utf-8')
                elif type_code == 5: val = list(struct.unpack(f'<{length}i', f.read(length * 4))) if length > 0 else []
                elif type_code == 6: val = list(struct.unpack(f'<{length}f', f.read(length * 4))) if length > 0 else []
                else: val = None
                constants[const_id] = val
            return bytecode, constants

    @staticmethod
    def _serialize_mep_constant(const_id: int, val) -> bytes:
        type_code, length, value_bytes = 0, 0, b''
        if val is None: type_code = 0
        elif isinstance(val, bool): type_code, length, value_bytes = 1, 1, struct.pack('<?', val)
        elif isinstance(val, int): type_code, length, value_bytes = 2, 8, struct.pack('<q', val)
        elif isinstance(val, float): type_code, length, value_bytes = 3, 8, struct.pack('<d', val)
        elif isinstance(val, str):
            enc = val.encode('utf-8')
            type_code, length, value_bytes = 4, len(enc), enc
        elif isinstance(val, list):
            if not val: type_code, length = 5, 0
            elif all(isinstance(x, int) for x in val): type_code, length, value_bytes = 5, len(val), struct.pack(f'<{len(val)}i', *val)
            else: type_code, length, value_bytes = 6, len(val), struct.pack(f'<{len(val)}f', *map(float, val))
        return struct.pack('<HBH', const_id, type_code, length) + value_bytes

    def get_mep_plan(self):
        if not getattr(self, '_orch_off', 0): return None, None
        with open(self._nac_path, 'rb') as f:
            f.seek(self._orch_off)
            if f.read(4) != b'ORCH': raise ValueError("Bad ORCH magic")
            blen, clen = struct.unpack('<II', f.read(8))
            if blen == 0: return None, None
            bytecode = f.read(blen)
            constants = {}
            for _ in range(clen):
                const_id, type_code, length = struct.unpack('<HBH', f.read(5))
                if type_code == 1: val = struct.unpack('<?', f.read(length))[0]
                elif type_code == 2: val = struct.unpack('<q', f.read(length))[0]
                elif type_code == 3: val = struct.unpack('<d', f.read(length))[0]
                elif type_code == 4: val = f.read(length).decode('utf-8')
                elif type_code == 5: val = list(struct.unpack(f'<{length}i', f.read(length * 4))) if length > 0 else []
                elif type_code == 6: val = list(struct.unpack(f'<{length}f', f.read(length * 4))) if length > 0 else []
                else: val = None
                constants[const_id] = val
        return bytecode, constants

    def _load_nac_file(self, nac_path: str):
        print(f"Loading NAC file: {nac_path}")
        self.id_to_canonical = {2: "<INPUT>", 3: "<OUTPUT>", 6: "<CONTROL_FLOW>", 7: "<CONVERGENCE>"}
        self.id_to_canonical.update(NAC_OPS)
        self.constants, self.permutations, self.parameters = {}, {}, {}
        self.operations, self.param_id_to_name, self.input_node_idx_to_name = [], {}, {}
        self.tensor_offsets = {}
        self.d_model, self.tokenizer = 0, None
        
        with open(nac_path, 'rb') as f:
            if f.read(3) != b'NAC': raise ValueError("'NAC' magic bytes not found.")
            version, raw_quant_id = struct.unpack('<BB', f.read(2))
            if version != 2: raise ValueError(f"Unsupported NAC version {version}")
            
            num_inputs, num_outputs, _ = struct.unpack('<HHB', f.read(5))
            self.io_counts = (num_inputs, num_outputs)
            self.weights_stored_internally = (raw_quant_id & 0x80) != 0
            self.has_trng = (raw_quant_id & 0x40) != 0
            self.quantization_method = {0:'none', 1:'FP16', 2:'INT8_TENSOR', 3:'INT8_CHANNEL', 4:'BLOCK_FP8'}.get(raw_quant_id & 0x3F, 'unknown')

            header_bytes = f.read(82)
            self.d_model, mmap_off, ops_off, cmap_off, cnst_off, perm_off, data_off, proc_off, orch_off, trng_off, rsrc_off = struct.unpack('<H10Q', header_bytes)

            self._orch_off, self._trng_off, self._nac_path = orch_off, trng_off, nac_path

            if cmap_off > 0:
                f.seek(cmap_off); f.read(4)
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    op_id, name_len = struct.unpack('<HB', f.read(3))
                    self.id_to_canonical[op_id] = f.read(name_len).decode('utf-8')
            
            if cnst_off > 0:
                f.seek(cnst_off); f.read(4)
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    const_id, type_code, length = struct.unpack('<HBH', f.read(5))
                    if type_code == 1: val = struct.unpack('<?', f.read(length))[0]
                    elif type_code == 2: val = struct.unpack('<q', f.read(length))[0]
                    elif type_code == 3: val = struct.unpack('<d', f.read(length))[0]
                    elif type_code == 4: val = f.read(length).decode('utf-8')
                    elif type_code == 5: val = list(struct.unpack(f'<{length}i', f.read(length * 4))) if length > 0 else []
                    elif type_code == 6: val = list(struct.unpack(f'<{length}f', f.read(length * 4))) if length > 0 else []
                    else: val = None
                    self.constants[const_id] = val
            
            if perm_off > 0:
                f.seek(perm_off); f.read(4)
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    p_id, p_len = struct.unpack('<HB', f.read(3))
                    self.permutations[p_id] = tuple(f.read(p_len).decode('utf-8'))
            
            if ops_off > 0:
                f.seek(ops_off); f.read(4)
                self.operations = [self._read_op(f) for _ in range(struct.unpack('<I', f.read(4))[0])]

            if data_off > 0:
                f.seek(data_off); f.read(4)
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    p_id, name_len = struct.unpack('<HH', f.read(4))
                    self.param_id_to_name[p_id] = f.read(name_len).decode('utf-8')
                num_inputs_data = struct.unpack('<I', f.read(4))[0]
                for _ in range(num_inputs_data):
                    i_idx, name_len = struct.unpack('<HH', f.read(4))
                    self.input_node_idx_to_name[i_idx] = f.read(name_len).decode('utf-8')
                
                companion = os.path.splitext(nac_path)[0] + ".safetensors"
                if os.path.isfile(companion) and self.weights_stored_internally:
                    try:
                        _st_data = load_file(companion)
                        _n2id = {v: k for k, v in self.param_id_to_name.items()}
                        for _name, _arr in _st_data.items():
                            if _name in _n2id: self.parameters[_n2id[_name]] = _arr
                        print(f"[NAC] Fast load: loaded weights from '{companion}'")
                    except Exception: pass
                
                # Check if we need to load internal weights
                if len(self.parameters) < len(self.param_id_to_name) and self.weights_stored_internally:
                    num_tensors = struct.unpack('<I', f.read(4))[0]
                    numpy_dtype_map = {0:np.float32, 1:np.float64, 2:np.float16, 3:"<bf16>", 4:np.int32, 5:np.int64, 6:np.int16, 7:np.int8, 8:np.uint8, 9:np.bool_}
                    quant_code_to_str = {0:'none', 1:'FP16', 2:'INT8_TENSOR', 3:'INT8_CHANNEL', 4:'BLOCK_FP8'}
                    for _ in range(num_tensors):
                        p_id, meta_len, data_len = struct.unpack('<HIQ', f.read(14))
                        meta_bytes = f.read(meta_len)
                        
                        data_offset = f.tell()
                        data_bytes = f.read(data_len)
                        
                        meta_offset, meta = 0, {}
                        dtype_id, rank = struct.unpack_from('<BB', meta_bytes, meta_offset); meta_offset += 2
                        shape = list(struct.unpack_from(f'<{rank}I', meta_bytes, meta_offset)) if rank > 0 else []; meta_offset += rank * 4
                        quant_type_str = quant_code_to_str.get(struct.unpack_from('<B', meta_bytes, meta_offset)[0], 'none'); meta_offset += 1
                        meta['quant_type'] = quant_type_str
                        
                        if quant_type_str == 'INT8_TENSOR':
                            meta['scale'], = struct.unpack_from('<f', meta_bytes, meta_offset); meta_offset += 4
                        elif quant_type_str == 'INT8_CHANNEL':
                            meta['axis'], num_scales = struct.unpack_from('<BI', meta_bytes, meta_offset); meta_offset += 5
                            meta['scales'] = list(struct.unpack_from(f'<{num_scales}f', meta_bytes, meta_offset))
                        elif quant_type_str == 'BLOCK_FP8':
                            meta['block_size'], original_rank = struct.unpack_from('<HB', meta_bytes, meta_offset); meta_offset += 3
                            meta['original_shape'] = list(struct.unpack_from(f'<{original_rank}I', meta_bytes, meta_offset)) if original_rank > 0 else []; meta_offset += original_rank * 4
                            num_scales, = struct.unpack_from('<I', meta_bytes, meta_offset); meta_offset += 4
                            meta['scales'] = list(struct.unpack_from(f'<{num_scales}f', meta_bytes, meta_offset))

                        self.tensor_offsets[p_id] = {'offset': data_offset, 'length': data_len, 'meta': meta}

                        if p_id not in self.parameters:
                            dtype = numpy_dtype_map.get(dtype_id, np.float32)
                            arr = (np.frombuffer(data_bytes, dtype=np.uint16).reshape(shape).astype(np.uint32) << 16).view(np.float32).copy() if dtype == "<bf16>" else np.frombuffer(data_bytes, dtype=dtype).reshape(shape).copy()
                            self.parameters[p_id] = self._dequantize(arr, meta)
                
                elif not self.weights_stored_internally:
                    safetensors_path = os.path.splitext(nac_path)[0] + '.safetensors'
                    if os.path.exists(safetensors_path):
                        from safetensors import safe_open
                        tensors, per_tensor_metadata = {}, {}
                        with safe_open(safetensors_path, framework="np", device="cpu") as st_f:
                            file_metadata = st_f.metadata() or {}
                            if "metadata" in file_metadata:
                                for name, meta_str in json.loads(file_metadata["metadata"]).items(): per_tensor_metadata[name] = json.loads(meta_str)
                            for key in st_f.keys(): tensors[key] = st_f.get_tensor(key)
                        for p_id, p_name in self.param_id_to_name.items():
                            if p_name in tensors: self.parameters[p_id] = self._dequantize(tensors[p_name], per_tensor_metadata.get(p_name, {}))
            
            tokenizer_resources_raw = {}
            if rsrc_off > 0:
                f.seek(rsrc_off); f.read(4)
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    name_len = struct.unpack('<H', f.read(2))[0]
                    filename = f.read(name_len).decode('utf-8')
                    data_len = struct.unpack('<I', f.read(4))[0]
                    tokenizer_resources_raw[filename] = f.read(data_len)
            else:
                tokenizer_dir = os.path.splitext(nac_path)[0] + "-tokenizer"
                if os.path.exists(tokenizer_dir):
                    for fname in os.listdir(tokenizer_dir):
                        fpath = os.path.join(tokenizer_dir, fname)
                        if os.path.isfile(fpath):
                            with open(fpath, 'rb') as rf: tokenizer_resources_raw[fname] = rf.read()

            if proc_off > 0:
                f.seek(proc_off); f.read(4)
                manifest_bytes = f.read(struct.unpack('<I', f.read(4))[0])
                vm_resources = {}
                if "vocab.b" in tokenizer_resources_raw:
                    content = tokenizer_resources_raw['vocab.b']
                    num_entries, = struct.unpack_from('<I', content, 0)
                    offset = 4 + num_entries * 4 
                    vocab = {}
                    for _ in range(num_entries):
                        key_len, = struct.unpack_from('<H', content, offset); offset += 2
                        key = content[offset : offset + key_len].decode('utf-8'); offset += key_len
                        val, _ = struct.unpack_from('<if', content, offset); offset += 8
                        vocab[key] = val
                    vm_resources['vocab'] = vocab
                    if "merges.b" in tokenizer_resources_raw:
                        content = tokenizer_resources_raw['merges.b']
                        num_entries, = struct.unpack_from('<I', content, 0)
                        offset = 4
                        ranks = {}
                        for i in range(num_entries):
                            p1_len, = struct.unpack_from('<H', content, offset); offset += 2
                            p1 = content[offset : offset + p1_len].decode('utf-8'); offset += p1_len
                            p2_len, = struct.unpack_from('<H', content, offset); offset += 2
                            p2 = content[offset : offset + p2_len].decode('utf-8'); offset += p2_len
                            ranks[(p1, p2)] = i
                        vm_resources['ranks'] = ranks
                else:
                    if "tokenizer.json" in tokenizer_resources_raw:
                        tok_json = json.loads(tokenizer_resources_raw["tokenizer.json"].decode('utf-8'))
                        model_data = tok_json.get('model', {})
                        if model_data.get('type') == 'Unigram':
                            vocab_list = model_data.get('vocab', [])
                            vm_resources['vocab'] = {token: i for i, (token, score) in enumerate(vocab_list)}
                        elif 'vocab' in model_data: vm_resources['vocab'] = model_data['vocab']
                        if "merges" in model_data: vm_resources['ranks'] = {tuple(line.split()): i for i, line in enumerate(model_data["merges"])}
                    elif "vocab.json" in tokenizer_resources_raw:
                        vm_resources['vocab'] = json.loads(tokenizer_resources_raw["vocab.json"].decode('utf-8'))
                    if "merges.txt" in tokenizer_resources_raw and 'ranks' not in vm_resources:
                        lines = [line for line in tokenizer_resources_raw["merges.txt"].decode('utf-8').splitlines() if line and not line.startswith("#")]
                        vm_resources['ranks'] = {tuple(line.split()): i for i, line in enumerate(lines)}
                
                if vm_resources.get('vocab'):
                    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
                    cs = bs[:]; n = 0
                    for b in range(256):
                        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
                    vm_resources['byte_map'] = dict(zip(bs, [chr(c) for c in cs]))
                    vm_resources.setdefault('ranks', {}); vm_resources.setdefault('unigram_scores', {})
                    self.tokenizer = TISAVM(vm_resources)
                    self.tokenizer.manifest = manifest_bytes
            
            self.trng_operations: List[Dict] = []
            if trng_off > 0:
                f.seek(trng_off)
                if f.read(4) != b'TRNG': raise ValueError("Bad TRNG magic")
                self.trng_operations = [self._read_op(f) for _ in range(struct.unpack('<I', f.read(4))[0])]

    def _dequantize(self, arr: np.ndarray, metadata: Dict) -> np.ndarray:
        q_type = metadata.get('quant_type')
        if q_type == 'INT8_TENSOR': return arr.astype(np.float32) * metadata.get('scale', 1.0)
        if q_type == 'INT8_CHANNEL':
            scales = np.array(metadata['scales'], dtype=np.float32)
            shape = [1] * arr.ndim
            shape[metadata.get('axis', 0)] = -1
            return arr.astype(np.float32) * scales.reshape(shape)
        if q_type == 'BLOCK_FP8':
            block_size = metadata['block_size']
            original_shape = metadata['original_shape']
            scales = np.array(metadata['scales'], dtype=np.float32)
            arr_fp32 = arr.astype(np.float32)
            num_blocks = arr_fp32.shape[0] // block_size
            blocked_tensor = arr_fp32.reshape(num_blocks, block_size)
            dequantized_blocks = blocked_tensor * scales.reshape(-1, 1)
            flat_tensor = dequantized_blocks.flatten()
            num_original_elements = np.prod(original_shape) if original_shape else 0
            truncated_tensor = flat_tensor[:num_original_elements]
            return truncated_tensor.reshape(original_shape)
        if arr.dtype == np.float16: return arr.astype(np.float32)
        return arr

    def _quantize_for_saving(self, arr: np.ndarray, meta: dict) -> bytes:
        q_type = meta.get('quant_type', 'none')
        if q_type == 'none': return arr.astype(np.float32).tobytes()
        if q_type == 'FP16': return arr.astype(np.float16).tobytes()
        if q_type == 'INT8_TENSOR': return np.round(arr / meta.get('scale', 1.0)).clip(-128, 127).astype(np.int8).tobytes()
        if q_type == 'INT8_CHANNEL':
            scales = np.array(meta['scales'], dtype=np.float32)
            shape = [1] * arr.ndim
            shape[meta.get('axis', 0)] = -1
            return np.round(arr / scales.reshape(shape)).clip(-128, 127).astype(np.int8).tobytes()
        if q_type == 'BLOCK_FP8':
            block_size = meta['block_size']
            scales = np.array(meta['scales'], dtype=np.float32)
            arr_flat = arr.flatten().astype(np.float32)
            rem = arr_flat.size % block_size
            if rem != 0: arr_flat = np.concatenate([arr_flat, np.zeros(block_size - rem, dtype=np.float32)])
            blocks = arr_flat.reshape(-1, block_size)
            q = np.round(blocks / scales.reshape(-1, 1)).clip(-128, 127).astype(np.int8)
            return q.flatten().tobytes()
        return arr.tobytes()

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

    def zero_grad(self):
        pass # self.results and cache will be recreated per run_training_step

    def save_weights(self, save_path: str = None, save_type: int = 0):
        tensors = {self.param_id_to_name.get(pid, f"param_{pid}"): np.asarray(arr, dtype=np.float32) for pid, arr in self.parameters.items()}
        if not tensors:
            print("[NAC] save_weights: nothing to save."); return

        if not self.weights_stored_internally:
            # Внешние веса - просто пишем в .safetensors
            try:
                from safetensors.numpy import save_file as _st_save
                companion = os.path.splitext(self._nac_path)[0] + ".safetensors"
                _st_save(tensors, companion)
                print(f"[NAC] External weights updated in '{companion}'")
            except ImportError:
                print("[NAC] Error: pip install safetensors")
        else:
            # Внутренние веса - умное in-place обновление с расширением секций
            print("[NAC] Analyzing internal weights for structural changes...")
            try:
                with open(self._nac_path, 'rb') as f:
                    f.seek(10)
                    header_bytes = f.read(82)
                    unpacked = struct.unpack('<H10Q', header_bytes)
                    d_model = unpacked[0]
                    offsets = list(unpacked[1:])
                    data_off = offsets[5] # mmap, ops, cmap, cnst, perm, DATA, proc, orch, trng, rsrc

                    if data_off == 0:
                        raise ValueError("No DATA section found in .nac")

                    # Находим начало блока тензоров внутри секции DATA
                    f.seek(data_off)
                    if f.read(4) != b'DATA': raise ValueError("Invalid DATA offset")
                    num_p = struct.unpack('<I', f.read(4))[0]
                    for _ in range(num_p):
                        f.seek(struct.unpack('<HH', f.read(4))[1], 1)
                    num_in = struct.unpack('<I', f.read(4))[0]
                    for _ in range(num_in):
                        f.seek(struct.unpack('<HH', f.read(4))[1], 1)
                    
                    tensors_start_offset = f.tell()

                    # Находим конец блока тензоров (начало следующей секции)
                    next_offsets = [off for off in offsets if off > data_off]
                    tensors_end_offset = min(next_offsets) if next_offsets else os.path.getsize(self._nac_path)

                    # Читаем чанки файла ДО и ПОСЛЕ тензоров
                    f.seek(0)
                    chunk_before = bytearray(f.read(tensors_start_offset))
                    
                    f.seek(tensors_end_offset)
                    chunk_after = f.read()

                # Формируем новый бинарный блок тензоров
                tensor_block_bytes = bytearray()
                tensor_block_bytes.extend(struct.pack('<I', len(self.parameters)))

                def get_dtype_enum(arr):
                    if arr.dtype == np.float32: return 0
                    if arr.dtype == np.float64: return 1
                    if arr.dtype == np.float16: return 2
                    if arr.dtype == np.int32: return 4
                    if arr.dtype == np.int64: return 5
                    if arr.dtype == np.int16: return 6
                    if arr.dtype == np.int8: return 7
                    if arr.dtype == np.uint8: return 8
                    if arr.dtype == np.bool_: return 9
                    return 0

                for pid, arr in self.parameters.items():
                    meta = self.tensor_offsets.get(pid, {}).get('meta', {'quant_type': 'none'})
                    q_type = meta.get('quant_type', 'none')
                    
                    saved_shape = arr.shape
                    saved_dtype = 0 # По умолчанию float32
                    
                    # Динамическая переквантизация при изменении размера
                    if q_type == 'none':
                        q_bytes = arr.astype(np.float32).tobytes()
                        saved_dtype = 0 # float32
                    elif q_type == 'FP16':
                        q_bytes = arr.astype(np.float16).tobytes()
                        saved_dtype = 2 # float16
                    elif q_type == 'INT8_TENSOR':
                        scale = float(np.max(np.abs(arr)) / 127.0)
                        if scale == 0: scale = 1e-9
                        meta['scale'] = scale
                        q_bytes = np.round(arr / scale).clip(-128, 127).astype(np.int8).tobytes()
                        saved_dtype = 7 # int8
                    elif q_type == 'INT8_CHANNEL':
                        axis = meta.get('axis', 0)
                        dims = tuple(i for i in range(arr.ndim) if i != axis)
                        scale_arr = np.max(np.abs(arr), axis=dims, keepdims=True) / 127.0
                        scale_arr[scale_arr == 0] = 1e-9
                        meta['scales'] = scale_arr.flatten().tolist()
                        q_bytes = np.round(arr / scale_arr).clip(-128, 127).astype(np.int8).tobytes()
                        saved_dtype = 7 # int8
                    elif q_type == 'BLOCK_FP8':
                        block_size = meta.get('block_size', 64)
                        arr_flat = arr.flatten().astype(np.float32)
                        rem = arr_flat.size % block_size
                        if rem != 0:
                            arr_flat = np.concatenate([arr_flat, np.zeros(block_size - rem, dtype=np.float32)])
                        blocks = arr_flat.reshape(-1, block_size)
                        scale_arr = np.max(np.abs(blocks), axis=1, keepdims=True) / 127.0
                        scale_arr[scale_arr == 0] = 1e-9
                        meta['scales'] = scale_arr.flatten().tolist()
                        meta['original_shape'] = list(arr.shape)
                        
                        q_arr = np.round(blocks / scale_arr).clip(-128, 127).astype(np.int8).flatten()
                        q_bytes = q_arr.tobytes()
                        saved_shape = q_arr.shape # BLOCK_FP8 сохраняется как плоский 1D массив!
                        saved_dtype = 7 # int8
                    else:
                        q_bytes = arr.tobytes()
                        saved_dtype = get_dtype_enum(arr)

                    # Собираем мета-данные с ПРАВИЛЬНЫМИ типами и размерами
                    meta_binary = bytearray()
                    meta_binary.extend(struct.pack('<BB', saved_dtype, len(saved_shape)))
                    if len(saved_shape) > 0:
                        meta_binary.extend(struct.pack(f'<{len(saved_shape)}I', *saved_shape))

                    quant_code = {'none':0, 'FP16':1, 'INT8_TENSOR':2, 'INT8_CHANNEL':3, 'BLOCK_FP8':4}.get(q_type, 0)
                    meta_binary.extend(struct.pack('<B', quant_code))

                    if quant_code == 2:
                        meta_binary.extend(struct.pack('<f', meta['scale']))
                    elif quant_code == 3:
                        s_list = meta['scales']
                        meta_binary.extend(struct.pack('<BI', meta.get('axis', 0), len(s_list)))
                        meta_binary.extend(struct.pack(f'<{len(s_list)}f', *s_list))
                    elif quant_code == 4:
                        orig_shape = meta['original_shape']
                        meta_binary.extend(struct.pack('<HB', meta.get('block_size', 64), len(orig_shape)))
                        if len(orig_shape) > 0:
                            meta_binary.extend(struct.pack(f'<{len(orig_shape)}I', *orig_shape))
                        s_list = meta['scales']
                        meta_binary.extend(struct.pack('<I', len(s_list)))
                        meta_binary.extend(struct.pack(f'<{len(s_list)}f', *s_list))

                    meta_bytes = bytes(meta_binary)
                    tensor_block_bytes.extend(struct.pack('<HIQ', pid, len(meta_bytes), len(q_bytes)))
                    tensor_block_bytes.extend(meta_bytes)
                    tensor_block_bytes.extend(q_bytes)

                # Вычисляем разницу в размере
                diff = len(tensor_block_bytes) - (tensors_end_offset - tensors_start_offset)

                if diff != 0:
                    print(f"[NAC] Model architecture changed. Expanding .nac sections (Delta: {diff} bytes).")
                    # Сдвигаем смещения в заголовке для секций, идущих ПОСЛЕ DATA
                    new_offsets = []
                    for off in offsets:
                        if off > data_off:
                            new_offsets.append(off + diff)
                        else:
                            new_offsets.append(off)
                    
                    new_header_bytes = struct.pack('<H10Q', d_model, *new_offsets)
                    chunk_before[10:10+82] = new_header_bytes
                else:
                    print("[NAC] Dimensions unchanged. Performing fast in-place update.")

                # Перезаписываем файл монолитно
                with open(self._nac_path, 'wb') as f:
                    f.write(chunk_before)
                    f.write(tensor_block_bytes)
                    f.write(chunk_after)
                
                print(f"[NAC] Successfully saved weights directly into '{self._nac_path}'")

            except Exception as e:
                import traceback
                print(f"[NAC] Binary update failed: {e}")
                traceback.print_exc()

        # Дополнительный бэкап, если запрошен извне
        if save_path and save_path != "":
            companion = os.path.splitext(self._nac_path)[0] + ".safetensors"
            if os.path.abspath(save_path) != os.path.abspath(companion) and os.path.abspath(save_path) != os.path.abspath(self._nac_path):
                try:
                    from safetensors.numpy import save_file as _st_save
                    _st_save(tensors, save_path)
                    print(f"[NAC] Additional weights backup saved to '{save_path}'")
                except ImportError:
                    pass

    def run_training_step(self, inputs: List[np.ndarray], targets: List[np.ndarray],
                          loss_type: int = 0, lr: float = 0.001,
                          logits=None,
                          head_weight_name: str = "",
                          head_bias_name: str = "") -> np.ndarray:
        """
        Универсальный training step с авто-расширением и надежным поиском активаций.
        """
        if not self.trng_operations:
            return np.float32(0.0)

        print(f"[TRNG] Universal training | loss_type={loss_type}, lr={lr:.6f}, head_weight='{head_weight_name or 'None'}'")

        # 1. Полный forward через OPS
        self.results = [None] * len(self.operations)
        import itertools
        self.input_stream = itertools.cycle([np.asarray(x) for x in inputs]) 

        try:
            fw_output = self._execute_block(0, len(self.operations), verbose=False)
        except Exception as e:
            print(f"[TRNG] Forward pass failed: {e}")
            fw_output = None
            for res in reversed(self.results):
                if isinstance(res, np.ndarray) and res.size > 0:
                    fw_output = res
                    break

        # 2. Извлечение logits
        if logits is None:
            if isinstance(fw_output, np.ndarray) and fw_output.ndim >= 2:
                logits = fw_output.astype(np.float32)
            elif isinstance(fw_output, (list, tuple)) and fw_output:
                logits = np.asarray(fw_output[0], dtype=np.float32)
            else:
                logits = None
                for i in range(len(self.results)-1, -1, -1):
                    if isinstance(self.results[i], np.ndarray) and self.results[i].ndim >= 2 and self.results[i].shape[-1] > 10:
                        logits = self.results[i].astype(np.float32)
                        break
                if logits is None:
                    print("[TRNG] ERROR: Could not find logits after forward pass")
                    return np.float32(0.0)

            if logits.ndim == 3:
                logits = logits.reshape(-1, logits.shape[-1])

        current_output_size = logits.shape[-1] if logits.ndim >= 2 else 0

        # 3. Авто-расширение по target
        target = np.asarray(targets[0]).flatten().astype(np.int64)
        max_target = int(target.max()) if len(target) > 0 else 0
        expand_dim = max(0, max_target - current_output_size + 1)

        if expand_dim > 0:
            if head_weight_name:
                print(f"[TRNG] Auto-expanding '{head_weight_name}' from {current_output_size} to {current_output_size + expand_dim}")
                for pid, name in list(self.param_id_to_name.items()):
                    if name == head_weight_name and pid in self.parameters:
                        param = self.parameters[pid]
                        if param.ndim == 2:
                            new_weight = np.zeros((current_output_size + expand_dim, param.shape[1]), dtype=param.dtype)
                            new_weight[:param.shape[0]] = param
                            self.parameters[pid] = new_weight
                            break

                if head_bias_name:
                    for pid, name in list(self.param_id_to_name.items()):
                        if name == head_bias_name and pid in self.parameters:
                            param = self.parameters[pid]
                            if param.ndim == 1:
                                # Исправлено: добавлено + expand_dim для bias
                                new_bias = np.zeros(current_output_size + expand_dim, dtype=param.dtype)
                                new_bias[:len(param)] = param
                                self.parameters[pid] = new_bias
                                break
            
            # --- ИСПРАВЛЕНИЕ: Расширяем массив logits нулями для новых классов ---
            pad_width = [(0, 0)] * logits.ndim
            pad_width[-1] = (0, expand_dim)
            logits = np.pad(logits, pad_width, mode='constant', constant_values=0.0)
            
            current_output_size += expand_dim

        target = np.clip(target, 0, current_output_size - 1)

        # 4. Вычисление Loss и градиентов
        if loss_type == 0:
            logits_max = logits.max(axis=-1, keepdims=True)
            log_sm = logits - logits_max
            log_sm = log_sm - np.log(np.exp(log_sm).sum(axis=-1, keepdims=True))
            loss_scalar = -np.mean(log_sm[np.arange(len(target)), target])
            grad_logits = np.exp(log_sm)
            grad_logits[np.arange(len(target)), target] -= 1.0
            grad_logits /= len(target)
        else:
            flat_target = np.zeros_like(logits)
            flat_target[np.arange(len(target)), target] = 1.0
            loss_scalar = np.mean((logits - flat_target) ** 2)
            grad_logits = 2.0 * (logits - flat_target) / logits.size

        print(f"[TRNG] Loss: {loss_scalar:.6f} | Output size: {current_output_size}")

        # 5. Обновление параметров
        updated = 0
        batch_size = grad_logits.shape[0]

        if head_weight_name:
            for pid, name in self.param_id_to_name.items():
                if name == head_weight_name and pid in self.parameters:
                    param = self.parameters[pid]
                    if param.ndim == 2:
                        in_features = param.shape[1]
                        input_act = None
                        
                        # Требуем точное совпадение размера: (batch_size * in_features)
                        for i in range(len(self.results)-1, -1, -1):
                            if self.results[i] is not None:
                                arr = np.asarray(self.results[i])
                                if arr.size == batch_size * in_features and arr.shape[0] == batch_size:
                                    input_act = arr.reshape(batch_size, in_features)
                                    break
                                    
                        if input_act is not None:
                            grad_w = np.dot(grad_logits.T, input_act)
                            self.parameters[pid] = (param - lr * grad_w).astype(param.dtype)
                            print(f"[TRNG] Updated weights '{head_weight_name}' | grad norm: {np.linalg.norm(grad_w):.4f}")
                            updated += 1
                        else:
                            print(f"[TRNG] WARNING: Could not find input activation for '{head_weight_name}'!")

        if head_bias_name:
            for pid, name in self.param_id_to_name.items():
                if name == head_bias_name and pid in self.parameters:
                    param = self.parameters[pid]
                    if param.ndim == 1:
                        grad_b = np.sum(grad_logits, axis=0)
                        self.parameters[pid] = (param - lr * grad_b).astype(param.dtype)
                        print(f"[TRNG] Updated bias '{head_bias_name}' | grad norm: {np.linalg.norm(grad_b):.4f}")
                        updated += 1

        return np.float32(loss_scalar)

    def run(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        # Очищаем кэш результатов для чистого прогона
        self.results = [None] * len(self.operations)
        if not self._is_initialized:
            print("\n--- Starting Model Execution ---")
            self._is_initialized = True
            self.perf_stats = {} 
        import itertools
        self.input_stream = itertools.cycle(inputs)
        # Если это первый запуск, выводим подробности
        verbose = len(self.perf_stats) == 0
        res = self._execute_block(0, len(self.operations), verbose=verbose)
        if verbose:
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
        all_offsets = sorted(branch_offsets + [jump_offset_after_branches])
        branch_definitions = []
        for i in range(num_branches):
            start_offset = branch_offsets[i]
            end_offset = all_offsets[all_offsets.index(start_offset) + 1]
            branch_definitions.append({"start": current_ip + start_offset, "len": end_offset - start_offset})
        branch_results = [res for branch in branch_definitions if (res := self._execute_block(branch['start'], branch['len'])) is not None]
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
            center_of_mass = np.mean(np.stack(padded_main_tensors, axis=0), axis=0)
            distances = [np.linalg.norm(t - center_of_mass) for t in padded_main_tensors]
            return branch_results[np.argmin(distances)]
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
        if len(d_values) != len(perm): raise ValueError(f"Instruction {current_idx}: Mismatch between perm length ({len(perm)}) and D field length ({len(d_values)}).")
        for i in range(len(perm)):
            d_val = d_values[i]
            if d_val != 0:
                ancestor_idx = current_idx + d_val
                args.append(self.results[ancestor_idx])
            else:
                try: args.append(self.constants.get(next(c_iter)))
                except StopIteration: raise ValueError(f"Instruction {current_idx}: D field expects a constant, but C field is exhausted.")
        return args

    def patch_instruction_by_ip(self, target_ip: int, new_instruction_bytes: bytes, permanent: bool = False) -> bool:
        """
        Продвинутый API для замены инструкции по её абсолютному адресу (IP).
        Адреса можно узнать через утилиту NAC_info.py.
        """
        if not hasattr(self, '_orch_off') or self._orch_off == 0:
            print("[NAC] Error: No MEP orchestration found in this file.")
            return False

        bytecode, _ = self.get_mep_plan()
        if not bytecode:
            return False

        if target_ip < 0 or target_ip >= len(bytecode):
            print(f"[NAC] Patch failed: Target IP 0x{target_ip:04X} is out of bounds (max 0x{len(bytecode)-1:04X}).")
            return False

        from MEP_compiler import MEPPatcher
        
        # Проверяем, действительно ли по этому адресу начинается валидная инструкция
        # (Опциональная защита, но полезная: проверяем, что мы не попали в середину операнда)
        offset = 0
        is_valid_start = False
        while offset < len(bytecode):
            if offset == target_ip:
                is_valid_start = True
                break
            length = MEPPatcher.instruction_length(bytecode, offset)
            offset += length
            
        if not is_valid_start:
            print(f"[NAC] Warning: IP 0x{target_ip:04X} might be inside another instruction's arguments. Proceed with caution!")

        # Проверка длины
        old_length = MEPPatcher.instruction_length(bytecode, target_ip)
        new_length = len(new_instruction_bytes)
        
        if old_length != new_length:
            print(f"[NAC] Patch failed: Length mismatch! Old is {old_length} bytes, new is {new_length} bytes.")
            return False

        # Замена в памяти
        patched_bytecode = bytearray(bytecode)
        patched_bytecode[target_ip : target_ip + new_length] = new_instruction_bytes
        self._patched_bytecode_cache = bytes(patched_bytecode)

        print(f"[NAC] Instruction at 0x{target_ip:04X} successfully patched in memory.")

        # Перезапись в файл
        if permanent:
            try:
                with open(self._nac_path, 'r+b') as f:
                    # Смещение в файле: начало ORCH + 4(магия) + 4(len) + 4(consts) + target_ip
                    exact_file_pos = self._orch_off + 12 + target_ip
                    f.seek(exact_file_pos)
                    f.write(new_instruction_bytes)
                print(f"[NAC] Patch at 0x{target_ip:04X} permanently saved to '{self._nac_path}'.")
            except Exception as e:
                print(f"[NAC] Permanent patch failed: {e}")
                return False

        return True

    def get_mep_plan(self):
        """
        Возвращает байткод (с учетом патчей в памяти) и константы.
        """
        if not getattr(self, '_orch_off', 0): return None, None
        
        with open(self._nac_path, 'rb') as f:
            f.seek(self._orch_off)
            if f.read(4) != b'ORCH': raise ValueError("Bad ORCH magic")
            blen, clen = struct.unpack('<II', f.read(8))
            if blen == 0: return None, None
            
            # Если есть пропатченный байткод в памяти, берем его. 
            # Иначе читаем оригинальный из файла.
            if hasattr(self, '_patched_bytecode_cache') and self._patched_bytecode_cache:
                bytecode = self._patched_bytecode_cache
                f.seek(blen, 1) # Пропускаем байткод в файле, чтобы дойти до констант
            else:
                bytecode = f.read(blen)
                
            constants = {}
            for _ in range(clen):
                const_id, type_code, length = struct.unpack('<HBH', f.read(5))
                if type_code == 1: val = struct.unpack('<?', f.read(length))[0]
                elif type_code == 2: val = struct.unpack('<q', f.read(length))[0]
                elif type_code == 3: val = struct.unpack('<d', f.read(length))[0]
                elif type_code == 4: val = f.read(length).decode('utf-8')
                elif type_code == 5: val = list(struct.unpack(f'<{length}i', f.read(length * 4))) if length > 0 else []
                elif type_code == 6: val = list(struct.unpack(f'<{length}f', f.read(length * 4))) if length > 0 else []
                else: val = None
                constants[const_id] = val
                
        return bytecode, constants

'''
import struct
from NAC_run import NacRuntime

# Загружаем рантайм
probe = NacRuntime("distilbert-sst2-sentiment.nac")

# Мы хотим заменить argmax (0x62, op 0x00) на argmin, или просто на логическое сравнение.
# Важно: новая инструкция должна быть ровно 4 байта.
# Например, заменим на MATH_UNARY (0x60, op=0x00, те же ключи)
new_instr = struct.pack('<BBBB', 0x60, 0x00, 0x0F, 0x0E)

# Применяем патч по абсолютному адресу 0x004F!
probe.patch_instruction_by_ip(target_ip=0x004F, new_instruction_bytes=new_instr, permanent=False)

# Теперь запускаем модель
# bytecode, constants = probe.get_mep_plan()
# ... передаем в MEPInterpreter и выполняем ...
'''

# --- END OF FILE NAC_run.py ---