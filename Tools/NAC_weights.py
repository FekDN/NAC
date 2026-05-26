# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

import os
import struct
import argparse
import json
import math
import glob
import collections
import torch
from safetensors.torch import save_file, load_file
from safetensors import safe_open

# =============================================================================
# NAC DType Maps (NAC v1.8 Spec §6.1.1)
# =============================================================================
NAC_TO_TORCH_DTYPE = {
    0: torch.float32, 1: torch.float64, 2: torch.float16, 3: torch.bfloat16,
    4: torch.int32,   5: torch.int64,   6: torch.int16,   7: torch.int8,
    8: torch.uint8,   9: torch.bool
}

TORCH_TO_NAC_DTYPE = {v: k for k, v in NAC_TO_TORCH_DTYPE.items()}

# =============================================================================
# Binary helpers
# =============================================================================
def read_string(f, length_format='<H'):
    length = struct.unpack(length_format, f.read(struct.calcsize(length_format)))[0]
    return f.read(length).decode('utf-8')

def write_string(f, string, length_format='<H'):
    encoded = string.encode('utf-8')
    f.write(struct.pack(length_format, len(encoded)))
    f.write(encoded)

# =============================================================================
# Quantization metadata parsers
# =============================================================================
def unpack_quant_meta(meta_bytes):
    dtype_id = meta_bytes[0]
    rank = meta_bytes[1]
    shape = tuple(struct.unpack(f'<{rank}I', meta_bytes[2:2 + rank*4]))
    pos = 2 + rank * 4
    quant_code = meta_bytes[pos]
    pos += 1
    
    quant_type_map = {0: 'none', 1: 'FP16', 2: 'INT8_TENSOR', 3: 'INT8_CHANNEL', 4: 'BLOCK_FP8'}
    q_meta = {
        "quant_code": quant_code,
        "quant_type": quant_type_map.get(quant_code, 'none')
    }
    
    if quant_code == 2:  # INT8_TENSOR
        q_meta["scale"] = struct.unpack('<f', meta_bytes[pos:pos+4])[0]
    elif quant_code == 3:  # INT8_CHANNEL
        q_meta["axis"] = meta_bytes[pos]
        num_scales = struct.unpack('<I', meta_bytes[pos+1:pos+5])[0]
        pos += 5
        q_meta["scales"] = list(struct.unpack(f'<{num_scales}f', meta_bytes[pos:pos+num_scales*4]))
    elif quant_code == 4:  # BLOCK_FP8
        q_meta["block_size"] = struct.unpack('<H', meta_bytes[pos:pos+2])[0]
        orig_rank = meta_bytes[pos+2]
        pos += 3
        q_meta["orig_shape"] = list(struct.unpack(f'<{orig_rank}I', meta_bytes[pos:pos+orig_rank*4]))
        pos += orig_rank * 4
        num_blocks = struct.unpack('<I', meta_bytes[pos:pos+4])[0]
        pos += 4
        q_meta["block_scales"] = list(struct.unpack(f'<{num_blocks}f', meta_bytes[pos:pos+num_blocks*4]))

    return dtype_id, shape, q_meta

def pack_quant_meta(dtype_id, shape, q_meta):
    rank = len(shape)
    quant_code = q_meta.get("quant_code")
    if quant_code is None:
        str_to_code = {'none':0, 'FP16':1, 'INT8_TENSOR':2, 'INT8_CHANNEL':3, 'BLOCK_FP8':4}
        quant_code = str_to_code.get(q_meta.get("quant_type", "none"), 0)
    
    b = struct.pack('<BB', dtype_id, rank)
    if rank > 0:
        b += struct.pack(f'<{rank}I', *shape)
    b += struct.pack('<B', quant_code)
    
    if quant_code == 2:
        b += struct.pack('<f', q_meta["scale"])
    elif quant_code == 3:
        scales = q_meta["scales"]
        b += struct.pack('<BI', q_meta["axis"], len(scales))
        b += struct.pack(f'<{len(scales)}f', *scales)
    elif quant_code == 4:
        orig_shape = q_meta["orig_shape"]
        block_scales = q_meta["block_scales"]
        b += struct.pack('<HB', q_meta["block_size"], len(orig_shape))
        if len(orig_shape) > 0:
            b += struct.pack(f'<{len(orig_shape)}I', *orig_shape)
        b += struct.pack('<I', len(block_scales))
        b += struct.pack(f'<{len(block_scales)}f', *block_scales)
    return b

# =============================================================================
# Main class of instrument
# =============================================================================
class NacWeightTool:
    def __init__(self, nac_path):
        self.nac_path = nac_path
        self.base_name = os.path.splitext(self.nac_path)[0]
        self.header_size = 100
        
    def _parse_header(self, f):
        f.seek(0)
        header_bytes = f.read(self.header_size)
        if header_bytes[:3] != b"NAC":
            raise ValueError("Invalid NAC file: Magic bytes not found.")
            
        quant_flags = header_bytes[4]
        is_internal = (quant_flags & 0x80) != 0
        offsets = list(struct.unpack('<11Q', header_bytes[12:100]))
        return header_bytes, is_internal, quant_flags, offsets

    def _write_header(self, f, header_bytes, quant_flags, offsets):
        f.seek(0)
        f.write(header_bytes[:4])
        f.write(struct.pack('<B', quant_flags))
        f.write(header_bytes[5:12])
        f.write(struct.pack('<11Q', *offsets))

    def extract(self, shards=1):
        """Extracts weights from .nac and automatically stores them in .safetensors (with optional sharding)."""
        with open(self.nac_path, 'rb') as f:
            header_bytes, is_internal, quant_flags, offsets = self._parse_header(f)
            if not is_internal:
                print("Weights are already external. Nothing to extract.")
                return

            data_offset = offsets[5]
            if data_offset == 0:
                raise ValueError("DATA section not found in NAC file.")

            f.seek(data_offset)
            if f.read(4) != b'DATA':
                raise ValueError("Expected 'DATA' tag")

            num_params = struct.unpack('<I', f.read(4))[0]
            id_to_name = {}
            for _ in range(num_params):
                pid = struct.unpack('<H', f.read(2))[0]
                name = read_string(f)
                id_to_name[pid] = name

            num_inputs = struct.unpack('<I', f.read(4))[0]
            for _ in range(num_inputs):
                f.read(2); read_string(f)

            end_of_block2_offset = f.tell()

            num_tensors = struct.unpack('<I', f.read(4))[0]
            tensors_dict = {}
            quant_metadata_dict = {}
            
            for _ in range(num_tensors):
                pid = struct.unpack('<H', f.read(2))[0]
                meta_len = struct.unpack('<I', f.read(4))[0]
                data_len = struct.unpack('<Q', f.read(8))[0]
                
                meta_bytes = f.read(meta_len)
                data_bytes = f.read(data_len)
                
                if pid not in id_to_name:
                    continue
                    
                name = id_to_name[pid]
                dtype_id, shape, q_meta = unpack_quant_meta(meta_bytes)
                mutable_bytes = bytearray(data_bytes)
                
                if q_meta["quant_code"] in (2, 3, 4):
                    tensor = torch.frombuffer(mutable_bytes, dtype=torch.int8).reshape(shape).clone()
                    quant_metadata_dict[name] = q_meta
                else:
                    torch_dtype = NAC_TO_TORCH_DTYPE[dtype_id]
                    tensor = torch.frombuffer(mutable_bytes, dtype=torch_dtype).reshape(shape).clone()
                    
                    if torch_dtype == torch.bfloat16:
                        tensor = tensor.to(torch.float32)
                
                # Due to weight-tying, multiple PIDs can have the same name, but the data is identical.
                tensors_dict[name] = tensor

            end_of_data_offset = f.tell()
            bytes_to_remove = end_of_data_offset - end_of_block2_offset

            keys = list(tensors_dict.keys())
            if shards <= 1:
                st_metadata = {"nac_quant_meta": json.dumps(quant_metadata_dict)} if quant_metadata_dict else None
                out_path = f"{self.base_name}.safetensors"
                save_file(tensors_dict, out_path, metadata=st_metadata)
                print(f"Extracted {len(tensors_dict)} tensors to {out_path}")
            else:
                chunk_size = math.ceil(len(keys) / shards)
                for i in range(shards):
                    chunk_keys = keys[i*chunk_size : (i+1)*chunk_size]
                    chunk_dict = {k: tensors_dict[k] for k in chunk_keys}
                    chunk_q_meta = {k: quant_metadata_dict[k] for k in chunk_keys if k in quant_metadata_dict}
                    
                    st_metadata = {"nac_quant_meta": json.dumps(chunk_q_meta)} if chunk_q_meta else None
                    out_path = f"{self.base_name}-{i+1:05d}-of-{shards:05d}.safetensors"
                    
                    save_file(chunk_dict, out_path, metadata=st_metadata)
                    print(f"Saved shard {i+1}/{shards} ({len(chunk_dict)} tensors) to {out_path}")

            f.seek(end_of_data_offset)
            rest_of_file = f.read()

        with open(self.nac_path, 'r+b') as f:
            new_quant_flags = quant_flags & ~0x80
            new_offsets = [off - bytes_to_remove if off > data_offset else off for off in offsets]
            
            self._write_header(f, header_bytes, new_quant_flags, new_offsets)
            f.seek(end_of_block2_offset)
            f.truncate()
            f.write(rest_of_file)
            
        print(f"NAC file optimized. Weights are now external.")

    def inject(self):
        """Injects weights from .safetensors into .nac (in-place)."""
    
        safetensors_paths = []
        single_file = f"{self.base_name}.safetensors"
        if os.path.exists(single_file):
            safetensors_paths.append(single_file)
        else:
            pattern = f"{self.base_name}-*-of-*.safetensors"
            safetensors_paths = sorted(glob.glob(pattern))

        if not safetensors_paths:
            raise FileNotFoundError(f"No external weights found for '{self.nac_path}'.")

        combined_state_dict = {}
        combined_quant_meta = {}
        for st_path in safetensors_paths:
            print(f"Loading '{st_path}'...")
            combined_state_dict.update(load_file(st_path))
            with safe_open(st_path, framework="pt") as f:
                st_meta = f.metadata()
                if st_meta and "nac_quant_meta" in st_meta:
                    combined_quant_meta.update(json.loads(st_meta["nac_quant_meta"]))
        print(f"Total unique tensors loaded: {len(combined_state_dict)}")

        with open(self.nac_path, 'rb') as f:
            header_bytes, is_internal, quant_flags, offsets = self._parse_header(f)
            data_offset = offsets[5]
        
            f.seek(data_offset)
            if f.read(4) != b'DATA':
                raise ValueError("Corrupted DATA section.")

            num_params = struct.unpack('<I', f.read(4))[0]
            name_to_pids = collections.defaultdict(list)
            for _ in range(num_params):
                pid = struct.unpack('<H', f.read(2))[0]
                name = read_string(f)
                name_to_pids[name].append(pid)

            num_inputs = struct.unpack('<I', f.read(4))[0]
            for _ in range(num_inputs):
                f.read(2); read_string(f)

            end_of_block2_offset = f.tell()
            expected_shapes = {}

            if is_internal:
                num_tensors_old = struct.unpack('<I', f.read(4))[0]
                for _ in range(num_tensors_old):
                    pid = struct.unpack('<H', f.read(2))[0]
                    m_len = struct.unpack('<I', f.read(4))[0]
                    d_len = struct.unpack('<Q', f.read(8))[0]
                    meta_bytes = f.read(m_len)
                    _, shape, _ = unpack_quant_meta(meta_bytes)
                    expected_shapes[pid] = shape
                    f.seek(d_len, os.SEEK_CUR)
                end_of_data_offset = f.tell()
                bytes_to_remove = end_of_data_offset - end_of_block2_offset
            else:
                end_of_data_offset = end_of_block2_offset
                bytes_to_remove = 0

            f.seek(0)
            prefix_data = f.read(end_of_block2_offset)
            f.seek(end_of_data_offset)
            rest_of_file = f.read()

        block3_data = bytearray()
        tensors_to_inject = []
    
        for name, pids in name_to_pids.items():
            if name in combined_state_dict:
                tensor = combined_state_dict[name]
                for pid in pids:
                    if is_internal and pid in expected_shapes:
                        if tuple(tensor.shape) != expected_shapes[pid]:
                            raise ValueError(
                                f"\n[FATAL] Shape mismatch for '{name}' (ID: {pid}).\n"
                                f"Expected: {expected_shapes[pid]}, Got: {tuple(tensor.shape)}"
                            )
                    tensors_to_inject.append((pid, name, tensor))
            else:
                print(f"[Warning] Tensor '{name}' (mapped to {len(pids)} PIDs) not found in safetensors, skipping.")

        tensors_to_inject.sort(key=lambda x: x[0])
        block3_data.extend(struct.pack('<I', len(tensors_to_inject)))
    
        for pid, name, tensor in tensors_to_inject:
            q_meta = combined_quant_meta.get(name, {})
            
            # Safe definition of quantization code
            quant_code = q_meta.get("quant_code")
            if quant_code is None:
                str_to_code = {'none':0, 'FP16':1, 'INT8_TENSOR':2, 'INT8_CHANNEL':3, 'BLOCK_FP8':4}
                quant_code = str_to_code.get(q_meta.get("quant_type", "none"), 0)
                q_meta["quant_code"] = quant_code
        
            if quant_code in (2, 3, 4):
                dtype_id = 7  # INT8 storage
            else:
                if tensor.dtype not in TORCH_TO_NAC_DTYPE:
                    raise ValueError(f"Unsupported dtype {tensor.dtype} for tensor '{name}'")
                dtype_id = TORCH_TO_NAC_DTYPE[tensor.dtype]

            meta_bytes = pack_quant_meta(dtype_id, tuple(tensor.shape), q_meta)
            data_bytes = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        
            block3_data.extend(struct.pack('<H', pid))
            block3_data.extend(struct.pack('<I', len(meta_bytes)))
            block3_data.extend(struct.pack('<Q', len(data_bytes)))
            block3_data.extend(meta_bytes)
            block3_data.extend(data_bytes)

        size_delta = len(block3_data) - bytes_to_remove
    
        with open(self.nac_path, 'wb') as f:
            new_quant_flags = quant_flags | 0x80
            new_offsets = [off + size_delta if off > data_offset else off for off in offsets]
            f.write(prefix_data)
            self._write_header(f, header_bytes, new_quant_flags, new_offsets)
            f.seek(end_of_block2_offset)
            f.write(block3_data)
            f.write(rest_of_file)

        print(f"Successfully injected {len(tensors_to_inject)} physical tensors into '{self.nac_path}'.")

        for st_path in safetensors_paths:
            try:
                os.remove(st_path)
                print(f"Removed '{st_path}' (weights are now internal).")
            except OSError as e:
                print(f"Notice: weights injected, but could not remove '{st_path}' (file may be locked by another process).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAC Weights Tool (v1.8 Spec) - Auto Discovery")
    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation", required=True)

    ext_parser = subparsers.add_parser("extract", help="Extract internal weights from .nac to .safetensors")
    ext_parser.add_argument("nac_file", type=str, help="Path to the source .nac file")
    ext_parser.add_argument("--shards", type=int, default=1, help="Number of safetensors shards to create (default: 1)")

    inj_parser = subparsers.add_parser("inject", help="Inject external .safetensors into .nac file in-place")
    inj_parser.add_argument("nac_file", type=str, help="Path to the target .nac file (safetensors are auto-discovered)")

    args = parser.parse_args()
    tool = NacWeightTool(args.nac_file)

    if args.mode == "extract":
        tool.extract(args.shards)
    elif args.mode == "inject":
        tool.inject()