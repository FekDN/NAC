# Copyright (c) 2025 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

import sys
import os
import struct
import json

from TISA_tokenizer import disassemble_TISA_manifest

def inspect_nac_file(filepath: str):
    """
    Reads a .nac file in the new robust binary format and prints its contents,
    including the tokenizer manifest if present.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return

    print(f"\n" + "="*20 + f" INSPECTING: {os.path.basename(filepath)} " + "="*20)

    with open(filepath, 'rb') as f:
        # --- 1. Read and Display Full Header ---
        print("\n--- Header Info ---")
        
        # --- Magic and Version ---
        magic = f.read(3)
        if magic != b'NAC':
            print(f"Error: Invalid NAC file format. Magic bytes mismatch. Found: {magic}")
            return
        version = struct.unpack('<B', f.read(1))[0]
        print(f"File Format: NAC v{version}")
        
        # --- Quantization and Storage ---
        quant_id = struct.unpack('<B', f.read(1))[0]
        weights_stored_internally = (quant_id & 0x80) > 0
        quant_method_id = quant_id & 0x7F
        quant_map = {0:'none', 1:'FP16', 2:'INT8_TENSOR', 3:'INT8_CHANNEL'}
        quant_method = quant_map.get(quant_method_id, f'UNKNOWN ({quant_method_id})')
        storage_method = "Internal" if weights_stored_internally else "External (.safetensors)"
        print(f"Quantization: {quant_method}")
        print(f"Weights Storage: {storage_method}")
        
        # --- IO Counts ---
        num_inputs, num_outputs, reserved = struct.unpack('<HHB', f.read(5))
        print(f"IO Counts: {num_inputs} Inputs, {num_outputs} Outputs")

        # --- Section Offsets ---
        offsets_header_format = '<H9Q6x'
        header_bytes = f.read(struct.calcsize(offsets_header_format))
        
        # Unpacking d_model and 9 offsets
        d_model, *offsets = struct.unpack(offsets_header_format, header_bytes)
        
        ops_offset, cmap_offset, cnst_offset, perm_offset, data_offset, \
        proc_offset, meta_offset, rsrc_offset, reserved2_offset = offsets
        
        print(f"Model Dimension (d_model): {d_model}")

        print("\n--- Section Offsets ---")
        print(f"  - OPS Section:      Starts at byte {ops_offset}")
        print(f"  - CMAP Section:     Starts at byte {cmap_offset}")
        print(f"  - CNST Section:     Starts at byte {cnst_offset}")
        print(f"  - PERM Section:     Starts at byte {perm_offset}")
        print(f"  - DATA Section:     Starts at byte {data_offset}")
        print(f"  - PROC Section:     Starts at byte {proc_offset} (Tokenizer Manifest)")
        print(f"  - RSRC Section:     Starts at byte {rsrc_offset} (Internal Resources)")
        print(f"  - META Section:     Starts at byte {meta_offset} (Reserved)")

        # --- 2. Read Operations Map (CMAP) ---
        print("\n--- Operations (CMAP) ---")
        if cmap_offset > 0:
            f.seek(cmap_offset)
            if f.read(4) != b'CMAP': print("Error: CMAP tag mismatch."); return
            
            num_ops = struct.unpack('<I', f.read(4))[0]
            print(f"Found {num_ops} operation mappings in this file:")
            
            ops_list = []
            for _ in range(num_ops):
                op_id, name_len = struct.unpack('<HB', f.read(3))
                op_name = f.read(name_len).decode('utf-8')
                ops_list.append((op_id, op_name))
            
            for op_id, op_name in sorted(ops_list):
                 print(f"  ID {op_id:<4}: {op_name}")
        else:
            print("No CMAP section found (offset is 0).")

        # --- 3. Read Constants Map (CNST) ---
        print("\n--- Constants (CNST) ---")
        if cnst_offset > 0:
            f.seek(cnst_offset)
            if f.read(4) != b'CNST': print("Error: CNST tag mismatch."); return

            num_consts = struct.unpack('<I', f.read(4))[0]
            print(f"Found {num_consts} constants stored in this file:")
            
            consts_list = []
            for _ in range(num_consts):
                const_id, val_len = struct.unpack('<HB', f.read(3))
                const_val_str = f.read(val_len).decode('utf-8')
                try:
                    const_val = json.loads(const_val_str)
                    consts_list.append((const_id, const_val, const_val_str))
                except json.JSONDecodeError:
                    consts_list.append((const_id, const_val_str, None))
            
            for const_id, const_val, raw_str in sorted(consts_list):
                display_val = repr(const_val)
                if raw_str and repr(raw_str) != display_val and len(display_val) < 80:
                     print(f"  ID {const_id:<4}: {display_val}")
                else:
                     print(f"  ID {const_id:<4}: {display_val}")
        else:
            print("No CNST section found (offset is 0).")

        # --- 4. Read Permutations Map (PERM) ---
        print("\n--- Permutations (PERM) ---")
        if perm_offset > 0:
            f.seek(perm_offset)
            if f.read(4) != b'PERM': print("Error: PERM tag mismatch."); return
            
            num_perms = struct.unpack('<I', f.read(4))[0]
            print(f"Found {num_perms} permutations stored in this file:")

            perms_list = []
            for _ in range(num_perms):
                p_id, p_len = struct.unpack('<HB', f.read(3))
                p_val_str = f.read(p_len).decode('utf-8')
                perms_list.append((p_id, p_val_str))

            for p_id, p_val_str in sorted(perms_list):
                print(f"  ID {p_id:<4}: '{p_val_str}'")
        else:
            print("No PERM section found (offset is 0).")

        # --- 5. Read Data Section Info (DATA) ---
        print("\n--- Data & Parameters (DATA) ---")
        if data_offset > 0:
            f.seek(data_offset)
            if f.read(4) != b'DATA': print("Error: DATA tag mismatch."); return

            # Parameter Names
            num_params = struct.unpack('<I', f.read(4))[0]
            print(f"Found {num_params} parameter name mappings:")
            param_names_list = []
            for _ in range(num_params):
                p_id, name_len = struct.unpack('<HH', f.read(4))
                p_name = f.read(name_len).decode('utf-8')
                param_names_list.append((p_id, p_name))
            
            param_names = dict(param_names_list)
            for p_id, p_name in sorted(param_names_list):
                print(f"  Param ID {p_id:<4}: '{p_name}'")

            # Input Names
            num_inputs = struct.unpack('<I', f.read(4))[0]
            print(f"\nFound {num_inputs} input name mappings:")
            input_names_list = []
            for _ in range(num_inputs):
                i_idx, name_len = struct.unpack('<HH', f.read(4))
                i_name = f.read(name_len).decode('utf-8')
                input_names_list.append((i_idx, i_name))
            for i_idx, i_name in sorted(input_names_list):
                print(f"  Node Index {i_idx:<4}: '{i_name}'")
            
            # Internal Weights Info
            if weights_stored_internally:
                num_tensors = struct.unpack('<I', f.read(4))[0]
                print(f"\nFound {num_tensors} internally stored tensors:")
                
                dtype_map = {0:'f32', 1:'f64', 2:'f16', 3:'bf16', 4:'i32', 5:'i64', 6:'i16', 7:'i8', 8:'u8', 9:'bool'}

                for _ in range(num_tensors):
                    p_id, num_props, meta_len, data_len = struct.unpack('<HBIQ', f.read(15))
                    meta_bytes = f.read(meta_len)
                    f.seek(data_len, 1) # Skip tensor data

                    meta, meta_offset = {}, 0
                    for _ in range(num_props):
                        key_len = meta_bytes[meta_offset]; meta_offset += 1
                        key = meta_bytes[meta_offset:meta_offset+key_len].decode('utf-8'); meta_offset += key_len
                        val_len = struct.unpack('<H', meta_bytes[meta_offset:meta_offset+2])[0]; meta_offset += 2
                        val_str = meta_bytes[meta_offset:meta_offset+val_len].decode('utf-8'); meta_offset += val_len
                        meta[key] = json.loads(val_str)
                    
                    shape = meta.get('shape', '[UNKNOWN]')
                    dtype_id = meta.get('dtype', -1)
                    dtype_str = dtype_map.get(dtype_id, 'UNK')
                    
                    param_name = f"'{param_names.get(p_id, 'N/A')}'"
                    print(f"  - Tensor ID {p_id:<3} ({param_name:<20}): Shape={str(shape):<20} DType={dtype_str:<5} Size={data_len} bytes")
        else:
            print("No DATA section found (offset is 0).")

        # --- 6. Read Processing/Tokenizer Info (PROC) ---
        print("\n--- Processing & Tokenizer (PROC) ---")
        if proc_offset > 0:
            f.seek(proc_offset)
            if f.read(4) != b'PROC':
                print("Error: PROC section tag mismatch.")
            else:
                manifest_len = struct.unpack('<I', f.read(4))[0]
                manifest_bytes = f.read(manifest_len)
                print(f"Found PROC section of {manifest_len} bytes.")
                print("Disassembling TISA Tokenizer Manifest:")
                try:
                    disassembled = disassemble_TISA_manifest(manifest_bytes)
                    print(json.dumps(disassembled, indent=2, ensure_ascii=False))
                except Exception as e:
                    print(f"  -> ERROR during disassembly: {e}")
        else:
            print("No PROC section found (offset is 0).")

        # --- 7. Read Internal Resources (RSRC) ---
        print("\n--- Internal Resources (RSRC) ---")
        if rsrc_offset > 0:
            f.seek(rsrc_offset)
            if f.read(4) != b'RSRC':
                print("Error: RSRC section tag mismatch.")
            else:
                num_files = struct.unpack('<I', f.read(4))[0]
                print(f"Found {num_files} internal resource files:")
                for _ in range(num_files):
                    name_len = struct.unpack('<H', f.read(2))[0]
                    filename = f.read(name_len).decode('utf-8')
                    data_len = struct.unpack('<I', f.read(4))[0]
                    f.seek(data_len, 1) # Skip data
                    print(f"  - File: '{filename}' (Size: {data_len} bytes)")
        else:
            print("No RSRC section found (offset is 0).")

    print("\n" + "="*20 + f" INSPECTION COMPLETE " + "="*20 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python NAC_info.py <path_to_nac_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    inspect_nac_file(filepath)
