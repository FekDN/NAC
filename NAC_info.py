# Файл: NAC_info.py

# Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

import sys
import os
import struct
import json

try:
    from TISA_tokenizer import disassemble_TISA_manifest
except ImportError:
    def disassemble_TISA_manifest(manifest_bytes):
        return {"error": "TISA_tokenizer library not found, cannot disassemble manifest."}

try:
    from NAC_kernels import NAC_OPS
except ImportError:
    print("Warning: NAC_kernels.py not found. Standard NAC operations will not be recognized.")
    NAC_OPS = {}

def inspect_nac_file(filepath: str):
    """
    Reads a .nac file in the new robust binary format and prints its contents,
    including the tokenizer manifest if present.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return

    print(f"\n" + "="*20 + f" INSPECTING: {os.path.basename(filepath)} " + "="*20)

    base_ops = {
        2: "<INPUT>",
        3: "<OUTPUT>",
        6: "<CONTROL_FLOW>",
        7: "<CONVERGENCE>",
    }
    base_ops.update(NAC_OPS)

    with open(filepath, 'rb') as f:
        # --- 0. Read and Display Full Header ---
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
        quant_map = {0:'none', 1:'FP16', 2:'INT8_TENSOR', 3:'INT8_CHANNEL', 4:'BLOCK_FP8'}
        quant_method = quant_map.get(quant_method_id, f'UNKNOWN ({quant_method_id})')
        storage_method = "Internal" if weights_stored_internally else "External (.safetensors)"
        print(f"Quantization: {quant_method}")
        print(f"Weights Storage: {storage_method}")
        
        # --- IO Counts ---
        num_inputs, num_outputs, reserved = struct.unpack('<HHB', f.read(5))
        print(f"IO Counts: {num_inputs} Inputs, {num_outputs} Outputs")

        # --- Section Offsets ---
        offsets_header_format = '<H9Q4x' 
        header_bytes = f.read(struct.calcsize(offsets_header_format))
        
        unpacked_header = struct.unpack(offsets_header_format, header_bytes)
        d_model = unpacked_header[0]
        offsets = unpacked_header[1:]
        
        mmap_off, ops_off, cmap_off, cnst_off, perm_off, data_off, \
        proc_off, orch_off, rsrc_off = offsets
        
        print(f"Model Dimension (d_model): {d_model}")

        print("\n--- Section Offsets ---")
        print(f"  - MMAP Section:     Starts at byte {mmap_off}")
        print(f"  - OPS Section:      Starts at byte {ops_off}")
        print(f"  - CMAP Section:     Starts at byte {cmap_off}")
        print(f"  - CNST Section:     Starts at byte {cnst_off}")
        print(f"  - PERM Section:     Starts at byte {perm_off}")
        print(f"  - DATA Section:     Starts at byte {data_off}")
        print(f"  - PROC Section:     Starts at byte {proc_off} (Tokenizer Manifest)")
        print(f"  - RSRC Section:     Starts at byte {rsrc_off} (Internal Resources)")
        print(f"  - ORCH Section:     Starts at byte {orch_off} (MEP Orchestrator)")

        # --- 1. Read Memory Map (MMAP) ---
        print("\n--- Memory Map (MMAP) ---")
        if mmap_off > 0:
            f.seek(mmap_off)
            if f.read(4) != b'MMAP': print("Error: MMAP tag mismatch.")
            else:
                num_records = struct.unpack('<I', f.read(4))[0]
                print(f"Found {num_records} memory management records:")
                
                action_map = {10: 'SAVE_RESULT', 20: 'FREE', 30: 'FORWARD', 40: 'PRELOAD'}
                
                for _ in range(num_records):
                    instr_id, num_cmds = struct.unpack('<HB', f.read(3))
                    cmds_str = []
                    for _ in range(num_cmds):
                        action_code, target_id = struct.unpack('<BH', f.read(3))
                        action_str = action_map.get(action_code, f'UNK({action_code})')
                        cmds_str.append(f"{action_str} -> {target_id}")
                    print(f"  - On Tick {instr_id:<4}: " + ", ".join(cmds_str))
        else:
            print("No MMAP section found.")

        # --- 2. Read Operations Map (CMAP) ---
        print("\n--- Custom Operations (CMAP) ---")
        custom_ops = {}
        if cmap_off > 0:
            f.seek(cmap_off)
            if f.read(4) != b'CMAP': print("Error: CMAP tag mismatch."); return
            
            num_ops = struct.unpack('<I', f.read(4))[0]
            print(f"Found {num_ops} custom operation mappings in this file:")
            
            for _ in range(num_ops):
                op_id, name_len = struct.unpack('<HB', f.read(3))
                op_name = f.read(name_len).decode('utf-8')
                custom_ops[op_id] = op_name
            
            for op_id, op_name in sorted(custom_ops.items()):
                 print(f"  ID {op_id:<4}: {op_name}")
        else:
            print("No CMAP section found. This file uses only standard operations.")
        
        all_ops = base_ops.copy()
        all_ops.update(custom_ops)

        # --- 3. Read Constants Map (CNST) ---
        print("\n--- Constants (CNST) ---")
        if cnst_off > 0:
            f.seek(cnst_off)
            if f.read(4) != b'CNST': print("Error: CNST tag mismatch."); return

            num_consts = struct.unpack('<I', f.read(4))[0]
            print(f"Found {num_consts} constants stored in this file:")
            
            consts_list = []
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
                consts_list.append((const_id, val))
            
            for const_id, const_val in sorted(consts_list):
                 print(f"  ID {const_id:<4}: {repr(const_val)}")
        else:
            print("No CNST section found.")

        # --- 4. Read Permutations Map (PERM) ---
        print("\n--- Permutations (PERM) ---")
        if perm_off > 0:
            f.seek(perm_off)
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
            print("No PERM section found.")

        # --- 5. Read Data Section Info (DATA) ---
        print("\n--- Data & Parameters (DATA) ---")
        if data_off > 0:
            f.seek(data_off)
            if f.read(4) != b'DATA': print("Error: DATA tag mismatch."); return

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

            num_inputs_map = struct.unpack('<I', f.read(4))[0]
            print(f"\nFound {num_inputs_map} input name mappings:")
            input_names_list = []
            for _ in range(num_inputs_map):
                i_idx, name_len = struct.unpack('<HH', f.read(4))
                i_name = f.read(name_len).decode('utf-8')
                input_names_list.append((i_idx, i_name))
            for i_idx, i_name in sorted(input_names_list):
                print(f"  Node Index {i_idx:<4}: '{i_name}'")
            
            if weights_stored_internally:
                pos = f.tell()
                f.seek(0, 2)
                end = f.tell()
                f.seek(pos)
                if pos < end:
                    num_tensors = struct.unpack('<I', f.read(4))[0]
                    print(f"\nFound {num_tensors} internally stored tensors:")
                    
                    dtype_map = {0:'f32', 1:'f64', 2:'f16', 3:'bf16', 4:'i32', 5:'i64', 6:'i16', 7:'i8', 8:'u8', 9:'bool'}

                    for _ in range(num_tensors):
                        p_id, meta_len, data_len = struct.unpack('<HIQ', f.read(14))
                        meta_bytes = f.read(meta_len)
                        
                        # --- START: New Pure Binary Metadata Deserialization ---
                        # This block replaces the json.loads() call to parse the binary metadata
                        meta_offset = 0
                        
                        # 1. Dtype and Rank
                        dtype_id, rank = struct.unpack_from('<BB', meta_bytes, meta_offset)
                        meta_offset += 2
                        
                        # 2. Shape
                        shape = []
                        if rank > 0:
                            shape = list(struct.unpack_from(f'<{rank}I', meta_bytes, meta_offset))
                        # --- END: New Pure Binary Metadata Deserialization ---
                        
                        # The info script doesn't need to read the full meta, just enough for display.
                        # Now skip the actual tensor data bytes to get to the next tensor header.
                        f.seek(data_len, 1) 
                        
                        dtype_str = dtype_map.get(dtype_id, 'UNK')
                        param_name = f"'{param_names.get(p_id, 'N/A')}'"
                        print(f"  - Tensor ID {p_id:<3} ({param_name:<20}): Shape={str(shape):<20} DType={dtype_str:<5} Size={data_len} bytes")
                else:
                    print("\nNo internally stored tensor data found.")
        else:
            print("No DATA section found.")

        # --- 6. Read Processing/Tokenizer Info (PROC) ---
        print("\n--- Processing & Tokenizer (PROC) ---")
        if proc_off > 0:
            f.seek(proc_off)
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
            print("No PROC section found.")

        # --- 7. Read Internal Resources (RSRC) ---
        print("\n--- Internal Resources (RSRC) ---")
        if rsrc_off > 0:
            f.seek(rsrc_off)
            if f.read(4) != b'RSRC':
                print("Error: RSRC section tag mismatch.")
            else:
                num_files = struct.unpack('<I', f.read(4))[0]
                print(f"Found {num_files} internal resource files:")
                for _ in range(num_files):
                    name_len = struct.unpack('<H', f.read(2))[0]
                    filename = f.read(name_len).decode('utf-8')
                    data_len = struct.unpack('<I', f.read(4))[0]
                    f.seek(data_len, 1)
                    print(f"  - File: '{filename}' (Size: {data_len} bytes)")
        else:
            print("No RSRC section found.")

        # --- 8. Read MEP Orchestrator Plan (ORCH) ---
        print("\n--- MEP Orchestrator Plan (ORCH) ---")
        if orch_off > 0:
            f.seek(orch_off)
            magic = f.read(4)
            if magic != b'ORCH':
                print(f"Error: ORCH tag mismatch (got {magic!r}).")
            else:
                # Format: bytecode_len(4) + const_count(4) written BEFORE bytecode
                bytecode_len, const_count = struct.unpack('<II', f.read(8))
                if bytecode_len == 0:
                    print("ORCH section present but empty (no MEP plan compiled into this file).")
                else:
                    bytecode = f.read(bytecode_len)

                    # Read constants pool
                    orch_consts = {}
                    for _ in range(const_count):
                        const_id, type_code, length = struct.unpack('<HBH', f.read(5))
                        val = None
                        if   type_code == 0: val = None
                        elif type_code == 1: val = struct.unpack('<?'  , f.read(length))[0]
                        elif type_code == 2: val = struct.unpack('<q'  , f.read(length))[0]
                        elif type_code == 3: val = struct.unpack('<d'  , f.read(length))[0]
                        elif type_code == 4: val = f.read(length).decode('utf-8')
                        elif type_code == 5: val = list(struct.unpack(f'<{length}i', f.read(length * 4))) if length > 0 else []
                        elif type_code == 6: val = list(struct.unpack(f'<{length}f', f.read(length * 4))) if length > 0 else []
                        orch_consts[const_id] = val

                    print(f"MEP bytecode: {bytecode_len} bytes")
                    print(f"MEP constants pool: {const_count} entries")

                    # Disassemble bytecode using opcode names
                    OPCODE_NAMES = {
                        0x02: "SRC_USER_PROMPT",    0x04: "SRC_CONSTANT",
                        0x10: "RES_LOAD_MODEL",     0x11: "RES_LOAD_DATAFILE",
                        0x12: "RES_LOAD_EXTERN",    0x13: "RES_LOAD_DYNAMIC",
                        0x1F: "RES_UNLOAD",
                        0x20: "PREPROC_ENCODE",     0x21: "PREPROC_DECODE",
                        0x22: "PREPROC_GET_ID",     0x2A: "STRING_FORMAT",
                        0x30: "TENSOR_CREATE",      0x38: "TENSOR_MANIPULATE",
                        0x39: "TENSOR_COMBINE",     0x3A: "TENSOR_INFO",
                        0x3B: "TENSOR_EXTRACT",
                        0x59: "SYS_COPY",           0x5F: "SYS_DEBUG_PRINT",
                        0x60: "MATH_UNARY",         0x61: "MATH_BINARY",
                        0x62: "MATH_AGGREGATE",     0x68: "LOGIC_COMPARE",
                        0x80: "MODEL_RUN_STATIC",
                        0xA0: "FLOW_LOOP_START",    0xA1: "FLOW_LOOP_END",
                        0xA8: "FLOW_BRANCH_IF",     0xA9: "FLOW_BREAK_LOOP_IF",
                        0xE0: "SERIALIZE_OBJECT",   0xF0: "IO_WRITE",
                        0xFE: "EXEC_RETURN",        0xFF: "EXEC_HALT",
                    }
                    print("\nMEP Bytecode disassembly:")
                    ip = 0
                    while ip < len(bytecode):
                        flag = bytecode[ip]
                        name = OPCODE_NAMES.get(flag, f"UNK_0x{flag:02X}")
                        rest = bytecode[ip+1:min(ip+9, len(bytecode))]
                        hex_args = ' '.join(f'{b:02X}' for b in rest)
                        print(f"  0x{ip:04X}  {flag:02X}  {name:<22}  {hex_args}")
                        # Advance IP by known fixed lengths (variable-length ops show at least args)
                        FIXED_LENGTHS = {
                            0x02:5, 0x04:4, 0x10:4, 0x11:5, 0x12:5, 0x13:4, 0x1F:3,
                            0x20:4, 0x21:4, 0x22:5, 0x30:5, 0x3B:4,
                            0x59:3, 0x5F:4, 0x60:4, 0x61:5, 0x62:4, 0x68:5,
                            0xA0:3, 0xA1:4, 0xA8:5, 0xA9:5, 0xE0:4, 0xF0:5,
                            0xFF:1,
                        }
                        if flag in FIXED_LENGTHS:
                            ip += FIXED_LENGTHS[flag]
                        elif flag == 0x2A:   # STRING_FORMAT: variable
                            count = bytecode[ip+4] if ip+4 < len(bytecode) else 0
                            ip += 5 + count
                        elif flag == 0x38:   # TENSOR_MANIPULATE
                            ip += 6
                        elif flag == 0x39:   # TENSOR_COMBINE
                            count = bytecode[ip+3] if ip+3 < len(bytecode) else 0
                            ip += 5 + count
                        elif flag == 0x3A:   # TENSOR_INFO
                            op = bytecode[ip+1] if ip+1 < len(bytecode) else 0
                            ip += 4 + (1 if op == 1 else 0)
                        elif flag == 0x80:   # MODEL_RUN_STATIC
                            n_in  = bytecode[ip+2] if ip+2 < len(bytecode) else 0
                            n_out = bytecode[ip+3+n_in] if ip+3+n_in < len(bytecode) else 0
                            ip += 4 + n_in + n_out
                        elif flag == 0xFE:   # EXEC_RETURN
                            count = bytecode[ip+1] if ip+1 < len(bytecode) else 0
                            ip += 2 + count
                        else:
                            ip += 1  # unknown: step 1 to avoid infinite loop

                    if orch_consts:
                        print("\nMEP Constants pool:")
                        for cid in sorted(orch_consts):
                            print(f"  [{cid:3d}] {repr(orch_consts[cid])}")
        else:
            print("No ORCH section found (file compiled without MEP plan).")

    print("\n" + "="*20 + f" INSPECTION COMPLETE " + "="*20 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python NAC_info.py <path_to_nac_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    inspect_nac_file(filepath)