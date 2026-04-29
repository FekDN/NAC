# --- START OF FILE MEP_compiler.py ---
# Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

import struct
import os
from typing import Dict, Any, List, Optional, Tuple

class MEPCompiler:
    def __init__(self):
        self.bytecode = bytearray()
        self.constants: Dict[str, Tuple[int, Any]] = {}
        self.next_const_id = 0
        self.context_vars: Dict[str, int] = {}
        self.next_context_key = 0
        self._labels: Dict[str, int] = {}
        self._jumps: Dict[str, List[int]] = {}
        self._loop_stack: List[Dict[str, Any]] = []

    def _get_const_id(self, value: Any) -> int:
        val_key = str(value)
        if val_key not in self.constants:
            if self.next_const_id > 65535: raise OverflowError("Constant pool limit exceeded.")
            self.constants[val_key] = (self.next_const_id, value)
            self.next_const_id += 1
        return self.constants[val_key][0]

    def _get_context_key(self, name: str) -> int:
        if name not in self.context_vars:
            if self.next_context_key > 255: raise OverflowError("Context variable limit exceeded.")
            self.context_vars[name] = self.next_context_key
            self.next_context_key += 1
        return self.context_vars[name]

    # 0x00-0x0F  Data Sources
    def src_user_prompt(self, out_var: str, prompt_text: str, data_type: int = 0):
        """0x02: data_type (0=str, 1=int, 2=float)"""
        key = self._get_context_key(out_var)
        self.bytecode.append(0x02)
        self.bytecode += struct.pack('<BBH', key, data_type, self._get_const_id(prompt_text))

    def src_constant(self, out_var: str, value: Any):
        """0x04: Load constant"""
        key = self._get_context_key(out_var)
        self.bytecode.append(0x04)
        self.bytecode += struct.pack('<BH', key, self._get_const_id(value))

    def src_exec_mode(self, out_var: str):
        """0x05: Load current exec_mode string ('infer', 'train', 'infer_train')"""
        key = self._get_context_key(out_var)
        self.bytecode.append(0x05)
        self.bytecode += struct.pack('<B', key)

    # 0x10-0x1F  Resource management
    def res_load_model(self, model_id: int, model_path: str):
        self.bytecode.append(0x10)
        self.bytecode += struct.pack('<BH', model_id, self._get_const_id(model_path))

    def res_load_datafile(self, out_var: str, file_path: str, file_type: int):
        self.bytecode.append(0x11)
        self.bytecode += struct.pack('<BBH', self._get_context_key(out_var), file_type, self._get_const_id(file_path))

    def res_load_dynamic(self, out_var: str, path_var: str, file_type: int):
        self.bytecode.append(0x13)
        self.bytecode += struct.pack('<BBB', self._get_context_key(out_var), self._get_context_key(path_var), file_type)

    def res_load_extern(self, out_var: str, res_type: int, resource_id: Any):
        self.bytecode.append(0x12)
        self.bytecode += struct.pack('<BBH', self._get_context_key(out_var), res_type, self._get_const_id(resource_id))

    def res_unload(self, res_type: int, id_or_var: str):
        self.bytecode.append(0x1F)
        val = int(id_or_var) if res_type == 0 else self._get_context_key(id_or_var)
        self.bytecode += struct.pack('<BB', res_type, val)

    # 0x20-0x2F  Preprocessing
    def preproc_encode(self, proc_var: str, in_var: str, out_var: str):
        self.bytecode.append(0x20)
        self.bytecode += struct.pack('<BBB', self._get_context_key(proc_var), self._get_context_key(in_var), self._get_context_key(out_var))

    def preproc_decode(self, proc_var: str, in_var: str, out_var: str):
        self.bytecode.append(0x21)
        self.bytecode += struct.pack('<BBB', self._get_context_key(proc_var), self._get_context_key(in_var), self._get_context_key(out_var))

    def preproc_get_id(self, proc_var: str, item_str: str, out_var: str):
        self.bytecode.append(0x22)
        self.bytecode += struct.pack('<BHB', self._get_context_key(proc_var), self._get_const_id(item_str), self._get_context_key(out_var))

    def string_format(self, out_var: str, format_str: str, in_vars: List[str]):
        in_keys = [self._get_context_key(v) for v in in_vars]
        self.bytecode.append(0x2A)
        self.bytecode += struct.pack('<BHB', self._get_context_key(out_var), self._get_const_id(format_str), len(in_keys))
        if in_keys: self.bytecode += struct.pack(f'<{len(in_keys)}B', *in_keys)

    # 0x30-0x4F  Tensor processing
    def tensor_create(self, from_py: Optional[dict] = None, arange: Optional[dict] = None, ones: Optional[dict] = None):
        self.bytecode.append(0x30)
        if from_py:
            self.bytecode += struct.pack('<BBBB', self._get_context_key(from_py['out_var']), from_py['dtype_code'], 0, self._get_context_key(from_py['in_var']))
        elif arange:
            self.bytecode += struct.pack('<BBBB', self._get_context_key(arange['out_var']), arange['dtype_code'], 1, self._get_context_key(arange['end_var']))
        elif ones:
            self.bytecode += struct.pack('<BBBB', self._get_context_key(ones['out_var']), ones['dtype_code'], 2, self._get_context_key(ones['shape_var']))
        else: raise ValueError("Specify from_py, arange, or ones.")

    def tensor_manipulate(self, op_type: str, out_var: str, in_var: str, **kwargs):
        op_code = {'pad': 1}.get(op_type)
        self.bytecode.append(0x38)
        self.bytecode += struct.pack('<BBB', op_code, self._get_context_key(out_var), self._get_context_key(in_var))
        if op_type == 'pad': self.bytecode += struct.pack('<BB', self._get_context_key(kwargs['pad_width_var']), self._get_context_key(kwargs['const_val_var']))

    def tensor_combine(self, op_type: str, out_var: str, in_vars: List[str], **kwargs):
        op_code = {'concat': 0}.get(op_type)
        in_keys = [self._get_context_key(v) for v in in_vars]
        self.bytecode.append(0x39)
        self.bytecode += struct.pack('<BBB', op_code, self._get_context_key(out_var), len(in_keys))
        if in_keys: self.bytecode += struct.pack(f'<{len(in_keys)}B', *in_keys)
        if op_type == 'concat': self.bytecode += struct.pack('<B', self._get_context_key(kwargs['axis_var']))

    def tensor_info(self, op_type: str, out_var: str, in_var: str, **kwargs):
        op_code = {'shape': 0, 'dim': 1, 'to_py': 2}.get(op_type)
        self.bytecode.append(0x3A)
        self.bytecode += struct.pack('<BBB', op_code, self._get_context_key(out_var), self._get_context_key(in_var))
        if op_type == 'dim': self.bytecode += struct.pack('<B', self._get_context_key(kwargs['dim_idx_var']))

    def tensor_extract(self, out_var: str, in_tensor_var: str, in_idx_var: str):
        self.bytecode.append(0x3B)
        self.bytecode += struct.pack('<BBB', self._get_context_key(out_var), self._get_context_key(in_tensor_var), self._get_context_key(in_idx_var))

    # 0x50-0x5F  System / utility
    def sys_copy(self, out_var: str, in_var: str):
        self.bytecode.append(0x59)
        self.bytecode += struct.pack('<BB', self._get_context_key(out_var), self._get_context_key(in_var))

    def sys_debug_print(self, var_name: str, msg: str = ""):
        self.bytecode.append(0x5F)
        self.bytecode += struct.pack('<BH', self._get_context_key(var_name), self._get_const_id(msg or f"{var_name}:"))

    # 0x60-0x7F  Post-processing & logic
    def math_unary(self, op_type: str, out_var: str, in_var: str):
        op_code = {'softmax': 0}.get(op_type)
        self.bytecode.append(0x60)
        self.bytecode += struct.pack('<BBB', op_code, self._get_context_key(out_var), self._get_context_key(in_var))

    def math_binary(self, op_type: str, out_var: str, in_var1: str, in_var2: str):
        op_code = {'add': 0, 'sub': 1, 'mul': 2}.get(op_type)
        self.bytecode.append(0x61)
        self.bytecode += struct.pack('<BBBB', op_code, self._get_context_key(out_var), self._get_context_key(in_var1), self._get_context_key(in_var2))

    def math_aggregate(self, op_type: str, out_var: str, in_var: str):
        op_code = {'argmax': 0}.get(op_type)
        self.bytecode.append(0x62)
        self.bytecode += struct.pack('<BBB', op_code, self._get_context_key(out_var), self._get_context_key(in_var))

    def logic_compare(self, op_type: str, out_var: str, in_var1: str, in_var2: str):
        op_code = {'eq': 0, 'neq': 1, 'gt': 2, 'lt': 3}.get(op_type)
        self.bytecode.append(0x68)
        self.bytecode += struct.pack('<BBBB', op_code, self._get_context_key(out_var), self._get_context_key(in_var1), self._get_context_key(in_var2))

    def analysis_top_k(self, in_var: str, k_var: str, out_indices_var: str, out_vals_var: str):
        self.bytecode.append(0x70)
        self.bytecode += struct.pack('<BBBB', self._get_context_key(in_var), self._get_context_key(k_var), self._get_context_key(out_indices_var), self._get_context_key(out_vals_var))

    def analysis_sample(self, logits_var: str, temp_var: str, topk_var: str, out_var: str):
        self.bytecode.append(0x71)
        self.bytecode += struct.pack('<BBBB', self._get_context_key(logits_var), self._get_context_key(temp_var), self._get_context_key(topk_var), self._get_context_key(out_var))

    # 0x80-0x8F  Model execution
    def model_run_static(self, model_id: int, in_vars: List[str], out_vars: List[str]):
        in_keys, out_keys = [self._get_context_key(v) for v in in_vars], [self._get_context_key(v) for v in out_vars]
        self.bytecode.append(0x80)
        self.bytecode += struct.pack('<BB', model_id, len(in_keys))
        if in_keys: self.bytecode += struct.pack(f'<{len(in_keys)}B', *in_keys)
        self.bytecode.append(len(out_keys))
        if out_keys: self.bytecode += struct.pack(f'<{len(out_keys)}B', *out_keys)

    def model_train_step(self, 
                         model_id: int, 
                         loss_type: int, 
                         in_vars: List[str], 
                         target_vars: List[str], 
                         out_loss_var: str, 
                         lr_var: str = "learning_rate",
                         logits_var: str = None,
                         head_weight_name: str = "",
                         head_bias_name: str = ""):
        """Простая и надёжная версия model_train_step"""
        in_keys = [self._get_context_key(v) for v in in_vars]
        target_keys = [self._get_context_key(v) for v in target_vars]

        self.bytecode.append(0x82)
        self.bytecode += struct.pack('<BBB', model_id, loss_type, len(in_keys))
        if in_keys:
            self.bytecode += struct.pack(f'<{len(in_keys)}B', *in_keys)

        self.bytecode.append(len(target_keys))
        if target_keys:
            self.bytecode += struct.pack(f'<{len(target_keys)}B', *target_keys)

        logits_key = self._get_context_key(logits_var) if logits_var else 0
        lr_key = self._get_context_key(lr_var)
        weight_name_id = self._get_const_id(head_weight_name) if head_weight_name else 0
        bias_name_id = self._get_const_id(head_bias_name) if head_bias_name else 0

        # Простая и стабильная упаковка (BBBHH вместо BBHHH)
        self.bytecode += struct.pack('<BBBHH', 
                                     self._get_context_key(out_loss_var),
                                     lr_key,
                                     logits_key,
                                     weight_name_id,
                                     bias_name_id)

    def model_zero_grad(self, model_id: int):
        self.bytecode.append(0x83)
        self.bytecode.append(model_id)

    def model_save_weights(self, model_id: int, path_var: str, save_type: int = 0):
        self.bytecode.append(0x85)
        self.bytecode += struct.pack('<BBB', model_id, self._get_context_key(path_var), save_type)

    # 0xA0-0xAF  Flow control
    def flow_loop_start(self, counter_var: str):
        self.bytecode.append(0xA0)
        self.bytecode.append(self._get_context_key(counter_var))
        self._loop_stack.append({'start_pos': len(self.bytecode), 'break_placeholders': []})

    def flow_loop_end(self):
        if not self._loop_stack: raise RuntimeError("flow_loop_end() without matching flow_loop_start().")
        loop_info = self._loop_stack[-1]
        break_target_pos = len(self.bytecode) + 3
        for ph in loop_info['break_placeholders']:
            self.bytecode[ph:ph+2] = struct.pack('<h', break_target_pos - (ph + 2))
        jump_offset = loop_info['start_pos'] - (len(self.bytecode) + 3)
        self.bytecode.append(0xA1)
        self.bytecode += struct.pack('<h', jump_offset)
        self._loop_stack.pop()

    def flow_branch_if(self, cond_var: str, jump_label: str):
        self.bytecode.append(0xA8)
        self.bytecode.append(self._get_context_key(cond_var))
        offset_pos = len(self.bytecode)
        self.bytecode += struct.pack('<h', 0)
        self._jumps.setdefault(jump_label, []).append(offset_pos)

    def flow_break_loop_if(self, cond_var: str):
        if not self._loop_stack: raise RuntimeError("flow_break_loop_if() outside loop.")
        self.bytecode.append(0xA9)
        self.bytecode.append(self._get_context_key(cond_var))
        offset_pos = len(self.bytecode)
        self.bytecode += struct.pack('<h', 0)
        self._loop_stack[-1]['break_placeholders'].append(offset_pos)

    def place_label(self, label_name: str):
        if label_name in self._labels: raise NameError(f"Label '{label_name}' already defined.")
        self._labels[label_name] = len(self.bytecode)

    # 0xE0-0xEF  Data serialization
    def serialize_object(self, out_var: str, in_var: str, format_type: int):
        self.bytecode.append(0xE0)
        self.bytecode += struct.pack('<BBB', self._get_context_key(out_var), self._get_context_key(in_var), format_type)

    # 0xF0-0xFF  I/O & termination
    def io_write(self, in_var: str, dest_type: int, dest_var: Optional[str] = None, write_mode: int = 0):
        dest_key = self._get_context_key(dest_var) if dest_var else 0
        self.bytecode.append(0xF0)
        self.bytecode += struct.pack('<BBBB', self._get_context_key(in_var), dest_type, dest_key, write_mode)

    def exec_return(self, var_names: List[str]):
        keys = [self._get_context_key(v) for v in var_names]
        self.bytecode.append(0xFE)
        self.bytecode.append(len(keys))
        if keys: self.bytecode += struct.pack(f'<{len(keys)}B', *keys)

    def exec_halt(self):
        self.bytecode.append(0xFF)

    # Finalization
    def get_program(self) -> Tuple[bytes, Dict[int, Any]]:
        if self._loop_stack: raise RuntimeError("Not all loops closed with flow_loop_end().")
        for label_name, jump_positions in self._jumps.items():
            if label_name not in self._labels: raise NameError(f"Jump label '{label_name}' never placed.")
            target_pos = self._labels[label_name]
            for offset_pos in jump_positions:
                jump_offset = target_pos - (offset_pos + 2)
                self.bytecode[offset_pos:offset_pos+2] = struct.pack('<h', jump_offset)
        const_map = {cid: val for _, (cid, val) in self.constants.items()}
        return bytes(self.bytecode), const_map

class MEPPatcher:
    _FIXED: Dict[int, int] = {
        0x02: 4, 0x03: 3, 0x04: 3, 0x05: 1,
        0x10: 3, 0x11: 4, 0x12: 4, 0x13: 3, 0x1F: 2,
        0x20: 3, 0x21: 3, 0x22: 4, 0x2A: -1,
        0x30: 4, 0x38: -1, 0x39: -1, 0x3A: -1, 0x3B: 3,
        0x59: 2, 0x5F: 3,
        0x60: 3, 0x61: 4, 0x62: 3, 0x68: 4,
        0x70: 4, 0x71: 4,
        0x80: -1, 0x82: -1, 0x83: 1, 0x85: 3,
        0xA0: 1, 0xA1: 2, 0xA8: 3, 0xA9: 3,
        0xE0: 3, 0xF0: 4, 0xFE: -1, 0xFF: 0,
    }

    @classmethod
    def instruction_length(cls, bytecode: bytes, offset: int) -> int:
        flag = bytecode[offset]
        length = cls._FIXED.get(flag)
        if length is None: raise NotImplementedError(f"Unknown opcode 0x{flag:02x}")
        if length != -1: return 1 + length
        if flag == 0x2A: return 5 + bytecode[offset + 4]
        if flag == 0x38: return 1 + (5 if bytecode[offset + 1] == 1 else 3)
        if flag == 0x39: return 4 + bytecode[offset + 3] + (1 if bytecode[offset + 1] == 0 else 0)
        if flag == 0x3A: return 4 + (1 if bytecode[offset + 1] == 1 else 0)
        if flag == 0x80: count_in = bytecode[offset + 2]; return 4 + count_in + bytecode[offset + 3 + count_in]
        if flag == 0x82: 
            count_in = bytecode[offset + 3]
            count_target = bytecode[offset + 4 + count_in]
            return 12 + count_in + count_target
        if flag == 0xFE: return 2 + bytecode[offset + 1]

    @classmethod
    def find_src_constant_offset(cls, bytecode: bytes, constants: Dict[int, Any], target_value: Any) -> Optional[int]:
        target_str = str(target_value)
        offset = 0
        while offset < len(bytecode):
            flag = bytecode[offset]
            length = cls.instruction_length(bytecode, offset)
            if flag == 0x04:
                cid = struct.unpack_from('<H', bytecode, offset + 2)[0]
                if str(constants.get(cid)) == target_str: return offset
            offset += length
        return None

    @classmethod
    def patch_src_constant_value(cls, bytecode: bytes, constants: Dict[int, Any], offset: int, new_value: Any) -> Tuple[bytes, Dict[int, Any]]:
        cid = struct.unpack_from('<H', bytecode, offset + 2)[0]
        new_consts = dict(constants)
        new_consts[cid] = new_value
        return bytecode, new_consts

    @classmethod
    def rewrite_constant_in_nac(cls, nac_path: str, old_value: Any, new_value: Any, occurrence: int = 0) -> bool:
        from NAC_run import NacRuntime
        bytecode, constants = NacRuntime.load_mep_from_nac(nac_path)
        target_str = str(old_value)
        found, target_cid = 0, None
        for k, v in constants.items():
            if str(v) == target_str:
                if found == occurrence: target_cid = k; break
                found += 1
        if target_cid is None: return False
        constants[target_cid] = new_value
        const_blob = b''.join(NacRuntime._serialize_mep_constant(cid, val) for cid, val in sorted(constants.items()))
        new_blob = b'ORCH' + struct.pack('<II', len(bytecode), len(constants)) + bytecode + const_blob
        with open(nac_path, 'r+b') as f:
            f.seek(10)
            offsets = struct.unpack('<H10Q', f.read(82))[1:]
            f.seek(offsets[7])
            f.write(new_blob)
        return True

# --- END OF FILE MEP_compiler.py ---