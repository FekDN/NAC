# --- START OF FILE MEP_interpreter.py ---
# Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

import struct
import sys
import numpy as np
from typing import List, Dict, Any
import json
import io
from PIL import Image
from NAC_kernels import softmax

class MEPInterpreter:
    def __init__(self, execution_plan: bytes, constants_pool: Dict[int, Any], pre_answers: List[str] = None, exec_mode: str = 'infer_train'):
        print("--- Initializing the MEP interpreter (v1.1) ---")
        self.plan       = execution_plan
        self.constants  = constants_pool
        self.ip         = 0
        self.context: List[Any] = [None] * 256
        self.resources: Dict[int, Any] = {}
        self.return_value: Any = None
        self.running    = False
        self.exec_mode  = exec_mode
        self.loop_stack: List[Dict[str, Any]] = []
        self._pre_answers: List[str] = list(pre_answers) if pre_answers else []

        self._instr_lengths: Dict[int, int] = {
            0x02: 4, 0x03: 3, 0x04: 3, 0x05: 1,
            0x10: 3, 0x11: 4, 0x12: 4, 0x13: 3, 0x18: 3, 0x1F: 2,
            0x20: 3, 0x21: 3, 0x22: 4, 0x2A: -1,
            0x30: 4, 0x31: 5, 0x38: -1, 0x39: -1, 0x3A: -1, 0x3B: 3,
            0x50: -1, 0x51: -1, 0x59: 2, 0x5F: 3,
            0x60: 3, 0x61: 4, 0x62: 3, 0x68: 4,
            0x70: 4, 0x71: 4,
            0x80: -1, 0x81: -1, 0x82: -1, 0x83: 1, 0x85: 3,
            0xA0: 1, 0xA1: 2, 0xA8: 3, 0xA9: 3,
            0xE0: 3, 0xE1: 3,
            0xF0: 4,
            0xFE: -1, 0xFF: 0,
        }

        self.handlers = {
            0x02: self._handle_src_user_prompt,
            0x04: self._handle_src_constant,
            0x05: self._handle_src_exec_mode,
            0x10: self._handle_res_load_model,
            0x11: self._handle_res_load_datafile,
            0x12: self._handle_res_load_extern,
            0x13: self._handle_res_load_dynamic,
            0x1F: self._handle_res_unload,
            0x20: self._handle_preproc_encode,
            0x21: self._handle_preproc_decode,
            0x22: self._handle_preproc_get_id,
            0x2A: self._handle_string_format,
            0x30: self._handle_tensor_create,
            0x38: self._handle_tensor_manipulate,
            0x39: self._handle_tensor_combine,
            0x3A: self._handle_tensor_info,
            0x3B: self._handle_tensor_extract,
            0x59: self._handle_sys_copy,
            0x5F: self._handle_sys_debug_print,
            0x60: self._handle_math_unary,
            0x61: self._handle_math_binary,
            0x62: self._handle_math_aggregate,
            0x68: self._handle_logic_compare,
            0x70: self._handle_analysis_top_k,
            0x71: self._handle_analysis_sample,
            0x80: self._handle_model_run_static,
            0x82: self._handle_model_train_step,
            0x83: self._handle_model_zero_grad,
            0x85: self._handle_model_save_weights,
            0xA0: self._handle_flow_loop_start,
            0xA1: self._handle_flow_loop_end,
            0xA8: self._handle_flow_branch_if,
            0xA9: self._handle_flow_break_loop_if,
            0xE0: self._handle_serialize_object,
            0xF0: self._handle_io_write,
            0xFE: self._handle_exec_return,
            0xFF: self._handle_exec_halt,
        }

    def _get_instruction_length(self, at_ip: int) -> int:
        flag = self.plan[at_ip]
        length = self._instr_lengths.get(flag)
        if length is None:
            raise NotImplementedError(f"Opcode 0x{flag:02x} unknown")

        if length != -1:
            return 1 + length

        if flag == 0x2A:
            return 5 + self.plan[at_ip + 4]
        if flag == 0x38:
            return 1 + (5 if self.plan[at_ip + 1] == 1 else 3)
        if flag == 0x39:
            return 4 + self.plan[at_ip + 3] + (1 if self.plan[at_ip + 1] == 0 else 0)
        if flag == 0x3A:
            return 4 + (1 if self.plan[at_ip + 1] == 1 else 0)
        if flag == 0x80:
            count_in = self.plan[at_ip + 2]
            return 4 + count_in + self.plan[at_ip + 3 + count_in]

        if flag == 0x82:
            # Простой и надёжный вариант - 15 байт базово + размеры списков
            count_in = self.plan[at_ip + 3]
            count_target = self.plan[at_ip + 4 + count_in]
            return 1 + 3 + count_in + 1 + count_target + 1 + 1 + 1 + 2 + 2

        if flag == 0xFE:
            return 2 + self.plan[at_ip + 1]

        raise NotImplementedError(f"Dynamic length for opcode 0x{flag:02x} not implemented")

    def _read_u8(self): val = self.plan[self.ip]; self.ip += 1; return val
    def _read_u16(self): val = struct.unpack('<H', self.plan[self.ip:self.ip+2])[0]; self.ip += 2; return val
    def _read_i16(self): val = struct.unpack('<h', self.plan[self.ip:self.ip+2])[0]; self.ip += 2; return val

    def _scalar(self, value: Any) -> Any:
        return value.item() if hasattr(value, 'item') else value

    def run(self):
        print("\n--- Launching the MEP plan execution ---")
        self.running = True
        while self.ip < len(self.plan) and self.running:
            flag = self._read_u8()
            #print(f"[MEP TRACE] IP={self.ip-1}, Opcode=0x{flag:02x}")   # <--- добавьте эту строку
            handler = self.handlers.get(flag)
            if handler: handler()
            else: raise NotImplementedError(f"Instruction 0x{flag:02x} not implemented.")
        print("--- The MEP plan has been completed. ---")
        return self.return_value

    def _handle_src_user_prompt(self):
        out_key, data_type, prompt_id = self._read_u8(), self._read_u8(), self._read_u16()
        prompt = self.constants[prompt_id]
        # When pre-answers are available, consume and echo them so the log is readable.
        # When the queue is empty, fall through to interactive input().
        if self._pre_answers:
            ans_str = str(self._pre_answers.pop(0))
            print(f"{prompt}{ans_str}")
        else:
            ans_str = input(prompt)
        if data_type == 1: val = int(ans_str)
        elif data_type == 2: val = float(ans_str)
        else: val = str(ans_str)
        self.context[out_key] = val

    def _handle_src_constant(self):
        out_key, const_id = self._read_u8(), self._read_u16()
        self.context[out_key] = self.constants[const_id]

    def _handle_src_exec_mode(self):
        self.context[self._read_u8()] = self.exec_mode

    def _handle_res_load_model(self):
        from NAC_run import NacRuntime
        model_id, path_id = self._read_u8(), self._read_u16()
        self.resources[model_id] = NacRuntime(self.constants[path_id])

    def _handle_res_load_extern(self):
        out_key, res_type, res_id_cid = self._read_u8(), self._read_u8(), self._read_u16()
        res_id = self.constants[res_id_cid]
        if res_type == 0: self.context[out_key] = self.resources[res_id].tokenizer

    def _handle_res_unload(self):
        res_type, id_or_key = self._read_u8(), self._read_u8()
        if res_type == 0: self.resources.pop(id_or_key, None)
        else: self.context[id_or_key] = None

    def _handle_res_load_datafile(self):
        out_key, file_type, path_cid = self._read_u8(), self._read_u8(), self._read_u16()
        if file_type == 2: self.context[out_key] = np.load(self.constants[path_cid])

    def _handle_res_load_dynamic(self):
        from NAC_run import NacRuntime
        out_key, path_key, file_type = self._read_u8(), self._read_u8(), self._read_u8()
        self.context[out_key] = NacRuntime.load_dynamic_file(self.context[path_key], file_type)

    def _handle_preproc_encode(self):
        proc_key, in_key, out_key = self._read_u8(), self._read_u8(), self._read_u8()
        tok = self.context[proc_key]
        self.context[out_key] = tok.run(tok.manifest, self.context[in_key])

    def _handle_preproc_decode(self):
        proc_key, in_key, out_key = self._read_u8(), self._read_u8(), self._read_u8()
        ids = self.context[in_key]
        if isinstance(ids, np.ndarray): ids = ids.flatten().tolist()
        if not isinstance(ids, list): ids = [self._scalar(ids)]
        self.context[out_key] = self.context[proc_key].decode(ids)

    def _handle_preproc_get_id(self):
        proc_key, item_cid, out_key = self._read_u8(), self._read_u16(), self._read_u8()
        self.context[out_key] = self.context[proc_key].get_token_id(self.constants[item_cid])

    def _handle_string_format(self):
        out_key, fmt_cid, count = self._read_u8(), self._read_u16(), self._read_u8()
        keys = [self._read_u8() for _ in range(count)]
        fmt = self.constants[fmt_cid]
        def _py(v): return v.item() if hasattr(v, 'item') else (v.tolist() if hasattr(v, 'tolist') else v)
        self.context[out_key] = fmt.format(*[_py(self.context[k]) for k in keys])

    def _handle_tensor_create(self):
        out_key, dtype_code, ctype = self._read_u8(), self._read_u8(), self._read_u8()
        dtype = {0: np.float32, 5: np.int64}.get(dtype_code, np.float32)
        if ctype == 0:
            val = self.context[self._read_u8()]
            arr = np.array(val, dtype=dtype)
            # Автоматически добавляем измерение batch_size = 1
            if arr.ndim == 0: 
                arr = arr.reshape(1, 1) # Скаляр -> (1, 1)
            elif arr.ndim == 1: 
                arr = np.expand_dims(arr, axis=0) # Список -> (1, N)
            self.context[out_key] = arr
        elif ctype == 1:
            # Используем _scalar для безопасного извлечения числа
            end_val = self._scalar(self.context[self._read_u8()])
            self.context[out_key] = np.arange(end_val, dtype=dtype).reshape(1, -1)
        elif ctype == 2:
            shape_val = self.context[self._read_u8()]
            # Для np.ones ожидается кортеж (tuple) размеров
            self.context[out_key] = np.ones(shape_val, dtype=dtype)

    def _handle_tensor_manipulate(self):
        op_type, out_key, in_key = self._read_u8(), self._read_u8(), self._read_u8()
        t = self.context[in_key]
        if op_type == 1:
            pw, cv = self._scalar(self.context[self._read_u8()]), self._scalar(self.context[self._read_u8()])
            self.context[out_key] = np.pad(t, [(0,0)] * (t.ndim - 1) + [(pw, 0)], 'constant', constant_values=cv)

    def _handle_tensor_combine(self):
        op_type, out_key, count = self._read_u8(), self._read_u8(), self._read_u8()
        tensors = [self.context[self._read_u8()] for _ in range(count)]
        if op_type == 0: self.context[out_key] = np.concatenate(tensors, axis=self._scalar(self.context[self._read_u8()]))

    def _handle_tensor_info(self):
        op_type, out_key, in_key = self._read_u8(), self._read_u8(), self._read_u8()
        t = self.context[in_key]
        if op_type == 0: self.context[out_key] = t.shape
        elif op_type == 1: self.context[out_key] = t.shape[self._scalar(self.context[self._read_u8()])]
        elif op_type == 2: self.context[out_key] = t.item()

    def _handle_tensor_extract(self):
        out_key, in_t_key, in_i_key = self._read_u8(), self._read_u8(), self._read_u8()
        t, idx = self.context[in_t_key], self._scalar(self.context[in_i_key])
        if t.ndim == 3: self.context[out_key] = t[0, idx, :]
        elif t.ndim == 2: self.context[out_key] = t[0, idx]
        elif t.ndim == 1: self.context[out_key] = t[idx]

    def _handle_sys_copy(self):
        out_key, in_key = self._read_u8(), self._read_u8()
        self.context[out_key] = self.context[in_key]

    def _handle_sys_debug_print(self):
        key, msg_cid = self._read_u8(), self._read_u16()
        print(f"[DEBUG] {self.constants[msg_cid]}: {self.context[key]}")

    def _handle_math_unary(self):
        op_type, out_key, in_key = self._read_u8(), self._read_u8(), self._read_u8()
        if op_type == 0: self.context[out_key] = softmax(self.context[in_key])

    def _handle_math_binary(self):
        op_type, out_key, k1, k2 = self._read_u8(), self._read_u8(), self._read_u8(), self._read_u8()
        v1, v2 = self.context[k1], self.context[k2]
        if op_type == 0: self.context[out_key] = np.add(v1, v2)
        elif op_type == 1: self.context[out_key] = np.subtract(v1, v2)
        elif op_type == 2: self.context[out_key] = np.multiply(v1, v2)

    def _handle_math_aggregate(self):
        op_type, out_key, in_key = self._read_u8(), self._read_u8(), self._read_u8()
        if op_type == 0: self.context[out_key] = np.argmax(self.context[in_key], axis=-1)

    def _handle_logic_compare(self):
        op_type, out_key, k1, k2 = self._read_u8(), self._read_u8(), self._read_u8(), self._read_u8()
        v1, v2 = self.context[k1], self.context[k2]
        if op_type == 0: self.context[out_key] = np.equal(v1, v2)
        elif op_type == 1: self.context[out_key] = np.not_equal(v1, v2)
        elif op_type == 2: self.context[out_key] = np.greater(v1, v2)
        elif op_type == 3: self.context[out_key] = np.less(v1, v2)

    def _handle_analysis_top_k(self):
        in_key, k_key, idx_key, vals_key = self._read_u8(), self._read_u8(), self._read_u8(), self._read_u8()
        logits = self.context[in_key].flatten()
        k = int(self._scalar(self.context[k_key]))
        top_idx = np.argpartition(logits, -k)[-k:]
        top_idx = top_idx[np.argsort(logits[top_idx])[::-1]]
        self.context[idx_key], self.context[vals_key] = top_idx, logits[top_idx]

    def _handle_analysis_sample(self):
        l_key, t_key, k_key, out_key = self._read_u8(), self._read_u8(), self._read_u8(), self._read_u8()
        logits = self.context[l_key].flatten().astype(np.float64)
        temp, k = float(self._scalar(self.context[t_key])), int(self._scalar(self.context[k_key]))
        if temp > 0: logits /= temp
        if 0 < k < len(logits): logits[logits < np.sort(logits)[-k]] = -np.inf
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        self.context[out_key] = int(np.random.choice(len(probs), p=probs))

    def _handle_model_run_static(self):
        model_id, count_in = self._read_u8(), self._read_u8()
        in_keys = [self._read_u8() for _ in range(count_in)]
        out_keys = [self._read_u8() for _ in range(self._read_u8())]
        outputs = self.resources[model_id].run([self.context[k] for k in in_keys])
        for i, ok in enumerate(out_keys): self.context[ok] = outputs[i]

    def _handle_model_train_step(self):
        model_id = self._read_u8()
        loss_type = self._read_u8()
        count_in = self._read_u8()
        in_keys = [self._read_u8() for _ in range(count_in)]

        target_count = self._read_u8()
        target_keys = [self._read_u8() for _ in range(target_count)]
        print(f"[DEBUG] Starting 0x82 at IP={self.ip-1}, next bytes: {self.plan[self.ip:self.ip+20]}")
        out_loss_key = self._read_u8()
        lr_key = self._read_u8()
        logits_key = self._read_u8()
        head_weight_name_id = self._read_u16()
        head_bias_name_id = self._read_u16()

        head_weight_name = self.constants.get(head_weight_name_id, "") if head_weight_name_id != 0 else ""
        head_bias_name = self.constants.get(head_bias_name_id, "") if head_bias_name_id != 0 else ""

        loss = self.resources[model_id].run_training_step(
            inputs=[self.context[k] for k in in_keys],
            targets=[self.context[k] for k in target_keys],
            loss_type=loss_type,
            lr=float(self._scalar(self.context[lr_key])),
            logits=self.context.get(logits_key) if logits_key != 0 else None,
            head_weight_name=head_weight_name,
            head_bias_name=head_bias_name
        )
        
        self.context[out_loss_key] = loss

    def _handle_model_zero_grad(self):
        self.resources[self._read_u8()].zero_grad()

    def _handle_model_save_weights(self):
        model_id, path_key, save_type = self._read_u8(), self._read_u8(), self._read_u8()
        self.resources[model_id].save_weights(self.context[path_key], save_type)

    def _handle_flow_loop_start(self):
        counter_key = self._read_u8()
        self.loop_stack.append({'counter_key': counter_key, 'count': self._scalar(self.context[counter_key])})

    def _handle_flow_loop_end(self):
        jump_offset = self._read_i16()
        info = self.loop_stack[-1]
        info['count'] -= 1
        if info['count'] > 0: self.ip += jump_offset
        else: self.loop_stack.pop()

    def _handle_flow_branch_if(self):
        cond_key, jump_offset = self._read_u8(), self._read_i16()
        if self._scalar(self.context[cond_key]): self.ip += jump_offset

    def _handle_flow_break_loop_if(self):
        cond_key, jump_offset = self._read_u8(), self._read_i16()
        if self._scalar(self.context[cond_key]):
            self.ip += jump_offset
            self.loop_stack.pop()

    def _handle_serialize_object(self):
        out_key, in_key, fmt_type = self._read_u8(), self._read_u8(), self._read_u8()
        obj = self.context[in_key]
        if fmt_type == 0: self.context[out_key] = str(obj)
        elif fmt_type == 1: self.context[out_key] = json.dumps(obj)
        elif fmt_type == 2:
            img_arr = obj.copy()
            if img_arr.dtype in (np.float32, np.float64):
                if img_arr.min() < 0: img_arr = (img_arr + 1.0) / 2.0
                img_arr = (img_arr * 255).clip(0, 255)
            if img_arr.ndim == 4 and img_arr.shape[0] == 1: img_arr = img_arr[0]
            if img_arr.ndim == 3 and img_arr.shape[0] in [1, 3, 4]: img_arr = np.transpose(img_arr, (1, 2, 0))
            if img_arr.shape[-1] == 1: img_arr = img_arr.squeeze(-1)
            buf = io.BytesIO()
            Image.fromarray(img_arr.astype(np.uint8)).save(buf, format="PNG")
            self.context[out_key] = buf.getvalue()

    def _handle_io_write(self):
        in_key, dest_type, dest_key, write_mode = self._read_u8(), self._read_u8(), self._read_u8(), self._read_u8()
        data = self.context[in_key]
        end_char = "" if write_mode == 2 else "\n"
        if dest_type == 0: print(data, end=end_char, flush=True)
        elif dest_type == 1: print(data, file=sys.stderr, end=end_char, flush=True)

    def _handle_exec_return(self):
        count = self._read_u8()
        keys = [self._read_u8() for _ in range(count)]
        self.return_value = self.context[keys[0]] if count == 1 else [self.context[k] for k in keys]
        self.running = False

    def _handle_exec_halt(self):
        self.running = False

# --- END OF FILE MEP_interpreter.py ---