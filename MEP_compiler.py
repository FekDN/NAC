# MEP_compiler.py
#
# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
# Licensed under the Apache License, Version 2.0
#
# This file contains the MEPCompiler, a compiler class that translates
# high-level Python calls into binary bytecode for the Model Execution
# Pipeline (MEP) virtual machine. It is the primary tool for creating
# executable MEP plans.

import struct
from typing import Dict, Any, List, Optional, Tuple

class MEPCompiler:
    """
    A compiler helper class to generate MEP ISA v1.0 bytecode 
    from high-level, Python-like method calls.

    This class manages the entire compilation state, including:
    - Allocation of context slots (registers).
    - Management of the constants pool.
    - Generation of binary instructions.
    - Handling of labels and calculating jump offsets (the linking phase).
    """
    def __init__(self):
        """Initializes a new compiler instance."""
        self.bytecode = bytearray()
        self.constants: Dict[str, Tuple[int, Any]] = {}
        self.next_const_id = 0
        self.context_vars: Dict[str, int] = {}
        self.next_context_key = 0
        
        # For handling jumps and labels during the linking phase
        self._labels: Dict[str, int] = {}
        self._jumps: Dict[str, List[int]] = {}
        
        # A stack to manage nested loops and calculate offsets for 'break' statements
        self._loop_stack: List[Dict[str, Any]] = []


    def _get_const_id(self, value: Any) -> int:
        """
        Adds a value to the constant pool if it doesn't exist and returns its ID.
        """
        # Use the string representation for caching simple values.
        # Note: This is not robust for complex, non-string-representable objects.
        val_key = str(value)
        if val_key not in self.constants:
            if self.next_const_id > 65535:
                raise OverflowError("Constant pool limit (65536) exceeded.")
            self.constants[val_key] = (self.next_const_id, value)
            self.next_const_id += 1
        return self.constants[val_key][0]

    def _get_context_key(self, name: str) -> int:
        """
        Assigns a context slot (key) to a variable name if it's new, and returns the key.
        """
        if name not in self.context_vars:
            if self.next_context_key > 255:
                raise OverflowError("Context variable limit (256) exceeded.")
            self.context_vars[name] = self.next_context_key
            self.next_context_key += 1
        return self.context_vars[name]

    # --- 0x00-0x0F: Data Sources & Parameters ---

    def src_user_prompt(self, out_var: str, prompt_text: str, data_type: int = 0):
        """0x02: SRC_USER_PROMPT. Prompts the user for input."""
        key = self._get_context_key(out_var)
        prompt_id = self._get_const_id(prompt_text)
        self.bytecode.append(0x02)
        # Parameters: out_key(1), data_type(1), prompt_const_id(2)
        self.bytecode += struct.pack('<BBH', key, data_type, prompt_id)

    def src_constant(self, out_var: str, value: Any):
        """0x04: SRC_CONSTANT. Loads a predefined constant into the context."""
        key = self._get_context_key(out_var)
        const_id = self._get_const_id(value)
        self.bytecode.append(0x04)
        # Parameters: out_key(1), const_id(2)
        self.bytecode += struct.pack('<BH', key, const_id)

    # --- 0x10-0x1F: Resource & Environment Management ---

    def res_load_model(self, model_id: int, model_path: str):
        """0x10: RES_LOAD_MODEL. Loads a model from a resource path."""
        path_const_id = self._get_const_id(model_path)
        self.bytecode.append(0x10)
        # Parameters: model_id(1), path_const_id(2)
        self.bytecode += struct.pack('<BH', model_id, path_const_id)

    def res_load_datafile(self, out_var: str, file_path: str, file_type: int):
        """0x11: RES_LOAD_DATAFILE. Loads a data file (e.g., .npy, .json)."""
        out_key = self._get_context_key(out_var)
        path_const_id = self._get_const_id(file_path)
        self.bytecode.append(0x11)
        # Parameters: out_key(1), file_type(1), path_const_id(2)
        self.bytecode += struct.pack('<BBH', out_key, file_type, path_const_id)

    def res_load_dynamic(self, out_var: str, path_var: str, file_type: int):
        """0x13: RES_LOAD_DYNAMIC. Loads a file whose path comes from a context variable.

        Unlike res_load_datafile (0x11) where the path is a compile-time string
        constant, here *path_var* is the name of a runtime MEP variable that holds
        the path string (typically the result of src_user_prompt).

        file_type codes (same as res_load_datafile):
            2  — .npy binary tensor, loaded as-is via np.load()
            3  — image file (jpg/png/…) with ImageNet preprocessing
                 → NacRuntime.preprocess_image_for_imagenet() → (1,3,224,224) float32
        """
        out_key  = self._get_context_key(out_var)
        path_key = self._get_context_key(path_var)
        self.bytecode.append(0x13)
        # Parameters: out_key(1), path_key(1), file_type(1)
        self.bytecode += struct.pack('<BBB', out_key, path_key, file_type)

    def res_load_extern(self, out_var: str, res_type: int, resource_id: Any):
        """0x12: RES_LOAD_EXTERN (универсальный).
        
        res_type:
            0 - извлечь компонент из уже загруженной модели (NacRuntime)
            1 - standalone ресурс (по пути/имени)
        """
        key = self._get_context_key(out_var)
        res_id_const = self._get_const_id(resource_id)
        
        self.bytecode.append(0x12)
        # Параметры: out_key(1), res_type(1), res_id_const_id(2)
        self.bytecode += struct.pack('<BBH', key, res_type, res_id_const)

    # --- 0x20-0x2F: Preprocessing ---
    
    def preproc_encode(self, proc_var: str, in_var: str, out_var: str):
        """0x20: PREPROC_ENCODE. Applies an encoder (e.g., tokenizer) to data."""
        proc_key, in_key, out_key = self._get_context_key(proc_var), self._get_context_key(in_var), self._get_context_key(out_var)
        self.bytecode.append(0x20)
        self.bytecode += struct.pack('<BBB', proc_key, in_key, out_key)
        
    def preproc_decode(self, proc_var: str, in_var: str, out_var: str):
        """0x21: PREPROC_DECODE. Applies a decoder to data."""
        proc_key, in_key, out_key = self._get_context_key(proc_var), self._get_context_key(in_var), self._get_context_key(out_var)
        self.bytecode.append(0x21)
        self.bytecode += struct.pack('<BBB', proc_key, in_key, out_key)

    def preproc_get_id(self, proc_var: str, item_str: str, out_var: str):
        """0x22: PREPROC_GET_ID. Gets a special ID (e.g., <eos>) from a preprocessor."""
        proc_key = self._get_context_key(proc_var)
        item_const_id = self._get_const_id(item_str)
        out_key = self._get_context_key(out_var)
        self.bytecode.append(0x22)
        self.bytecode += struct.pack('<BHB', proc_key, item_const_id, out_key)

    def string_format(self, out_var: str, format_str: str, in_vars: List[str]):
        """0x2A: STRING_FORMAT. Creates a formatted string."""
        out_key, format_id = self._get_context_key(out_var), self._get_const_id(format_str)
        in_keys = [self._get_context_key(v) for v in in_vars]
        self.bytecode.append(0x2A)
        self.bytecode += struct.pack('<BHB', out_key, format_id, len(in_keys))
        if in_keys: self.bytecode += struct.pack(f'<{len(in_keys)}B', *in_keys)

    # --- 0x30-0x4F: Tensor Processing ---

    def tensor_create(self, from_py: Optional[dict] = None, arange: Optional[dict] = None, ones: Optional[dict] = None):
        """0x30: TENSOR_CREATE. Creates a new tensor."""
        self.bytecode.append(0x30)
        if from_py:
            out_key, in_key = self._get_context_key(from_py['out_var']), self._get_context_key(from_py['in_var'])
            self.bytecode += struct.pack('<BBBB', out_key, from_py['dtype_code'], 0, in_key)
        elif arange:
            out_key, end_key = self._get_context_key(arange['out_var']), self._get_context_key(arange['end_var'])
            self.bytecode += struct.pack('<BBBB', out_key, arange['dtype_code'], 1, end_key)
        elif ones:
            out_key, shape_key = self._get_context_key(ones['out_var']), self._get_context_key(ones['shape_var'])
            self.bytecode += struct.pack('<BBBB', out_key, ones['dtype_code'], 2, shape_key)
        else: raise ValueError("A tensor creation type (from_py, arange, ones) must be specified.")

    def tensor_manipulate(self, op_type: str, out_var: str, in_var: str, **kwargs):
        """0x38: TENSOR_MANIPULATE. Performs simple tensor manipulations (pad, reshape, etc.)."""
        op_map = {'pad': 1} # 0 is reserved for future reshape
        op_code = op_map.get(op_type)
        if op_code is None: raise ValueError(f"Unknown TENSOR_MANIPULATE op_type: {op_type}")
        
        out_key, in_key = self._get_context_key(out_var), self._get_context_key(in_var)
        self.bytecode.append(0x38)
        self.bytecode += struct.pack('<BBB', op_code, out_key, in_key)

        if op_type == 'pad':
            pad_width_key = self._get_context_key(kwargs['pad_width_var'])
            const_val_key = self._get_context_key(kwargs['const_val_var'])
            self.bytecode += struct.pack('<BB', pad_width_key, const_val_key)

    def tensor_combine(self, op_type: str, out_var: str, in_vars: List[str], **kwargs):
        """0x39: TENSOR_COMBINE. Combines multiple tensors (concat, stack)."""
        op_map = {'concat': 0}
        op_code = op_map.get(op_type)
        if op_code is None: raise ValueError(f"Unknown TENSOR_COMBINE op_type: {op_type}")
        
        out_key, in_keys = self._get_context_key(out_var), [self._get_context_key(v) for v in in_vars]
        self.bytecode.append(0x39)
        self.bytecode += struct.pack('<BBB', op_code, out_key, len(in_keys))
        self.bytecode += struct.pack(f'<{len(in_keys)}B', *in_keys)

        if op_type == 'concat':
            axis_key = self._get_context_key(kwargs['axis_var'])
            self.bytecode += struct.pack('<B', axis_key)

    def tensor_info(self, op_type: str, out_var: str, in_var: str, **kwargs):
        """0x3A: TENSOR_INFO. Gets metadata from a tensor (shape, dim, etc.)."""
        op_map = {'shape': 0, 'dim': 1, 'to_py': 2}
        op_code = op_map.get(op_type)
        if op_code is None: raise ValueError(f"Unknown TENSOR_INFO op_type: {op_type}")
        out_key, in_key = self._get_context_key(out_var), self._get_context_key(in_var)
        self.bytecode.append(0x3A)
        self.bytecode += struct.pack('<BBB', op_code, out_key, in_key)
        if op_type == 'dim':
            dim_key = self._get_context_key(kwargs['dim_idx_var'])
            self.bytecode += struct.pack('<B', dim_key)

    def tensor_extract(self, out_var: str, in_tensor_var: str, in_idx_var: str):
        """0x3B: TENSOR_EXTRACT. Extracts an element or slice by index."""
        out_key, in_tensor_key, in_idx_key = self._get_context_key(out_var), self._get_context_key(in_tensor_var), self._get_context_key(in_idx_var)
        self.bytecode.append(0x3B)
        self.bytecode += struct.pack('<BBB', out_key, in_tensor_key, in_idx_key)

    # --- 0x50-0x5F: System & External Calls ---

    def sys_copy(self, out_var: str, in_var: str):
        """0x59: SYS_COPY. Copies an object within the context."""
        out_key, in_key = self._get_context_key(out_var), self._get_context_key(in_var)
        self.bytecode.append(0x59)
        self.bytecode += struct.pack('<BB', out_key, in_key)

    def sys_debug_print(self, var_name: str, msg: str = ""):
        """0x5F: SYS_DEBUG_PRINT. Prints a context value for debugging."""
        key = self._get_context_key(var_name)
        msg_id = self._get_const_id(msg or f"{var_name}:")
        self.bytecode.append(0x5F)
        self.bytecode += struct.pack('<BH', key, msg_id)

    # --- 0x60-0x7F: Post-processing & Logic ---

    def math_unary(self, op_type: str, out_var: str, in_var: str):
        """0x60: MATH_UNARY. Performs a unary math operation (e.g., softmax)."""
        op_map = {'softmax': 0}
        op_code = op_map.get(op_type)
        if op_code is None: raise ValueError(f"Unknown unary math op_type: {op_type}")
        out_key, in_key = self._get_context_key(out_var), self._get_context_key(in_var)
        self.bytecode.append(0x60)
        self.bytecode += struct.pack('<BBB', op_code, out_key, in_key)

    def math_binary(self, op_type: str, out_var: str, in_var1: str, in_var2: str):
        """0x61: MATH_BINARY. Performs a binary arithmetic operation."""
        op_map = {'add': 0, 'sub': 1, 'mul': 2}
        op_code = op_map.get(op_type)
        if op_code is None: raise ValueError(f"Unknown binary math op_type: {op_type}")
        out_key, in_key1, in_key2 = self._get_context_key(out_var), self._get_context_key(in_var1), self._get_context_key(in_var2)
        self.bytecode.append(0x61)
        self.bytecode += struct.pack('<BBBB', op_code, out_key, in_key1, in_key2)

    def math_aggregate(self, op_type: str, out_var: str, in_var: str):
        """0x62: MATH_AGGREGATE. Performs an aggregate operation (e.g., argmax)."""
        op_map = {'argmax': 0}
        op_code = op_map.get(op_type)
        if op_code is None: raise ValueError(f"Unknown aggregate math op_type: {op_type}")
        out_key, in_key = self._get_context_key(out_var), self._get_context_key(in_var)
        self.bytecode.append(0x62)
        self.bytecode += struct.pack('<BBB', op_code, out_key, in_key)

    def logic_compare(self, op_type: str, out_var: str, in_var1: str, in_var2: str):
        """0x68: LOGIC_COMPARE. Compares two values."""
        op_map = {'eq': 0, 'neq': 1, 'gt': 2, 'lt': 3}
        op_code = op_map.get(op_type)
        if op_code is None: raise ValueError(f"Unknown comparison op_type: {op_type}")
        out_key, in_key1, in_key2 = self._get_context_key(out_var), self._get_context_key(in_var1), self._get_context_key(in_var2)
        self.bytecode.append(0x68)
        self.bytecode += struct.pack('<BBBB', op_code, out_key, in_key1, in_key2)

    # --- 0x80-0x8F: Model Execution ---

    def model_run_static(self, model_id: int, in_vars: List[str], out_vars: List[str]):
        """0x80: MODEL_RUN_STATIC. Runs a model with a fixed I/O contract."""
        in_keys = [self._get_context_key(v) for v in in_vars]
        out_keys = [self._get_context_key(v) for v in out_vars]
        self.bytecode.append(0x80)
        self.bytecode += struct.pack('<B', model_id)
        self.bytecode += struct.pack('<B', len(in_keys))
        if in_keys: self.bytecode += struct.pack(f'<{len(in_keys)}B', *in_keys)
        self.bytecode += struct.pack('<B', len(out_keys))
        if out_keys: self.bytecode += struct.pack(f'<{len(out_keys)}B', *out_keys)
        
    # --- 0xA0-0xAF: Flow Control ---

    def flow_loop_start(self, counter_var: str):
        """0xA0: FLOW_LOOP_START. Marks the beginning of a loop."""
        counter_key = self._get_context_key(counter_var)
        self.bytecode.append(0xA0)
        self.bytecode.append(counter_key)
        # Save the starting position of the loop body for the end instruction to jump back to.
        # Also prepare a list to hold placeholders for any 'break' statements.
        self._loop_stack.append({'start_pos': len(self.bytecode), 'break_placeholders': []})

    def flow_loop_end(self):
        """0xA1: FLOW_LOOP_END. Marks the end of a loop body."""
        if not self._loop_stack:
            raise RuntimeError("FLOW_LOOP_END called without a matching FLOW_LOOP_START.")
        
        loop_info = self._loop_stack[-1]
        
        # --- Backpatching for 'break' statements ---
        # The target for a 'break' is the instruction immediately *after* this FLOW_LOOP_END.
        # Length of FLOW_LOOP_END is 1 (flag) + 2 (offset) = 3 bytes.
        break_target_pos = len(self.bytecode) + 3
        for placeholder_pos in loop_info['break_placeholders']:
            # The jump offset is from the byte *after* the placeholder to the target.
            jump_offset = break_target_pos - (placeholder_pos + 2)
            self.bytecode[placeholder_pos:placeholder_pos+2] = struct.pack('<h', jump_offset)

        # --- Calculate backward jump for the loop itself ---
        start_pos = loop_info['start_pos']
        # The jump offset is from the byte *after* this instruction back to the start of the loop body.
        jump_offset = start_pos - (len(self.bytecode) + 3)
        
        self.bytecode.append(0xA1)
        self.bytecode += struct.pack('<h', jump_offset)
        self._loop_stack.pop()

    def flow_branch_if(self, cond_var: str, jump_label: str):
        """0xA8: FLOW_BRANCH_IF. Jumps to a label if the condition is true."""
        cond_key = self._get_context_key(cond_var)
        self.bytecode.append(0xA8)
        self.bytecode.append(cond_key)
        # Add a 2-byte placeholder for the jump offset.
        offset_pos = len(self.bytecode)
        self.bytecode += struct.pack('<h', 0)
        # Record this position to be patched later during the linking stage.
        if jump_label not in self._jumps: self._jumps[jump_label] = []
        self._jumps[jump_label].append(offset_pos)

    def flow_break_loop_if(self, cond_var: str):
        """0xA9: FLOW_BREAK_LOOP_IF. Exits the current loop if the condition is true."""
        if not self._loop_stack:
            raise RuntimeError("FLOW_BREAK_LOOP_IF called outside of a loop.")
        
        cond_key = self._get_context_key(cond_var)
        self.bytecode.append(0xA9)
        self.bytecode.append(cond_key)
        
        # Add a placeholder for the jump offset and record its position.
        # This will be patched by the corresponding FLOW_LOOP_END instruction.
        offset_pos = len(self.bytecode)
        self.bytecode += struct.pack('<h', 0)
        self._loop_stack[-1]['break_placeholders'].append(offset_pos)

    def place_label(self, label_name: str):
        """Places a named label at the current position in the bytecode."""
        if label_name in self._labels:
            raise NameError(f"Label '{label_name}' is already defined.")
        self._labels[label_name] = len(self.bytecode)

    # --- 0xE0-0xEF: Data Serialization & Conversion ---

    def serialize_object(self, out_var: str, in_var: str, format_type: int):
        """0xE0: SERIALIZE_OBJECT. Serializes an object into bytes or a string."""
        out_key, in_key = self._get_context_key(out_var), self._get_context_key(in_var)
        self.bytecode.append(0xE0)
        # Parameters: out_key(1), in_key(1), format_type(1)
        self.bytecode += struct.pack('<BBB', out_key, in_key, format_type)

    # --- 0xF0-0xFF: I/O Sinks & Termination ---

    def io_write(self, in_var: str, dest_type: int, dest_var: Optional[str] = None, write_mode: int = 0):
        """0xF0: IO_WRITE. Writes data to a specified destination (e.g., STDOUT, FILE)."""
        in_key = self._get_context_key(in_var)
        # For STDOUT/STDERR, dest_key is not used, so we can use a placeholder.
        dest_key = self._get_context_key(dest_var) if dest_var else 0
        self.bytecode.append(0xF0)
        # Parameters: in_key(1), dest_type(1), dest_key(1), write_mode(1)
        self.bytecode += struct.pack('<BBBB', in_key, dest_type, dest_key, write_mode)

    def exec_return(self, var_names: List[str]):
        """0xFE: EXEC_RETURN. Terminates execution and returns specified values."""
        keys = [self._get_context_key(v) for v in var_names]
        self.bytecode.append(0xFE)
        self.bytecode.append(len(keys))
        if keys: self.bytecode += struct.pack(f'<{len(keys)}B', *keys)

    def exec_halt(self):
        """0xFF: EXEC_HALT. Immediately halts execution."""
        self.bytecode.append(0xFF)

    # --- Compilation Finalization ---

    def get_program(self) -> Tuple[bytes, Dict[int, Any]]:
        """
        Finalizes the bytecode, performs linking, and returns the program.

        Returns:
            A tuple containing:
            - The final, linked bytecode as a bytes object.
            - The map of constant IDs to their Python values for the runtime.
        """
        if self._loop_stack:
            raise RuntimeError("Not all loops were closed with flow_loop_end().")
        
        # --- Linking Stage ---
        # Resolve all forward jumps (from FLOW_BRANCH_IF) by patching their offsets.
        if self._jumps:
            for label_name, jump_positions in self._jumps.items():
                if label_name not in self._labels:
                    raise NameError(f"Jump label '{label_name}' was never placed.")
                
                target_pos = self._labels[label_name]
                for offset_pos in jump_positions:
                    # Calculate the relative offset from the byte *after* the offset placeholder.
                    jump_offset = target_pos - (offset_pos + 2)
                    if not -32768 <= jump_offset <= 32767:
                        raise OverflowError("Jump offset exceeds i16 range.")
                    # Patch the bytecode with the calculated offset.
                    self.bytecode[offset_pos:offset_pos+2] = struct.pack('<h', jump_offset)
        
        # Prepare the constants map for the interpreter.
        const_map_for_runtime = {cid: val for _, (cid, val) in self.constants.items()}
        
        return bytes(self.bytecode), const_map_for_runtime

    def save_to_file(self, mep_path: str = "hello.mep", constants_path: str = "hello.constants.json"):
        """Сохраняет скомпилированный план и пул констант в файлы."""
        program, const_map = self.get_program()
        
        with open(mep_path, "wb") as f:
            f.write(program)
        
        if constants_path:
            import json
            with open(constants_path, "w", encoding="utf-8") as f:
                json.dump(const_map, f, ensure_ascii=False, indent=2)
        
        print(f"✅ MEP успешно сохранён:")
        print(f"   • Байт-код:     {mep_path}  ({len(program)} байт)")
        print(f"   • Константы:    {constants_path}  ({len(const_map)} записей)")