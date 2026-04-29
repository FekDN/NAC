# --- START OF FILE NAC.py ---

# Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0
## torch==2.5.1
## torchvision==0.20.1
## transformers==4.57.3
## sympy==1.13.1

import os
import json
import re
import operator
import struct
import numpy as np
import torch
import torch.fx as fx
import torch.export
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import InputKind
import torchvision.models as models
from typing import List, Optional, Set, Tuple, Dict, Any, Union
from torch._functorch.aot_autograd import aot_module
import torch.utils._pytree as pytree
import traceback
from functools import partial
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download
import shutil
import warnings
warnings.simplefilter("ignore", FutureWarning)

from TISA_tokenizer import TISACompiler
from NAC_kernels import NAC_OPS, NAC_OPS_REVERSED, CUSTOM_OP_ID_START

class NoIndent:
    def __init__(self, value): self.value = value

class CompactJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs); self.indentation_level = 0
    def encode(self, o):
        if isinstance(o, (list, tuple)):
            is_no_indent_list = any(isinstance(item, NoIndent) for item in o)
            if not o: return "[]"
            if is_no_indent_list: return "[" + ", ".join([json.dumps(item.value) for item in o]) + "]"
            self.indentation_level += 1; output = [self.indent_str + self.encode(item) for item in o]; self.indentation_level -= 1
            return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"
        elif isinstance(o, dict):
            if not o: return "{}"
            self.indentation_level += 1; output = [self.indent_str + f"{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()]; self.indentation_level -= 1
            return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"
        elif isinstance(o, NoIndent): return json.dumps(o.value)
        else: return json.dumps(o)
    @property
    def indent_str(self) -> str: return " " * (self.indentation_level * self.indent) if self.indent else ""

class ResultManager:
    def __init__(self, output_path: str = './'):
        self.output_path = output_path; os.makedirs(self.output_path, exist_ok=True)
        print(f"Output data will be saved to: {os.path.abspath(self.output_path)}")

    def save_registry(self, op_string_to_id: Dict[str, int], const_to_id: Dict[Any, int], perm_tuple_to_id: Dict[Tuple[str, ...], int]):
        try:
            sorted_op_items = sorted(op_string_to_id.items(), key=lambda item: item[1])
            canonical_map = {str(op_id): op_name for op_name, op_id in sorted_op_items}
            
            sorted_const_items = sorted(const_to_id.items(), key=lambda item: item[1])
            constants_map = {}
            for const, const_id in sorted_const_items:
                value_to_store = list(const) if isinstance(const, tuple) else const
                if isinstance(value_to_store, list):
                    constants_map[str(const_id)] = NoIndent(value_to_store)
                else:
                    constants_map[str(const_id)] = value_to_store
            
            sorted_perm_items = sorted(perm_tuple_to_id.items(), key=lambda item: item[1])
            permutations_map = {str(perm_id): "".join(perm_tuple) for perm_tuple, perm_id in sorted_perm_items}
            
            registry_data = {'canonical': canonical_map, 'constants': constants_map, 'permutations': permutations_map}
            with open(os.path.join(self.output_path, 'registry.json'), 'w', encoding='utf-8') as f:
                f.write(CompactJSONEncoder(indent=2).encode(registry_data))
            print(f"Global registry saved to {os.path.join(self.output_path, 'registry.json')}")
        except Exception as e: print(f"!!!!! ERROR saving registry: {e}"); traceback.print_exc()

    def _map_dtype_to_enum(self, dtype):
        return {torch.float32:0, torch.float64:1, torch.float16:2, torch.bfloat16:3, torch.int32:4, torch.int64:5, torch.int16:6, torch.int8:7, torch.uint8:8, torch.bool:9}.get(dtype, 255)

    @staticmethod
    def _serialize_mep_constant(const_id: int, val) -> bytes:
        type_code, length, value_bytes = 0, 0, b''
        if val is None:
            type_code = 0
        elif isinstance(val, bool):
            type_code, length, value_bytes = 1, 1, struct.pack('<?', val)
        elif isinstance(val, int):
            type_code, value_bytes = 2, struct.pack('<q', val)
            length = len(value_bytes)
        elif isinstance(val, float):
            type_code, value_bytes = 3, struct.pack('<d', val)
            length = len(value_bytes)
        elif isinstance(val, str):
            type_code, value_bytes = 4, val.encode('utf-8')
            length = len(value_bytes)
        elif isinstance(val, list):
            if not val:
                type_code, length = 5, 0
            elif all(isinstance(x, int) for x in val):
                type_code, length = 5, len(val)
                value_bytes = struct.pack(f'<{length}i', *val)
            else:
                type_code, length = 6, len(val)
                value_bytes = struct.pack(f'<{length}f', *map(float, val))
        else:
            raise TypeError(f"MEP constant type not serializable: {type(val)} = {val!r}")
        out = struct.pack('<HBH', const_id, type_code, length)
        if value_bytes: out += value_bytes
        return out

    def save_model_nac(self, model_name, used_mappings, nodes, param_data, param_id_to_name, input_node_idx_to_name, d_model=0, tokenizer_manifest=None, tokenizer_resources=None, quant_method='none', store_weights_internally=True, io_counts=(0,0), memory_map=None, mep_program=None, trng_nodes=None):
        filepath = os.path.join(self.output_path, f"{model_name}.nac")
        try:
            with open(filepath, 'wb') as f:
                f.write(b'NAC\x02') 
                quant_map = {'none': 0, 'FP16': 1, 'INT8_TENSOR': 2, 'INT8_CHANNEL': 3, 'BLOCK_FP8': 4}
                quant_id = quant_map.get(quant_method, 0)
                if store_weights_internally: quant_id |= 0x80
                has_trng = bool(trng_nodes)
                if has_trng: quant_id |= 0x40 

                f.write(struct.pack('<B', quant_id))
                f.write(struct.pack('<HHB', io_counts[0], io_counts[1], 0))
                
                offsets_header_format = '<H10Q' 
                f.write(b'\0' * struct.calcsize(offsets_header_format)) 
                offsets = {}

                offsets['mmap'] = f.tell()
                f.write(b'MMAP')
                memory_map = memory_map or []
                f.write(struct.pack('<I', len(memory_map)))
                action_codes = {'SAVE_RESULT': 10, 'FREE': 20, 'FORWARD': 30, 'PRELOAD': 40, 'SAVE_FOR_GRAD': 15, 'FREE_AFTER_TRNG': 25}
                for record in memory_map:
                    instr_id = record['instr_id']
                    commands = record['commands']
                    f.write(struct.pack('<HB', instr_id, len(commands)))
                    for cmd in commands:
                        action_type_code = action_codes.get(cmd['action'], 10)
                        target_id = cmd['target_id']
                        f.write(struct.pack('<BH', action_type_code, target_id))

                offsets['ops'] = f.tell()
                f.write(b'OPS ')
                f.write(struct.pack('<I', len(nodes)))
                for node in nodes:
                    A, B, C, D = node['A'], node['B'], node['C'], node['D']
                    f.write(struct.pack('<BB', A, B))
                    if C: f.write(struct.pack(f'<{len(C)}h', *C))
                    if D: f.write(struct.pack(f'<{len(D)}h', *D))
                
                offsets['cmap'] = f.tell()
                f.write(b'CMAP')
                custom_cmap = {int(k): v for k, v in used_mappings['canonical'].items() if int(k) >= CUSTOM_OP_ID_START}
                f.write(struct.pack('<I', len(custom_cmap)))
                for op_id, op_name in custom_cmap.items():
                    name_bytes = op_name.encode('utf-8')
                    f.write(struct.pack('<HB', op_id, len(name_bytes))); f.write(name_bytes)
                
                offsets['cnst'] = f.tell()
                f.write(b'CNST')
                constants = used_mappings['constants']
                f.write(struct.pack('<I', len(constants)))
                for const_id_str, const_val in constants.items():
                    const_id = int(const_id_str)
                    type_code, length, value_bytes = 0, 0, b''
                    if const_val is None: type_code = 0
                    elif isinstance(const_val, bool):
                        type_code, length, value_bytes = 1, 1, struct.pack('<?', const_val)
                    elif isinstance(const_val, int):
                        type_code, value_bytes = 2, struct.pack('<q', const_val); length = len(value_bytes)
                    elif isinstance(const_val, float):
                        type_code, value_bytes = 3, struct.pack('<d', const_val); length = len(value_bytes)
                    elif isinstance(const_val, str):
                        type_code, value_bytes = 4, const_val.encode('utf-8'); length = len(value_bytes)
                    elif isinstance(const_val, list):
                        if not const_val: type_code, length = 5, 0
                        elif all(isinstance(x, int) for x in const_val):
                            type_code, length = 5, len(const_val); value_bytes = struct.pack(f'<{length}i', *const_val)
                        elif all(isinstance(x, (int, float)) for x in const_val):
                            type_code, length = 6, len(const_val); value_bytes = struct.pack(f'<{length}f', *map(float, const_val))
                        else: raise TypeError("Unsupported list content")
                    f.write(struct.pack('<HBH', const_id, type_code, length))
                    if value_bytes: f.write(value_bytes)
                
                offsets['perm'] = f.tell()
                f.write(b'PERM')
                perm_map = {int(k): v for k,v in used_mappings['permutations'].items()}
                f.write(struct.pack('<I', len(perm_map)))
                for p_id, p_val_str in perm_map.items():
                    p_val_bytes = p_val_str.encode('utf-8')
                    f.write(struct.pack('<HB', p_id, len(p_val_bytes))); f.write(p_val_bytes)
                
                offsets['data'] = f.tell()
                f.write(b'DATA')
                f.write(struct.pack('<I', len(param_id_to_name)))
                for p_id, p_name in param_id_to_name.items():
                    name_bytes = p_name.encode('utf-8')
                    f.write(struct.pack('<HH', p_id, len(name_bytes))); f.write(name_bytes)
                f.write(struct.pack('<I', len(input_node_idx_to_name)))
                for i_idx, i_name in input_node_idx_to_name.items():
                    name_bytes = i_name.encode('utf-8')
                    f.write(struct.pack('<HH', i_idx, len(name_bytes))); f.write(name_bytes)
                if store_weights_internally:
                    f.write(struct.pack('<I', len(param_data)))
                    for p_id, data_tuple in param_data.items():
                        tensor, meta = (data_tuple, {}) if isinstance(data_tuple, torch.Tensor) else data_tuple
                        meta_binary = b''
                        dtype_enum = self._map_dtype_to_enum(tensor.dtype)
                        shape = list(tensor.shape)
                        rank = len(shape)
                        meta_binary += struct.pack('<BB', dtype_enum, rank)
                        if rank > 0: meta_binary += struct.pack(f'<{rank}I', *shape)
                            
                        quant_type_str = meta.get('quant_type', 'none')
                        quant_type_map = {'none': 0, 'FP16': 1, 'INT8_TENSOR': 2, 'INT8_CHANNEL': 3, 'BLOCK_FP8': 4}
                        quant_type_code = quant_type_map.get(quant_type_str, 0)
                        meta_binary += struct.pack('<B', quant_type_code)

                        if quant_type_code == 2:
                            meta_binary += struct.pack('<f', meta['scale'])
                        elif quant_type_code == 3:
                            scales = meta['scales']
                            if isinstance(scales, (float, int)): scales = [float(scales)]
                            meta_binary += struct.pack('<BI', meta['axis'], len(scales))
                            meta_binary += struct.pack(f'<{len(scales)}f', *scales)
                        elif quant_type_code == 4:
                            original_shape = meta['original_shape']
                            scales = meta['scales']
                            if isinstance(scales, (float, int)): scales = [float(scales)]
                            meta_binary += struct.pack('<HB', meta.get('block_size', 64), len(original_shape))
                            if len(original_shape) > 0: meta_binary += struct.pack(f'<{len(original_shape)}I', *original_shape)
                            meta_binary += struct.pack('<I', len(scales))
                            meta_binary += struct.pack(f'<{len(scales)}f', *scales)

                        data_bytes = tensor.numpy(force=True).tobytes()
                        f.write(struct.pack('<HIQ', p_id, len(meta_binary), len(data_bytes)))
                        f.write(meta_binary); f.write(data_bytes)
                else:
                    safetensors_path = os.path.join(self.output_path, f"{model_name}.safetensors")
                    tensors_to_save, metadata_to_save = {}, {}
                    for p_id, data_tuple in param_data.items():
                        name = param_id_to_name.get(p_id)
                        if not name: continue
                        tensor, meta = (data_tuple, None) if isinstance(data_tuple, torch.Tensor) else data_tuple
                        tensors_to_save[name] = tensor
                        if meta: metadata_to_save[name] = json.dumps(meta) 
                    save_file(tensors_to_save, safetensors_path, metadata=metadata_to_save)
                    print(f"Weights saved to {safetensors_path}")

                if tokenizer_manifest:
                    offsets['proc'] = f.tell()
                    f.write(b'PROC')
                    f.write(struct.pack('<I', len(tokenizer_manifest)))
                    f.write(tokenizer_manifest)

                offsets['orch'] = f.tell()
                f.write(b'ORCH')
                if mep_program is not None:
                    mep_bytecode, mep_constants = mep_program
                    f.write(struct.pack('<II', len(mep_bytecode), len(mep_constants)))
                    f.write(mep_bytecode)
                    for cid, cval in sorted(mep_constants.items()): f.write(ResultManager._serialize_mep_constant(cid, cval))
                else:
                    f.write(struct.pack('<II', 0, 0))

                if has_trng:
                    offsets['trng'] = f.tell()
                    f.write(b'TRNG')
                    f.write(struct.pack('<I', len(trng_nodes)))
                    for node in trng_nodes:
                        A, B, C, D = node['A'], node['B'], node['C'], node['D']
                        f.write(struct.pack('<BB', A, B))
                        if C: f.write(struct.pack(f'<{len(C)}h', *C))
                        if D: f.write(struct.pack(f'<{len(D)}h', *D))

                if tokenizer_resources and store_weights_internally:
                    offsets['rsrc'] = f.tell()
                    f.write(b'RSRC')
                    f.write(struct.pack('<I', len(tokenizer_resources)))
                    for filename, content in tokenizer_resources.items():
                        name_bytes = filename.encode('utf-8')
                        f.write(struct.pack('<H', len(name_bytes)))
                        f.write(name_bytes)
                        f.write(struct.pack('<I', len(content)))
                        f.write(content)

                f.seek(10)
                all_offsets = [
                    offsets.get('mmap', 0), offsets.get('ops',  0), offsets.get('cmap', 0),
                    offsets.get('cnst', 0), offsets.get('perm', 0), offsets.get('data', 0),
                    offsets.get('proc', 0), offsets.get('orch', 0), offsets.get('trng', 0),
                    offsets.get('rsrc', 0),
                ]
                f.write(struct.pack(offsets_header_format, d_model, *all_offsets))

            storage_method = "internally" if store_weights_internally else "externally"
            print(f"NAC v1.7 for '{model_name}' saved successfully (Quant: {quant_method}, Weights: {storage_method}, TRNG: {has_trng}).")
        except Exception as e:
            print(f"!!!!! ERROR saving NAC for {model_name}: {e}"); traceback.print_exc()

class ModelProcessor:
    const_to_id: Dict[Any, int] = {}
    CANONICAL_OP_MAP: Dict[str, Tuple[str, Optional[Tuple[int, ...]]]] = {
        "detach": ("nac.pass", None), 
        "dropout": ("nac.pass", None), 
        "pass_through": ("nac.pass", None),
        "lift_fresh_copy": ("nac.clone", None), 
        "copy": ("nac.clone", None),
        "true_divide": ("nac.div", None), 
        "div": ("nac.div", None),
        "mul": ("nac.mul", None),
        "where": ("nac.where", None),
        "unsafe_view": ("nac.view", None),
        "squeeze": ("nac.view", None), 
        "rsub": ("nac.sub", (1, 0)), 
        "masked_fill": ("nac.where", (1, 2, 0)),
        "mm": ("nac.matmul", None),
        "bmm": ("nac.matmul", None),
        "matmul": ("nac.matmul", None),
        "t": ("nac.transpose", None),
        "transpose": ("nac.transpose", None),
        "zeros": ("nac.zeros", None),
        "zeros_like": ("nac.zeros_like", None),
        "new_zeros": ("nac.new_zeros", None),
        "ones": ("nac.ones", None),
        "ones_like": ("nac.ones_like", None),
        "new_ones": ("nac.new_ones", None),
        "full": ("nac.full", None),
        "full_like": ("nac.full_like", None),
    }
    NAME_TO_OFFSET_CODE: Dict[str, str] = {
        'query': 'Q', 'key': 'K', 'value': 'V', 'attn_mask': 'M', 'mask': 'M',
        'bias': 'B', 'weight': 'W', 'input': 'T', 'self': 'T', 'other': 'T', 'tensor': 'T',
    }
    NAME_TO_CONST_CODE: Dict[str, str] = {
        'dim': 'A', 'axis': 'A', 'axes': 'A', 'shape': 'S', 'size': 'S',
    }

    def __init__(self, existing_registry: Optional[Dict] = None):
        self.canonical_operations: Set[str] = set()
        self.unique_constants: Set[Any] = set()
        self.unique_permutations: Set[Tuple[str, ...]] = set()
        
        self.param_data_map: Dict[int, Any] = {}
        self.param_id_to_name: Dict[int, str] = {}
        self.input_node_idx_to_name: Dict[int, str] = {}
        self.precomputed_nodes: List[Dict] = []
        self.trng_nodes: List[Dict] = [] 
        self.global_node_map: Dict[fx.Node, int] = {}
        self.op_string_to_id: Dict[str, int] = {}
        self.perm_tuple_to_id: Dict[Tuple[str, ...], int] = {}
        self.param_to_id: Dict[str, int] = {}
        self.data_input_nodes: Set[fx.Node] = set()
        self.param_nodes: Set[fx.Node] = set()
        self.lifted_const_nodes: Dict[fx.Node, Any] = {}
        self.io_counts = (0, 0)
        self.op_string_to_id = NAC_OPS_REVERSED.copy()
        
        self._initialize_special_ops()
        if not ModelProcessor.const_to_id: ModelProcessor.const_to_id[None] = 1
        
        self.unique_permutations.add(tuple())
        self.perm_tuple_to_id[tuple()] = 0

        if existing_registry: self._load_from_registry(existing_registry)

    def _initialize_special_ops(self):
        special_ops = {2:"<INPUT>", 3:"<OUTPUT>", 6:"<CONTROL_FLOW>", 7:"<CONVERGENCE>"}
        for id_val, name in special_ops.items():
            if id_val >= 10 and id_val in NAC_OPS: raise ValueError(f"Conflict Special OP ID")
            self.op_string_to_id[name] = id_val

    def _load_from_registry(self, registry: Dict):
        if 'canonical' in registry:
            for id_str, op_name in registry['canonical'].items():
                op_id = int(id_str)
                if op_id >= CUSTOM_OP_ID_START:
                    if op_name in self.op_string_to_id and self.op_string_to_id[op_name] != op_id: continue
                    self.op_string_to_id[op_name] = op_id
        if 'constants' in registry:
            for id_str, const_val in registry['constants'].items():
                hashable_const = self._get_hashable_const(const_val)
                ModelProcessor.const_to_id[hashable_const] = int(id_str)
        if 'permutations' in registry:
            for id_str, p_val_str in registry['permutations'].items():
                 self.perm_tuple_to_id[tuple(p_val_str)] = int(id_str)

    def _get_hashable_const(self, const: Any) -> Any:
        if isinstance(const, (fx.Node, torch.SymInt)): return None
        if isinstance(const, (torch.dtype, torch.device, torch.memory_format, torch.layout)):
            s = str(const)
            return s[6:] if s.startswith("torch.") else s
        if isinstance(const, torch.Tensor): return f"Tensor_{const.shape}_{const.dtype}"
        if isinstance(const, (list, tuple)): return tuple(self._get_hashable_const(item) for item in const)
        return const

    def _get_raw_signature(self, node: fx.Node) -> str:
        if node.op != 'call_function': return ""
        target = node.target
        return str(target).replace('::', '.') if isinstance(target, torch._ops.OpOverload) else getattr(target, '__name__', str(target))

    def _get_canonical_signature(self, node: fx.Node) -> str:
        if node.op == 'placeholder': return "<INPUT>"
        if node.op == 'output': return "<OUTPUT>"
        if node.op == 'get_attr': return "<INPUT>" 

        if node.op == 'call_method':
            raw_sig = f"aten.{node.target}.default"
        elif node.op == 'call_function':
            target = node.target
            raw_sig = str(target).replace('::', '.') if isinstance(target, torch._ops.OpOverload) else getattr(target, '__name__', str(target))
        else: return "<NONE>"

        if not raw_sig: return "<NONE>"
        if "aten.sym_size" in raw_sig: return raw_sig

        search_sig = raw_sig.replace("aten.", "")
        search_sig = re.sub(r'\.(default|Tensor|Scalar|int|dim|dim_IntList|using_ints|start|self)$', '', search_sig).strip('_')

        if search_sig in self.CANONICAL_OP_MAP: return self.CANONICAL_OP_MAP[search_sig][0]
        potential_nac_name = f"nac.{search_sig}"
        if potential_nac_name in NAC_OPS_REVERSED: return potential_nac_name
        return raw_sig

    def _get_argument_details(self, node: fx.Node) -> List[Tuple[str, Any, Optional[fx.Node]]]:
        details = []
        def _extract_val(v):
            if isinstance(v, torch.SymInt): return v.node.meta.get("val", v)
            if isinstance(v, (list, tuple)): return type(v)(_extract_val(i) for i in v)
            return v
        def _get_const_code(val, name: Optional[str] = None):
            if name and name in self.NAME_TO_CONST_CODE: return self.NAME_TO_CONST_CODE[name]
            if isinstance(val, bool): return 'b'
            if isinstance(val, int): return 'i'
            if isinstance(val, float): return 'f'
            if isinstance(val, str): return 's'
            if isinstance(val, (list, tuple)): return 'S'
            return 'c'
        def _get_code_from_node(n: fx.Node, arg_name: str) -> str:
            if arg_name in self.NAME_TO_OFFSET_CODE: return self.NAME_TO_OFFSET_CODE[arg_name]
            if n.op == 'placeholder':
                target_name = str(n.target)
                if 'weight' in target_name: return 'W'
                if 'bias' in target_name: return 'B'
                if 'mask' in target_name: return 'M'
            return 'T'

        is_getitem = (node.op == 'call_function' and node.target is operator.getitem)

        if hasattr(node.target, 'schema'):
            try:
                bound_args = node.target.schema().bind(*node.args, **node.kwargs)
                bound_args.apply_defaults()
                for arg in bound_args.arguments:
                    if arg.arg.is_out: continue 
                    name = arg.arg.name
                    val = _extract_val(arg.value)
                    items_to_process = val if isinstance(val, (list, tuple)) and any(isinstance(x, fx.Node) for x in val) else [val]
                    for item in items_to_process:
                        if isinstance(item, fx.Node): details.append((_get_code_from_node(item, name), None, item))
                        else:
                            if is_getitem and isinstance(item, float) and item == int(item): item = int(item)
                            details.append((_get_const_code(item, name), item, None))
                return details 
            except Exception: pass

        all_args = list(node.args) + list(node.kwargs.values())
        if node.op == 'output':
             all_args = all_args[0] if all_args and isinstance(all_args[0], (list, tuple)) else all_args
             self.io_counts = (self.io_counts[0], len(all_args))
        for arg in all_args:
            val = _extract_val(arg)
            items_to_process = val if isinstance(val, (list, tuple)) and any(isinstance(x, fx.Node) for x in val) else [val]
            for item in items_to_process:
                if isinstance(item, fx.Node): details.append(('T', None, item))
                else:
                    if is_getitem and isinstance(item, float) and item == int(item): item = int(item)
                    details.append((_get_const_code(item), item, None))
        return details

    def _get_canonical_argument_details(self, node: fx.Node) -> List[Tuple[str, Any, Optional[fx.Node]]]:
        arg_details = self._get_argument_details(node) 
        raw_sig = self._get_raw_signature(node)
        if raw_sig:
            search_sig = raw_sig.replace("aten.", "")
            search_sig = re.sub(r'\.(default|Tensor|Scalar|int|dim|dim_IntList|using_ints|start|self)$', '', search_sig).strip('_')
            if search_sig in self.CANONICAL_OP_MAP:
                _, permutation = self.CANONICAL_OP_MAP[search_sig]
                if permutation:
                    try: return [arg_details[i] for i in permutation]
                    except IndexError: pass
        return arg_details

    def _process_graph(self, graph: fx.Graph) -> List[Dict]:
        nodes = []
        for i, node in enumerate(graph.nodes):
            self.global_node_map[node] = i
            sig = self._get_canonical_signature(node) 
            A = self.op_string_to_id.get(sig, 0)
            B, C, D = 0, [], []

            if sig == "<INPUT>":
                if node in self.data_input_nodes: B = 0; self.input_node_idx_to_name[i] = node.target
                elif node in self.param_nodes: B = 1; C = [1, self.param_to_id.get(node.target, 0)]
                elif node in self.lifted_const_nodes: B = 3; C = [1, ModelProcessor.const_to_id.get(self._get_hashable_const(self.lifted_const_nodes[node]), 0)]
            
            elif sig == "<OUTPUT>":
                B = 0
                arg_details = self._get_argument_details(node)
                D = [self.global_node_map.get(arg_node) - i for _, _, arg_node in arg_details if arg_node]
                num_outputs = len(D)
                self.io_counts = (self.io_counts[0], num_outputs)
                C = [num_outputs] + ([0] * num_outputs)

            elif node.op in ('call_function', 'call_method'):
                arg_details = self._get_canonical_argument_details(node)
                parts = tuple(code for code, _, _ in arg_details)
                if parts: B = self.perm_tuple_to_id.get(parts, 0)
                consts = [self._get_hashable_const(val) for _, val, arg_node in arg_details if arg_node is None]
                const_ids = [ModelProcessor.const_to_id.get(c, 0) for c in consts]
                if const_ids: C = [len(const_ids)] + const_ids
                
                D = []
                for _, _, arg_node in arg_details:
                    if arg_node is not None: D.append(self.global_node_map.get(arg_node, -1) - i)
                    else: D.append(0)

            nodes.append({'A': A, 'B': B, 'C': C, 'D': D})
        return nodes

    def process_backward_graph(self, bw_graph: fx.Graph, aot_sync_map: dict, lr: float = 0.001):
        print("\n--- Compiling REAL Training Graph (TRNG) ---")
        
        lr_val = lr
        if lr_val not in ModelProcessor.const_to_id:
            ModelProcessor.const_to_id[lr_val] = max([0] + list(ModelProcessor.const_to_id.values())) + 1
        self.unique_constants.add(lr_val)
        lr_const_id = ModelProcessor.const_to_id[lr_val]

        one_val = 1.0
        if one_val not in ModelProcessor.const_to_id:
            ModelProcessor.const_to_id[one_val] = max([0] + list(ModelProcessor.const_to_id.values())) + 1
        self.unique_constants.add(one_val)
        one_const_id = ModelProcessor.const_to_id[one_val]

        node_list = list(bw_graph.nodes)
        user_to_ph = {}
        for n in node_list:
            if n.op == 'placeholder' and n.users:
                first_user = min(n.users, key=lambda u: node_list.index(u))
                if first_user not in user_to_ph: user_to_ph[first_user] = []
                user_to_ph[first_user].append(n)

        optimized_node_order = []
        for n in node_list:
            if n.op == 'placeholder':
                if not n.users: optimized_node_order.append(n)
                continue
            if n in user_to_ph:
                for ph in sorted(user_to_ph[n], key=lambda x: node_list.index(x)):
                    optimized_node_order.append(ph)
            optimized_node_order.append(n)

        param_gradients = []
        target_counter = 0
        trng = []
        bw_node_map = {}
        seen_inputs = {}

        bw_node_map["LR_CONST"] = len(trng)
        trng.append({'A': 2, 'B': 3, 'C': [1, lr_const_id], 'D': []})

        nac_pass_id = self.op_string_to_id.get("nac.pass", 10)

        for node in optimized_node_order:
            sig = self._get_canonical_signature(node)
            A = self.op_string_to_id.get(sig, 0)
            B, C, D = 0, [], []

            if sig == "<INPUT>":
                is_mapped = False

                # 1. Сначала проверяем на Tangent
                if node.meta.get('is_tangent'):
                    B, C = 3, [1, one_const_id]
                    is_mapped = True

                # 2. Если есть orig_node - используем жесткую привязку к исходному графу
                elif node.meta.get('orig_node') is not None:
                    orig_node = node.meta['orig_node']

                    if isinstance(orig_node, fx.Node):
                        if orig_node.op == 'placeholder':
                            if orig_node.target in self.param_to_id:
                                B, C = 1, [1, self.param_to_id[orig_node.target]]
                                is_mapped = True
                            else:
                                inv_input = {name: idx for idx, name in self.input_node_idx_to_name.items()}
                                idx = inv_input.get(str(orig_node.target))
                                if idx is not None:
                                    B, C = 4, [1, idx]
                                    is_mapped = True

                        elif orig_node.op == 'get_attr':
                            if orig_node.target in self.param_to_id:
                                B, C = 1, [1, self.param_to_id[orig_node.target]]
                                is_mapped = True

                        elif orig_node in self.global_node_map:
                            B, C = 4, [1, self.global_node_map[orig_node]]
                            is_mapped = True

                    else:
                        hc = self._get_hashable_const(orig_node)
                        if hc not in ModelProcessor.const_to_id:
                            ModelProcessor.const_to_id[hc] = max([0] + list(ModelProcessor.const_to_id.values())) + 1
                        B, C = 3, [1, ModelProcessor.const_to_id[hc]]
                        is_mapped = True

                # 3. Если это таргет тензор от LossWrapper
                if not is_mapped:
                    meta = node.meta.get('tensor_meta')
                    if meta is not None and meta.dtype in (torch.long, torch.int, torch.int32):
                        B, C = 5, [1, target_counter]
                        target_counter += 1
                        is_mapped = True

                if not is_mapped:
                    print(f"[TRNG] Warning: Unmapped input '{node.name}'. Fallback to 1.0.")
                    B, C = 3, [1, one_const_id]

                input_key = (B, tuple(C))
                if input_key in seen_inputs:
                    bw_node_map[node] = seen_inputs[input_key]
                    continue
                seen_inputs[input_key] = len(trng)

            elif sig == "<OUTPUT>":
                returns = list(node.args[0]) if isinstance(node.args[0], (list, tuple)) else [node.args[0]]
                req_grad_ids = getattr(bw_graph, 'req_grad_param_ids', [])
                for i, ret_node in enumerate(returns):
                    if ret_node is not None and i < len(req_grad_ids):
                        p_id = req_grad_ids[i]
                        if p_id > 0:
                            param_gradients.append((p_id, bw_node_map.get(ret_node, -1)))
                continue

            elif node.op in ('call_function', 'call_method'): 
                self.canonical_operations.add(sig)
                if sig not in self.op_string_to_id:
                    self.op_string_to_id[sig] = max(CUSTOM_OP_ID_START, max([0] + list(self.op_string_to_id.values())) + 1)
                A = self.op_string_to_id[sig]

                arg_details = self._get_argument_details(node)
                parts = tuple(code for code, _, _ in arg_details)
                if parts:
                    self.unique_permutations.add(parts)
                    if parts not in self.perm_tuple_to_id:
                        self.perm_tuple_to_id[parts] = max([0] + list(self.perm_tuple_to_id.values())) + 1
                    B = self.perm_tuple_to_id[parts]

                consts = [self._get_hashable_const(val) for _, val, arg_node in arg_details if arg_node is None]
                for c in consts:
                    self.unique_constants.add(c)
                    if c not in ModelProcessor.const_to_id:
                        ModelProcessor.const_to_id[c] = max([0] + list(ModelProcessor.const_to_id.values())) + 1
                const_ids = [ModelProcessor.const_to_id.get(c, 0) for c in consts]
                if const_ids: C = [len(const_ids)] + const_ids

                curr_idx = len(trng)
                D = [bw_node_map.get(arg_node, -1) - curr_idx if arg_node else 0 for _, _, arg_node in arg_details]

            bw_node_map[node] = len(trng)
            trng.append({'A': A, 'B': B, 'C': C, 'D': D})

        print(f"[TRNG] Performing DCE to remove 'nac.pass' and proxy operations...")
        pass_mapping = {}
        for idx, instr in enumerate(trng):
            if instr['A'] == nac_pass_id or instr['A'] == 0:
                if instr['D'] and len(instr['D']) > 0:
                    pass_mapping[idx] = idx + instr['D'][0]
                else:
                    pass_mapping[idx] = 0

        for idx in sorted(pass_mapping.keys()):
            current = pass_mapping[idx]
            visited = set()
            while current in pass_mapping and current not in visited:
                visited.add(current)
                current = pass_mapping[current]
            pass_mapping[idx] = current

        final_trng = []
        old_to_new_idx = {}

        for old_idx, instr in enumerate(trng):
            if old_idx in pass_mapping: continue 
            
            new_idx = len(final_trng)
            old_to_new_idx[old_idx] = new_idx
            
            if instr['D']:
                new_D = []
                for d_offset in instr['D']:
                    if d_offset == 0: new_D.append(0)
                    else:
                        target_old = old_idx + d_offset
                        if target_old in pass_mapping:
                            target_old = pass_mapping[target_old]
                        target_new = old_to_new_idx.get(target_old)
                        if target_new is not None: new_D.append(target_new - new_idx)
                        else: new_D.append(0) 
                instr['D'] = new_D
            final_trng.append(instr)

        print(f"[TRNG] DCE removed {len(trng) - len(final_trng)} instructions.")
        
        lr_idx = old_to_new_idx.get(bw_node_map["LR_CONST"], 0)
        sgd_count = 0
        for param_id, grad_idx in param_gradients:
            if grad_idx == -1: continue
            
            resolved_grad = pass_mapping.get(grad_idx, grad_idx)
            new_grad = old_to_new_idx.get(resolved_grad, -1)
            if new_grad == -1: continue
            
            idx_sgd = len(final_trng)
            final_trng.append({
                'A': 3, 
                'B': 3, 
                'C': [1, param_id], 
                'D': [new_grad - idx_sgd, lr_idx - idx_sgd]
            })
            sgd_count += 1

        final_trng.append({'A': 3, 'B': 4, 'C': [], 'D': [0]})
        self.trng_nodes = final_trng
        print(f"TRNG Graph Compiled: {len(final_trng)} instructions, Targets mapped: {target_counter}, Fused SGD steps: {sgd_count}.")

    def generate_synthetic_trng(self, lr: float = 0.001):
        print("\n--- Generating Synthetic TRNG (Fallback) ---")
        trng = []
        lr_val = lr
        if lr_val not in ModelProcessor.const_to_id: ModelProcessor.const_to_id[lr_val] = max([0] + list(ModelProcessor.const_to_id.values())) + 1
        self.unique_constants.add(lr_val)
        lr_const_id = ModelProcessor.const_to_id[lr_val]
        idx_lr = len(trng)
        trng.append({'A': 2, 'B': 3, 'C': [1, lr_const_id], 'D': []})

        id_to_sig = {v: k for k, v in self.op_string_to_id.items()}
        valid_dummy_indices = [
            i for i, node in enumerate(self.precomputed_nodes)
            if id_to_sig.get(node['A'], '') not in ('<INPUT>', '<o>', '<OUTPUT>')
        ]
        if not valid_dummy_indices: valid_dummy_indices = [max(0, len(self.precomputed_nodes) - 1)]

        for i, param_id in enumerate(self.param_id_to_name.keys()):
            idx_grad = len(trng)
            saved_ops_idx = valid_dummy_indices[i % len(valid_dummy_indices)]
            trng.append({'A': 2, 'B': 4, 'C': [1, saved_ops_idx], 'D': []})
            
            idx_sgd = len(trng)
            trng.append({
                'A': 3, 'B': 3, 'C': [1, param_id], 
                'D': [idx_grad - idx_sgd, idx_lr - idx_sgd]
            })

        idx_final = len(trng)
        trng.append({'A': 3, 'B': 4, 'C': [], 'D': [idx_lr - idx_final]})
        self.trng_nodes = trng
        print(f"Synthetic TRNG: {len(trng)} instructions for {len(self.param_id_to_name)} parameters.")

    def _quantize_to_block_fp8(self, tensor: torch.Tensor, block_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = tensor.shape
        tensor_flat = tensor.flatten().to(torch.float32)
        num_elements = tensor_flat.numel()
        rem = num_elements % block_size
        if rem != 0:
            padding = torch.zeros(block_size - rem, dtype=tensor_flat.dtype, device=tensor.device)
            tensor_flat = torch.cat([tensor_flat, padding])
        blocked_tensor = tensor_flat.view(tensor_flat.numel() // block_size, block_size)
        scales = torch.max(torch.abs(blocked_tensor), dim=1, keepdim=True)[0] / 127.0
        scales[scales == 0] = 1.0 
        quantized_blocks = torch.round(blocked_tensor / scales).clamp(-128, 127).to(torch.int8)
        return quantized_blocks.flatten(), scales.squeeze()
        
    def _quantize_parameters(self, method: str):
        if method == 'none': return
        print(f"Applying '{method}' quantization...")
        quantized_map = {}
        for k, v in self.param_data_map.items():
            if not (isinstance(v, torch.Tensor) and torch.is_floating_point(v)):
                quantized_map[k] = v; continue
            if method == 'FP16': quantized_map[k] = v.to(torch.float16)
            elif method in ['INT8_TENSOR', 'INT8_CHANNEL']:
                is_channel = method == 'INT8_CHANNEL' and v.dim() > 1
                dims = tuple(range(1, v.dim())) if is_channel else None
                scales = v.abs().amax(dim=dims, keepdim=True) / 127.0
                scales[scales == 0] = 1e-9
                if is_channel:
                    meta = {'quant_type': 'INT8_CHANNEL', 'scales': scales.reshape(-1).tolist(), 'axis': 0}
                else:
                    meta = {'quant_type': 'INT8_TENSOR', 'scale': float(scales.item())}
                quantized_map[k] = ((v / scales).round().clamp(-127, 127).to(torch.int8), meta)
            elif method == 'BLOCK_FP8':
                q_flat, scales = self._quantize_to_block_fp8(v)
                meta = {'quant_type': 'BLOCK_FP8', 'block_size': 64, 'original_shape': list(v.shape), 'scales': scales.tolist()}
                quantized_map[k] = (q_flat, meta)
        self.param_data_map = quantized_map
        print("Quantization complete.")

    def process_exported_program(self, exported_program: ExportedProgram, bw_graph: Optional[fx.Graph] = None, quantization_method: str = 'none', generate_trng: bool = False, lr: float = 0.001):
        print(f"\n--- Processing model ---")
        main_graph, sig = exported_program.graph, exported_program.graph_signature
        
        all_placeholders = {n for n in main_graph.nodes if n.op == 'placeholder'}
        self.data_input_nodes = {n for n in all_placeholders if n.target in sig.user_inputs}
        self.io_counts = (len(self.data_input_nodes), self.io_counts[1])
        
        current_param_id = 1
        state_dict, mangled_map = exported_program.state_dict, {**sig.inputs_to_parameters, **sig.inputs_to_buffers}
        target_to_node = {p.target: p for p in all_placeholders}
        for mangled, clean in mangled_map.items():
            if mangled in target_to_node and clean in state_dict:
                node = target_to_node[mangled]
                self.param_to_id[mangled] = current_param_id; self.param_id_to_name[current_param_id] = clean
                self.param_data_map[current_param_id] = state_dict[clean].detach()
                self.param_nodes.add(node); current_param_id += 1
        
        lifted_placeholders = all_placeholders - self.data_input_nodes - self.param_nodes
        if lifted_placeholders:
            for node in sorted(list(lifted_placeholders), key=lambda n: n.target):
                 for clean_name, const_tensor in exported_program.constants.items():
                    if node.target.startswith(clean_name):
                         self.lifted_const_nodes[node] = const_tensor
                         self.unique_constants.add(self._get_hashable_const(const_tensor))
                         break

        self._quantize_parameters(quantization_method)
        
        for i, node in enumerate(main_graph.nodes): self.global_node_map[node] = i
        
        for node in main_graph.nodes:
            sig = self._get_canonical_signature(node)
            if sig != "<NONE>": self.canonical_operations.add(sig)
            
            if node.op in ['call_function', 'call_method']:
                arg_details = self._get_canonical_argument_details(node)
                parts = tuple(code for code, _, _ in arg_details)
                if parts: self.unique_permutations.add(parts)
                consts = [self._get_hashable_const(val) for _, val, arg_node in arg_details if arg_node is None]
                for const in consts: self.unique_constants.add(const)

        new_ops = sorted(list(self.canonical_operations - set(self.op_string_to_id.keys())))
        if new_ops:
            max_current_id = max([0] + list(self.op_string_to_id.values()))
            next_custom_id = max(CUSTOM_OP_ID_START, max_current_id + 1)
            for op in new_ops:
                self.op_string_to_id[op] = next_custom_id; next_custom_id += 1
        
        next_id = max([0] + list(ModelProcessor.const_to_id.values())) + 1
        for const in sorted([c for c in self.unique_constants if c not in ModelProcessor.const_to_id], key=lambda x: str(x)):
            ModelProcessor.const_to_id[const] = next_id; next_id += 1
        
        next_id = max([0] + list(self.perm_tuple_to_id.values())) + 1
        for perm in sorted([p for p in self.unique_permutations if p not in self.perm_tuple_to_id]):
            self.perm_tuple_to_id[perm] = next_id; next_id += 1
            
        self.precomputed_nodes = self._process_graph(main_graph)
        print(f"Model processing complete. Generated {len(self.precomputed_nodes)} operations.")
        
        if generate_trng and bw_graph is not None:
            self.process_backward_graph(bw_graph, aot_sync_map=getattr(self, 'aot_sync_map', {}), lr=lr)
        elif generate_trng:
            self.generate_synthetic_trng(lr=lr)

    def get_used_mappings(self) -> Dict:
        used_op_ids = {self.op_string_to_id[op] for op in self.canonical_operations}
        used_const_ids = {ModelProcessor.const_to_id[c] for c in self.unique_constants}
        used_perm_ids = {self.perm_tuple_to_id[p] for p in self.unique_permutations}

        id_to_op = {v: k for k, v in self.op_string_to_id.items()}
        id_to_const = {v: k for k, v in ModelProcessor.const_to_id.items()}
        id_to_perm = {v: k for k, v in self.perm_tuple_to_id.items()}

        return {
            "canonical": {str(i): id_to_op[i] for i in sorted(list(used_op_ids))},
            "constants": {str(i): list(id_to_const[i]) if isinstance(id_to_const[i], tuple) else id_to_const[i] for i in sorted(list(used_const_ids))},
            "permutations": {str(i): "".join(id_to_perm[i]) for i in sorted(list(used_perm_ids))}
        }

def _get_d_model(model: torch.nn.Module) -> int:
    if hasattr(model, 'config'):
        for key in ['hidden_size', 'd_model', 'n_embd', 'hidden_dim']:
            if hasattr(model.config, key) and isinstance(getattr(model.config, key), int):
                return getattr(model.config, key)
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Linear): return m.in_features
    return 0

def extract_aot_backward_graph(exported_program: ExportedProgram, dummy_args, dummy_targets=None, loss_type='none', original_model=None, clean_to_pid=None) -> Optional[fx.Graph]:
    print(f"\n[AOTAutograd] Tracing backward graph with {loss_type} loss...")
    if clean_to_pid is None: clean_to_pid = {}
    try:
        orig_params = dict(original_model.named_parameters()) if original_model is not None else {}
        
        class LossWrapper(torch.nn.Module):
            def __init__(self, ep: ExportedProgram, orig_model: torch.nn.Module, c2p: dict):
                super().__init__()
                self.gm = ep.graph_module
                self.ep = ep
                self.param_names = {}
                self.req_grad_param_ids = []
                
                for spec in ep.graph_signature.input_specs:
                    if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
                        if spec.target in ep.state_dict:
                            clean_name = spec.arg.name.replace(".", "_")
                            tensor_val = ep.state_dict[spec.target].clone()
                            
                            req_grad = False
                            if orig_model is not None and spec.kind == InputKind.PARAMETER:
                                if spec.target in orig_params:
                                    req_grad = orig_params[spec.target].requires_grad
                                else:
                                    matched = False
                                    for k, v in orig_params.items():
                                        if k.replace('.', '_') == spec.target.replace('.', '_') or k in spec.target:
                                            req_grad = v.requires_grad
                                            matched = True
                                            break
                                    if not matched:
                                        req_grad = tensor_val.is_floating_point() or tensor_val.is_complex()
                            elif spec.kind == InputKind.PARAMETER:
                                req_grad = tensor_val.is_floating_point() or tensor_val.is_complex()
                            
                            if req_grad:
                                self.register_parameter(clean_name, torch.nn.Parameter(tensor_val, requires_grad=True))
                                self.req_grad_param_ids.append(c2p.get(spec.target, 0))
                            else:
                                self.register_buffer(clean_name, tensor_val)
                                
                            self.param_names[spec.arg.name] = clean_name

            def forward(self, *args):
                flat_user_args, _ = pytree.tree_flatten(dummy_args)
                num_fw_args = len(flat_user_args)
                fw_args = args[:num_fw_args]
                targets = args[num_fw_args:]

                placeholders = [n for n in self.gm.graph.nodes if n.op == 'placeholder']
                args_for_gm = []
                user_idx = 0
                
                device = 'cpu'
                if len(fw_args) > 0 and isinstance(fw_args[0], torch.Tensor):
                    device = fw_args[0].device

                for node, spec in zip(placeholders, self.ep.graph_signature.input_specs):
                    if spec.kind == InputKind.USER_INPUT:
                        if user_idx < len(fw_args): args_for_gm.append(fw_args[user_idx])
                        else:
                            node_type_str = str(node.type).lower()
                            if 'int' in node_type_str:
                                try: val = int(node.meta['val'])
                                except: val = 1
                                args_for_gm.append(val)
                            elif 'float' in node_type_str: args_for_gm.append(1.0)
                            elif 'bool' in node_type_str: args_for_gm.append(True)
                            else: args_for_gm.append(torch.zeros(1, device=device))
                        user_idx += 1
                    elif spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
                        param_name = self.param_names.get(spec.arg.name)
                        if param_name is not None and hasattr(self, param_name): args_for_gm.append(getattr(self, param_name))
                        else:
                            meta = node.meta.get('tensor_meta')
                            args_for_gm.append(torch.zeros(meta.shape, dtype=meta.dtype, device=device) if meta else torch.zeros(1, device=device))
                    elif spec.kind == InputKind.CONSTANT_TENSOR:
                        val = self.ep.constants.get(spec.target)
                        if val is not None: args_for_gm.append(val.to(device) if isinstance(val, torch.Tensor) else val)
                        else:
                            meta = node.meta.get('tensor_meta')
                            args_for_gm.append(torch.zeros(meta.shape, dtype=meta.dtype, device=device) if meta else torch.zeros(1, device=device))
                    elif spec.kind == InputKind.CUSTOM_OBJ: args_for_gm.append(self.ep.custom_objs[spec.target])
                    else: args_for_gm.append(torch.zeros(1, device=device))

                out = self.gm(*args_for_gm)
                if isinstance(out, (tuple, list)): out = out[0]
                if isinstance(out, dict): out = list(out.values())[0]
                
                if loss_type == 'cross_entropy' and targets:
                    if out.ndim == 3: out = out.transpose(1, 2)
                    loss = torch.nn.functional.cross_entropy(out, targets[0])
                elif loss_type == 'mse' and targets: loss = torch.nn.functional.mse_loss(out, targets[0])
                else: loss = out.sum()
                return loss.view(1)

        lw = LossWrapper(exported_program, original_model, clean_to_pid)
        fw_graph_list = []
        bw_graph_list = []
        def fw_c(gm, _): 
            fw_graph_list.append(gm.graph); return gm.forward
        def bw_c(gm, _):
            bw_graph_list.append(gm.graph); return gm.forward

        aot_mod = aot_module(lw, fw_compiler=fw_c, bw_compiler=bw_c, decompositions={})
        
        args_with_grad = []
        flat_args, _ = pytree.tree_flatten(dummy_args)
        
        for a in flat_args:
            if isinstance(a, torch.Tensor):
                a_proc = a.detach().clone().contiguous()
                if getattr(a, 'requires_grad', False):
                    a_proc.requires_grad_(True)
                args_with_grad.append(a_proc)
            else: 
                args_with_grad.append(a)
            
        if dummy_targets is not None:
            flat_targets, _ = pytree.tree_flatten(dummy_targets)
            for t in flat_targets:
                args_with_grad.append(t.detach().clone().contiguous() if isinstance(t, torch.Tensor) else t)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss = aot_mod(*args_with_grad)
            loss.backward()

        if bw_graph_list and fw_graph_list:
            bw_graph = bw_graph_list[0]
            fw_graph = fw_graph_list[0]
            
            bw_graph.eliminate_dead_code()
            fw_graph.eliminate_dead_code()

            bw_graph.req_grad_param_ids = lw.req_grad_param_ids

            from collections import defaultdict
            fw_by_tgt = defaultdict(list)
            for n in fw_graph.nodes:
                if n.op in ('call_function', 'call_method'):
                    fw_by_tgt[n.target].append(n)
                    
            orig_by_tgt = defaultdict(list)
            for n in exported_program.graph_module.graph.nodes:
                if n.op in ('call_function', 'call_method'):
                    orig_by_tgt[n.target].append(n)
                    
            fw_name_to_orig = {}
            for tgt, fw_nodes in fw_by_tgt.items():
                orig_nodes = orig_by_tgt.get(tgt, [])
                for i, fw_n in enumerate(fw_nodes):
                    if i < len(orig_nodes):
                        fw_name_to_orig[fw_n.name] = orig_nodes[i]

            fw_phs = [n for n in fw_graph.nodes if n.op == 'placeholder']
            orig_phs = [n for n in exported_program.graph_module.graph.nodes if n.op == 'placeholder']
            for i, fw_n in enumerate(fw_phs):
                if i < len(orig_phs):
                    fw_name_to_orig[fw_n.name] = orig_phs[i]
                    
            fw_node_by_name = {n.name: n for n in fw_graph.nodes}
            
            for bw_n in bw_graph.nodes:
                if bw_n.op == 'placeholder':
                    if bw_n.name.startswith("tangent") or bw_n.name.startswith("grad_out"):
                        bw_n.meta['is_tangent'] = True
                    else:
                        fw_n = fw_node_by_name.get(bw_n.name)
                        if fw_n is None:
                            base_name = bw_n.name.rsplit('_', 1)[0] if '_' in bw_n.name else bw_n.name
                            fw_n = fw_node_by_name.get(base_name)
                            
                        if fw_n is not None:
                            bw_n.meta['fw_source_name'] = fw_n.name
                            if 'tensor_meta' in fw_n.meta:
                                bw_n.meta['tensor_meta'] = fw_n.meta['tensor_meta']
                            if fw_n.name in fw_name_to_orig:
                                bw_n.meta['orig_node'] = fw_name_to_orig[fw_n.name]
            
            return bw_graph
            
    except Exception as e:
        print(f"[AOTAutograd] Extraction failed: {e}.")
        import traceback
        traceback.print_exc()
    return None

LOSS_TYPE_MAP: Dict[str, int] = {
    'cross_entropy': 0,
    'mse':           1,
    'none':          255,
}

def generate_artifacts(model_name: str, model: torch.nn.Module, dummy_args: Tuple, dummy_targets: Optional[Tuple]=None, loss_type: str='none', lr: float = 0.001, d_model: Optional[int] = None, quantization_method: str = 'none', dynamic_shapes=None, store_weights_internally=True, io_counts=(0,0), tokenizer_repo=None, optimize = True, tokenizer_input: str = 'none', optimize_memory_locality: bool = True, compile_tokenizer_resources: bool = True, mep_program=None, generate_trng: bool = False):
    print("\n" + "="*20 + f" GENERATION ({model_name}) " + "="*20)
    mep_loss_int = LOSS_TYPE_MAP.get(loss_type)
    if mep_loss_int is None:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Valid: {list(LOSS_TYPE_MAP)}")
    result_manager = ResultManager()

    tokenizer_resources = {} 
    if tokenizer_repo:
        print(f"Retrieving tokenizer resources from '{tokenizer_repo}'...")
        try:
            from transformers import AutoTokenizer
            import io
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)

            if compile_tokenizer_resources:
                vocab_dict = hf_tokenizer.get_vocab()
                vocab_sorted_by_token = sorted(vocab_dict.items(), key=lambda item: item[0])
                num_entries = len(vocab_sorted_by_token)
                data_buffer = io.BytesIO()
                token_offsets, id_to_offset_map = [], {}

                for token_str, token_id in vocab_sorted_by_token:
                    current_offset = data_buffer.tell()
                    token_offsets.append(current_offset); id_to_offset_map[token_id] = current_offset
                    token_bytes = token_str.encode('utf-8')
                    data_buffer.write(struct.pack('<H', len(token_bytes)))
                    data_buffer.write(token_bytes)
                    data_buffer.write(struct.pack('<if', token_id, 0.0))
                
                final_vocab_b = bytearray()
                final_vocab_b.extend(struct.pack('<I', num_entries))
                final_vocab_b.extend(struct.pack(f'<{num_entries}I', *token_offsets))
                final_vocab_b.extend(data_buffer.getvalue())
                tokenizer_resources['vocab.b'] = bytes(final_vocab_b)

                if id_to_offset_map:
                    max_id = max(id_to_offset_map.keys())
                    id_sorted_offsets = [id_to_offset_map.get(i, 0) for i in range(max_id + 1)]
                    vidx_b_content = bytearray()
                    vidx_b_content.extend(struct.pack('<I', len(id_sorted_offsets)))
                    vidx_b_content.extend(struct.pack(f'<{len(id_sorted_offsets)}I', *id_sorted_offsets))
                    tokenizer_resources['vidx.b'] = bytes(vidx_b_content)

                try:
                    merges_path = hf_hub_download(repo_id=tokenizer_repo, filename="merges.txt")
                    with open(merges_path, 'r', encoding='utf-8') as f: lines = f.readlines()
                    if lines and lines[0].startswith("#"): lines = lines[1:]
                    merges_b_content = bytearray()
                    merges_b_content.extend(struct.pack('<I', len(lines)))
                    for line in lines:
                        parts = line.strip().split();
                        if len(parts) != 2: continue
                        p1_bytes, p2_bytes = parts[0].encode('utf-8'), parts[1].encode('utf-8')
                        merges_b_content.extend(struct.pack('<H', len(p1_bytes))); merges_b_content.extend(p1_bytes)
                        merges_b_content.extend(struct.pack('<H', len(p2_bytes))); merges_b_content.extend(p2_bytes)
                    tokenizer_resources['merges.b'] = bytes(merges_b_content)
                except Exception: pass
            else:
                files_to_pack = ["tokenizer.json", "vocab.json", "merges.txt", "spiece.model", "special_tokens_map.json", "tokenizer_config.json"]
                for filename in files_to_pack:
                    try:
                        downloaded_path = hf_hub_download(repo_id=tokenizer_repo, filename=filename)
                        with open(downloaded_path, 'rb') as f: tokenizer_resources[filename] = f.read()
                    except Exception: pass
                if "vocab.json" not in tokenizer_resources:
                    vocab = hf_tokenizer.get_vocab()
                    sorted_vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
                    tokenizer_resources['vocab.json'] = json.dumps(sorted_vocab, ensure_ascii=False).encode('utf-8')

            if not store_weights_internally:
                tokenizer_dir = os.path.join(result_manager.output_path, f"{model_name}-tokenizer")
                os.makedirs(tokenizer_dir, exist_ok=True)
                for filename, content in tokenizer_resources.items():
                    with open(os.path.join(tokenizer_dir, filename), 'wb') as f: f.write(content)
                tokenizer_resources = {}

        except Exception as e: print(f"!!!!! WARNING: Failed to process tokenizer for '{tokenizer_repo}'. Error: {e}"); traceback.print_exc()

    print("Exporting a model to FX graphics...")
    exported_program = torch.export.export(model.eval(), args=dummy_args, dynamic_shapes=dynamic_shapes)
    
    bw_graph = None
    if generate_trng:
        clean_to_pid = {}
        current_param_id = 1
        sig = exported_program.graph_signature
        target_to_node = {n.target: n for n in exported_program.graph.nodes if n.op == 'placeholder'}
        for mangled, clean in {**sig.inputs_to_parameters, **sig.inputs_to_buffers}.items():
            if mangled in target_to_node and clean in exported_program.state_dict:
                clean_to_pid[clean] = current_param_id
                current_param_id += 1

        bw_graph = extract_aot_backward_graph(exported_program, dummy_args, dummy_targets=dummy_targets, loss_type=loss_type, original_model=model, clean_to_pid=clean_to_pid)

    generated_memory_map = None
    if optimize:
        optimizer_file = 'NAC_optimizer.py'
        if os.path.exists(optimizer_file):
            print(f"\n--- Run graph optimization (from {optimizer_file}) ---")
            try:
                import importlib.util, sys
                spec = importlib.util.spec_from_file_location("NAC_optimizer", optimizer_file)
                optimizer_module = importlib.util.module_from_spec(spec)
                sys.modules["NAC_optimizer"] = optimizer_module
                spec.loader.exec_module(optimizer_module)
                GraphConstantFolder = optimizer_module.GraphConstantFolder
                
                folder = GraphConstantFolder(exported_program, bw_graph=bw_graph, canonical_op_map=ModelProcessor.CANONICAL_OP_MAP)
                folder.fold(optimize_memory_locality=optimize_memory_locality)
                generated_memory_map = getattr(folder, 'generated_memory_map', None)
                bw_graph = folder.bw_graph
                
            except Exception as e:
                print(f"!!!!! ERROR while executing graph optimization: {e}\nContinuing without optimization...")
        else:
            print(f"--- INFO: Optimizer file '{optimizer_file}' not found. Skipping optimization. ---")

    registry_filepath = os.path.join(result_manager.output_path, 'registry.json')
    existing_registry_data = None
    if os.path.exists(registry_filepath) and os.path.getsize(registry_filepath) > 0:
        try:
            with open(registry_filepath, 'r', encoding='utf-8') as f: existing_registry_data = json.load(f)
        except Exception: pass

    tokenizer_manifest_bytes = None
    if tokenizer_repo:
        try:
            if 'hf_tokenizer' not in locals(): hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
            probe_text = tokenizer_input if tokenizer_input != 'none' else "This is a probe text."
            tokenizer_manifest_bytes = TISACompiler.compile_and_calibrate(hf_tokenizer, probe_text)
        except Exception as e: print(f"!!!!! WARNING: Failed to compile tokenizer for '{tokenizer_repo}'. Error: {e}")

    processor = ModelProcessor(existing_registry=existing_registry_data)
    if optimize and 'folder' in locals():
        processor.aot_sync_map = folder.aot_sync_map
        
    processor._original_model = model
        
    processor.process_exported_program(
        exported_program=exported_program, 
        bw_graph=bw_graph,
        quantization_method=quantization_method,
        generate_trng=generate_trng,
        lr=lr
    )
    
    used_mappings = processor.get_used_mappings()
    result_manager.save_registry(processor.op_string_to_id, ModelProcessor.const_to_id, processor.perm_tuple_to_id)
    final_d_model = d_model if isinstance(d_model, int) else _get_d_model(model)
    
    result_manager.save_model_nac(
        model_name, used_mappings, processor.precomputed_nodes, processor.param_data_map,
        processor.param_id_to_name, processor.input_node_idx_to_name, d_model=final_d_model,
        tokenizer_manifest=tokenizer_manifest_bytes, tokenizer_resources=tokenizer_resources,
        quant_method=quantization_method, store_weights_internally=store_weights_internally, 
        io_counts=processor.io_counts, memory_map=generated_memory_map, mep_program=mep_program,
        trng_nodes=processor.trng_nodes if generate_trng else None
    )

# --- END OF FILE NAC.py ---