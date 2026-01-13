# Copyright (c) 2025 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0
## torch==2.5.1
## transformers==4.57.3

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
import torchvision.models as models
from typing import List, Optional, Set, Tuple, Dict, Any, Union
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
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs); self.indentation_level = 0
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
        """Save FULL GLOBAL maps in registry.json."""
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

    def save_model_nac(self, model_name, used_mappings, nodes, param_data, param_id_to_name, input_node_idx_to_name, d_model=0, tokenizer_manifest=None, tokenizer_resources=None, quant_method='none', store_weights_internally=True, io_counts=(0,0), memory_map=None):
        filepath = os.path.join(self.output_path, f"{model_name}.nac")
        try:
            with open(filepath, 'wb') as f:
                # --- HEADER ---
                f.write(b'NAC\x01')
                quant_map = {'none':0, 'FP16':1, 'INT8_TENSOR':2, 'INT8_CHANNEL':3}
                quant_id = quant_map.get(quant_method, 0)
                if store_weights_internally: quant_id |= 0x80
                f.write(struct.pack('<B', quant_id))
                f.write(struct.pack('<HHB', io_counts[0], io_counts[1], 0))
                
                offsets_header_format = '<H9Q4x' 
                f.write(b'\0' * struct.calcsize(offsets_header_format)) 
                
                offsets = {}

                # 1. MMAP Section (Memory Map)
                offsets['mmap'] = f.tell()
                f.write(b'MMAP')
                memory_map = memory_map or []
                f.write(struct.pack('<I', len(memory_map)))
                action_codes = {'SAVE_RESULT': 10, 'FREE': 20, 'FORWARD': 30, 'PRELOAD': 40}
                for record in memory_map:
                    instr_id = record['instr_id']
                    commands = record['commands']
                    f.write(struct.pack('<HB', instr_id, len(commands)))
                    for cmd in commands:
                        action_type_code = action_codes[cmd['action']]
                        target_id = cmd['target_id']
                        f.write(struct.pack('<BH', action_type_code, target_id))

                # 2. OPS Section
                offsets['ops'] = f.tell()
                f.write(b'OPS ')
                f.write(struct.pack('<I', len(nodes)))
                for node in nodes:
                    A, B, C, D = node['A'], node['B'], node['C'], node['D']
                    f.write(struct.pack('<BB', A, B))
                    if C: f.write(struct.pack(f'<{len(C)}h', *C))
                    if D: f.write(struct.pack(f'<{len(D)}h', *D))
                
                # 3. CMAP Section
                offsets['cmap'] = f.tell()
                f.write(b'CMAP')
                custom_cmap = {int(k): v for k, v in used_mappings['canonical'].items() if int(k) >= CUSTOM_OP_ID_START}
                f.write(struct.pack('<I', len(custom_cmap)))
                for op_id, op_name in custom_cmap.items():
                    name_bytes = op_name.encode('utf-8')
                    if len(name_bytes) > 255: raise ValueError(f"Op name too long: {op_name}")
                    f.write(struct.pack('<HB', op_id, len(name_bytes))); f.write(name_bytes)
                
                # 4. CNST Section (Constants)
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
                        else: raise TypeError(f"Unsupported list content for binary serialization: {const_val}")
                    else: raise TypeError(f"Unsupported constant type for binary serialization: {type(const_val)}")
                    f.write(struct.pack('<HBH', const_id, type_code, length))
                    if value_bytes: f.write(value_bytes)
                
                # 5. PERM Section
                offsets['perm'] = f.tell()
                f.write(b'PERM')
                perm_map = {int(k): v for k,v in used_mappings['permutations'].items()}
                f.write(struct.pack('<I', len(perm_map)))
                for p_id, p_val_str in perm_map.items():
                    p_val_bytes = p_val_str.encode('utf-8')
                    if len(p_val_bytes) > 255: raise ValueError(f"Permutation too long: {p_val_str}")
                    f.write(struct.pack('<HB', p_id, len(p_val_bytes))); f.write(p_val_bytes)
                
                # 6. DATA Section
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
                        
                        # --- START: New Pure Binary Metadata Serialization ---
                        # This block replaces the previous JSON-based metadata serialization.
                        # The format is designed for easy parsing on low-memory devices.
                        #
                        # C-like struct for the metadata block:
                        # struct TensorMeta {
                        #     uint8_t  dtype;             // Dtype enum (0-9 from _map_dtype_to_enum)
                        #     uint8_t  rank;              // Number of dimensions (e.g., 2 for a matrix)
                        #     uint32_t dims[rank];        // Array of dimensions (shape)
                        #     uint8_t  quant_type;        // 0:none, 1:INT8_TENSOR, 2:INT8_CHANNEL
                        #     // Depending on quant_type, the following fields are present:
                        #     // if quant_type == 1:
                        #     //     float scale;
                        #     // if quant_type == 2:
                        #     //     uint8_t  axis;
                        #     //     uint32_t num_scales;
                        #     //     float    scales[num_scales];
                        # };
                        
                        meta_binary = b''
                        # 1. Dtype and Rank
                        dtype_enum = self._map_dtype_to_enum(tensor.dtype)
                        shape = list(tensor.shape)
                        rank = len(shape)
                        meta_binary += struct.pack('<BB', dtype_enum, rank)
                        
                        # 2. Dimensions (Shape)
                        if rank > 0:
                            meta_binary += struct.pack(f'<{rank}I', *shape)
                            
                        # 3. Quantization Info
                        quant_type_str = meta.get('quant_type', 'none')
                        quant_type_map = {'none': 0, 'INT8_TENSOR': 1, 'INT8_CHANNEL': 2}
                        quant_type_code = quant_type_map.get(quant_type_str, 0)
                        meta_binary += struct.pack('<B', quant_type_code)

                        if quant_type_code == 1: # INT8_TENSOR
                            scale = meta['scale']
                            meta_binary += struct.pack('<f', scale)
                        elif quant_type_code == 2: # INT8_CHANNEL
                            axis = meta['axis']
                            scales = meta['scales']
                            num_scales = len(scales)
                            meta_binary += struct.pack('<BI', axis, num_scales)
                            meta_binary += struct.pack(f'<{num_scales}f', *scales)

                        data_bytes = tensor.numpy(force=True).tobytes()

                        f.write(struct.pack('<HIQ', p_id, len(meta_binary), len(data_bytes)))
                        f.write(meta_binary)
                        f.write(data_bytes)
                        # --- END: New Pure Binary Metadata Serialization ---

                else:
                    safetensors_path = os.path.join(self.output_path, f"{model_name}.safetensors")
                    tensors_to_save, metadata_to_save = {}, {}
                    for p_id, data_tuple in param_data.items():
                        name = param_id_to_name.get(p_id)
                        if not name: continue
                        tensor, meta = (data_tuple, None) if isinstance(data_tuple, torch.Tensor) else data_tuple
                        tensors_to_save[name] = tensor
                        # safetensors metadata is still JSON, but it's in a separate file, which is fine.
                        if meta: metadata_to_save[name] = json.dumps(meta) 
                    save_file(tensors_to_save, safetensors_path, metadata=metadata_to_save)
                    print(f"Weights saved to {safetensors_path}")

                # 7. PROC Section
                if tokenizer_manifest:
                    offsets['proc'] = f.tell()
                    f.write(b'PROC')
                    f.write(struct.pack('<I', len(tokenizer_manifest)))
                    f.write(tokenizer_manifest)

                # 8. RSRC Section
                if tokenizer_resources and store_weights_internally:
                    offsets['rsrc'] = f.tell()
                    f.write(b'RSRC')
                    num_files = len(tokenizer_resources)
                    f.write(struct.pack('<I', num_files))
                    for filename, content in tokenizer_resources.items():
                        name_bytes = filename.encode('utf-8')
                        f.write(struct.pack('<H', len(name_bytes)))
                        f.write(name_bytes)
                        f.write(struct.pack('<I', len(content)))
                        f.write(content)
                    print(f"Saved {num_files} resource files internally.")

                f.seek(10)
                
                all_offsets = [
                    offsets.get('mmap', 0),
                    offsets.get('ops', 0),
                    offsets.get('cmap', 0),
                    offsets.get('cnst', 0),
                    offsets.get('perm', 0),
                    offsets.get('data', 0),
                    offsets.get('proc', 0),
                    offsets.get('meta', 0),
                    offsets.get('rsrc', 0)
                ]
                f.write(struct.pack(offsets_header_format, d_model, *all_offsets))

            storage_method = "internally" if store_weights_internally else "externally"
            print(f"NAC for '{model_name}' saved successfully (Format: Full Binary, Quant: {quant_method}, Weights: {storage_method}).")
        except Exception as e:
            print(f"!!!!! ERROR saving NAC for {model_name}: {e}"); traceback.print_exc()

class ModelProcessor:
    const_to_id: Dict[Any, int] = {}

    # Format: "original.name.op"->"name": ("nac.name", (optional_argument_index_reorder))
    # The permutation (1, 0) for op(a, b) means that the canonical call is op(b, a).
    CANONICAL_OP_MAP: Dict[str, Tuple[str, Optional[Tuple[int, ...]]]] = {
        "detach": ("nac.pass", None),
        "dropout": ("nac.pass", None),
        "pass_through": ("nac.pass", None),
        "lift_fresh_copy": ("nac.clone", None),
        "true_divide": ("nac.div", None),
        "unsafe_view": ("nac.view", None),
        "squeeze": ("nac.view", None),
        "rsub": ("nac.sub", (1, 0)),  # rsub(a, b) -> b - a -> sub(b, a)
        "masked_fill": ("nac.where", (1, 2, 0)),
    }
    
    # Mappings for semantic argument codes.
    # From argument name to code
    NAME_TO_OFFSET_CODE: Dict[str, str] = {
        'query': 'Q', 'key': 'K', 'value': 'V',
        'attn_mask': 'M', 'mask': 'M',
        'bias': 'B', 'weight': 'W',
        # Let's add common names to make it easier to catch them.
        'input': 'T', 'self': 'T', 'other': 'T', 'tensor': 'T',
    }
    NAME_TO_CONST_CODE: Dict[str, str] = {
        'dim': 'A', 'axis': 'A', 'axes': 'A',
        'shape': 'S', 'size': 'S',
    }

    def __init__(self, existing_registry: Optional[Dict] = None):
        # Local sets for the CURRENT model. Start empty.
        self.canonical_operations: Set[str] = set()
        self.unique_constants: Set[Any] = set()
        self.unique_permutations: Set[Tuple[str, ...]] = set()
        
        # Other instance variables
        self.param_data_map: Dict[int, Any] = {}
        self.param_id_to_name: Dict[int, str] = {}
        self.input_node_idx_to_name: Dict[int, str] = {}
        self.precomputed_nodes: List[Dict] = []
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

        # Initializing global maps (done once)
        if not ModelProcessor.const_to_id:
            ModelProcessor.const_to_id[None] = 1

        # Loading the registry
        if existing_registry:
            self._load_from_registry(existing_registry)

    def _initialize_special_ops(self):
        special_ops = {2:"<INPUT>", 3:"<OUTPUT>", 6:"<CONTROL_FLOW>", 7:"<CONVERGENCE>"}
        for id_val, name in special_ops.items():
            # Сheck that special IDs do not conflict with standard ones.
            if id_val >= 10 and id_val in NAC_OPS:
                raise ValueError(f"Special OP ID {id_val} for '{name}' conflicts with standard NAC_OPS. Please re-assign.")
            self.op_string_to_id[name] = id_val

    def _load_from_registry(self, registry: Dict):
        """
        Loads data from the registry.
        Fixed IDs cannot be overridden. 
        Only user-defined operations are loaded.
        """
        loaded_custom_ops = 0
        if 'canonical' in registry:
            for id_str, op_name in registry['canonical'].items():
                op_id = int(id_str)
                # Load only custom operations, avoiding the standard range
                if op_id >= CUSTOM_OP_ID_START:
                    # Сheck for conflicts with those already loaded (including those from NAC_OPS)
                    if op_name in self.op_string_to_id and self.op_string_to_id[op_name] != op_id:
                         print(f"Warning: Registry tried to remap op '{op_name}' to {op_id}, but it's already {self.op_string_to_id[op_name]}. Keeping existing.")
                         continue
                    self.op_string_to_id[op_name] = op_id
                    loaded_custom_ops += 1
        if 'constants' in registry:
            for id_str, const_val in registry['constants'].items():
                hashable_const = self._get_hashable_const(const_val)
                ModelProcessor.const_to_id[hashable_const] = int(id_str)
        if 'permutations' in registry:
            for id_str, p_val_str in registry['permutations'].items():
                 self.perm_tuple_to_id[tuple(p_val_str)] = int(id_str)
        print(f"Loaded {len(self.op_string_to_id)} total ops ({loaded_custom_ops} custom), {len(ModelProcessor.const_to_id)} consts, {len(self.perm_tuple_to_id)} perms from existing registry.")

    def _get_hashable_const(self, const: Any) -> Any:
        if isinstance(const, (fx.Node, torch.SymInt)): return None
        if isinstance(const, (torch.dtype, torch.device, torch.memory_format, torch.layout)):
            # Convert to a string and remove the 'torch.' prefix.
            s = str(const)
            return s[6:] if s.startswith("torch.") else s
        if isinstance(const, torch.Tensor): return f"Tensor_{const.shape}_{const.dtype}"
        if isinstance(const, (list, tuple)): return tuple(self._get_hashable_const(item) for item in const)
        return const

    def _get_raw_signature(self, node: fx.Node) -> str:
        """Gets the raw name of the operation before canonicalization."""
        if node.op != 'call_function': return ""
        target = node.target
        return str(target).replace('::', '.') if isinstance(target, torch._ops.OpOverload) else getattr(target, '__name__', str(target))

    def _get_canonical_signature(self, node: fx.Node) -> str:
        """
        Gets the canonical name of the nac.* operation.
        1. Clears the original operation name (removes prefixes, suffixes).
        2. Looks up the cleaned name in CANONICAL_OP_MAP (for aliases and permutations).
        3. If not found, attempts to match it directly with an operation from NAC_OPS.
        4. If nothing matches, returns the FULL ORIGINAL name for registration as a custom operation in CMAP.
        """
        if node.op == 'placeholder': return "<INPUT>"
        if node.op == 'output': return "<OUTPUT>"
        if node.op != 'call_function': return "<NONE>"

        raw_sig = self._get_raw_signature(node)
        if not raw_sig: return "<NONE>"
        
        # Special case for skipping dynamic forms that are not operations
        if "aten.sym_size" in raw_sig:
            return raw_sig

        # Step 1: Clear the operation name for searching
        # Remove the prefix 'aten.'
        search_sig = raw_sig.replace("aten.", "")
        # Removing standard suffixes
        search_sig = re.sub(r'\.(default|Tensor|Scalar|int|dim|dim_IntList|using_ints|start|self)$', '', search_sig)
        # Remove underscores at the beginning/end (for _unsafe_view, add_, relu_, _softmax)
        search_sig = search_sig.strip('_')

        # Step 2: Search CANONICAL_OP_MAP (for special cases)
        if search_sig in self.CANONICAL_OP_MAP:
            return self.CANONICAL_OP_MAP[search_sig][0]
        
        # Step 3: Direct mapping to NAC_OPS
        potential_nac_name = f"nac.{search_sig}"
        if potential_nac_name in NAC_OPS_REVERSED:
            return potential_nac_name
            
        # Step 4: If the operation isn't found in the standard ones, return its
        # FULL name so it's registered as a user operation in CMAP.
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
                        if isinstance(item, fx.Node):
                            details.append((_get_code_from_node(item, name), None, item))
                        else:
                            details.append((_get_const_code(item, name), item, None))
                return details 
            except Exception as e:
                pass

        all_args = list(node.args) + list(node.kwargs.values())
        if node.op == 'output':
             all_args = all_args[0] if all_args and isinstance(all_args[0], (list, tuple)) else all_args
             self.io_counts = (self.io_counts[0], len(all_args))
        for arg in all_args:
            val = _extract_val(arg)
            items_to_process = val if isinstance(val, (list, tuple)) and any(isinstance(x, fx.Node) for x in val) else [val]
            for item in items_to_process:
                if isinstance(item, fx.Node): details.append(('T', None, item))
                else: details.append((_get_const_code(item), item, None))
        return details

    def _get_canonical_argument_details(self, node: fx.Node) -> List[Tuple[str, Any, Optional[fx.Node]]]:
        """Gets the argument details and applies a permutation to them, if defined.."""
        arg_details = self._get_argument_details(node) 

        raw_sig = self._get_raw_signature(node)
        if raw_sig:
            search_sig = raw_sig.replace("aten.", "")
            search_sig = re.sub(r'\.(default|Tensor|Scalar|int|dim|dim_IntList|using_ints|start|self)$', '', search_sig)
            search_sig = search_sig.strip('_')
            
            if search_sig in self.CANONICAL_OP_MAP:
                _, permutation = self.CANONICAL_OP_MAP[search_sig]
                if permutation:
                    try:
                        return [arg_details[i] for i in permutation]
                    except IndexError:
                        print(f"Warning: Could not apply permutation {permutation} for op '{raw_sig}'. Arg count mismatch. Using original order.")
        
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

            elif node.op == 'call_function':
                arg_details = self._get_canonical_argument_details(node)
                parts = tuple(code for code, _, _ in arg_details)
                
                if parts:
                    B = self.perm_tuple_to_id.get(parts, 0)
                    
                    # --- ИСПРАВЛЕННАЯ ЛОГИКА СОЗДАНИЯ C и D ---
                    
                    # 1. Собираем ID только для констант
                    consts = [self._get_hashable_const(val) for _, val, arg_node in arg_details if arg_node is None]
                    const_ids = [ModelProcessor.const_to_id.get(c, 0) for c in consts]
                    if const_ids:
                        C = [len(const_ids)] + const_ids
                    
                    # 2. Собираем поле D, которое соответствует пермутации.
                    #    Вставляем смещение для тензора или 0 для константы.
                    D = []
                    for _, _, arg_node in arg_details:
                        if arg_node is not None:
                            D.append(self.global_node_map.get(arg_node, -1) - i)
                        else:
                            D.append(0)
                    # --- КОНЕЦ ИСПРАВЛЕННОЙ ЛОГИКИ ---

            nodes.append({'A': A, 'B': B, 'C': C, 'D': D})
        return nodes
        
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
                scales = v.abs().amax(dim=dims, keepdim=True)/127.0; scales[scales==0]=1e-9
                meta = {'quant_type': 'INT8_CHANNEL' if is_channel else 'INT8_TENSOR', 'scales' if is_channel else 'scale': scales.squeeze().tolist() if is_channel else scales.item()}
                if is_channel: meta['axis'] = 0
                quantized_map[k] = ((v / scales).round().clamp(-127, 127).to(torch.int8), meta)
        self.param_data_map = quantized_map
        print("Quantization complete.")

    def process_exported_program(self, exported_program: ExportedProgram, quantization_method: str = 'none'):
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
            print(f"Found {len(lifted_placeholders)} lifted constant placeholder(s).")
            for node in sorted(list(lifted_placeholders), key=lambda n: n.target):
                 for clean_name, const_tensor in exported_program.constants.items():
                    if node.target.startswith(clean_name):
                         self.lifted_const_nodes[node] = const_tensor
                         self.unique_constants.add(self._get_hashable_const(const_tensor))
                         break

        print(f"Found {len(self.data_input_nodes)} data inputs, {len(self.param_nodes)} params, {len(self.lifted_const_nodes)} lifted consts.")
        self._quantize_parameters(quantization_method)
        
        for i, node in enumerate(main_graph.nodes): self.global_node_map[node] = i
        
        # Collection of all unique canonical operations, constants, and permutations
        for node in main_graph.nodes:
            sig = self._get_canonical_signature(node)
            if sig != "<NONE>":
                self.canonical_operations.add(sig)
            
            if node.op in ['call_function', 'output']:
                arg_details = self._get_canonical_argument_details(node)
                parts = tuple(code for code, _, _ in arg_details)
                if parts:
                    self.unique_permutations.add(parts)
                
                consts = [self._get_hashable_const(val) for _, val, arg_node in arg_details if arg_node is None]
                for const in consts:
                    self.unique_constants.add(const)

        new_ops = sorted(list(self.canonical_operations - set(self.op_string_to_id.keys())))
        if new_ops:
            print(f"Found {len(new_ops)} new custom operations: {new_ops}")
            # Find the maximum ID among all existing operations (standard and custom)
            max_current_id = 0
            if self.op_string_to_id:
                max_current_id = max(v for v in self.op_string_to_id.values() if v < 256) # Ignore possible future ranges

            # Start with CUSTOM_OP_ID_START or with the next one after the maximum, if it is greater
            next_custom_id = max(CUSTOM_OP_ID_START, max_current_id + 1)
            
            for op in new_ops:
                # In NAC format 'A' is 1 byte (0-255), so check the upper limit
                if next_custom_id > 255:
                    raise ValueError(f"Ran out of available operation IDs (0-255). Cannot register new op: {op}")
                self.op_string_to_id[op] = next_custom_id
                next_custom_id += 1
        
        # Assigning IDs to new constants
        next_id = max([0] + list(ModelProcessor.const_to_id.values())) + 1
        for const in sorted([c for c in self.unique_constants if c not in ModelProcessor.const_to_id], key=lambda x: str(x)):
            ModelProcessor.const_to_id[const] = next_id; next_id += 1
        
        # Assigning IDs to new permutations
        next_id = max([0] + list(self.perm_tuple_to_id.values())) + 1
        for perm in sorted([p for p in self.unique_permutations if p not in self.perm_tuple_to_id]):
            self.perm_tuple_to_id[perm] = next_id; next_id += 1
            
        self.precomputed_nodes = self._process_graph(main_graph)
        print(f"Model processing complete. Generated {len(self.precomputed_nodes)} operations.")

    def get_used_mappings(self) -> Dict:
        """
        Assembles ID->value maps for a .nac file using LOCAL data found ONLY in this model.
        """
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
    """
    Tries to automatically infer the model's hidden dimension (d_model).
    1. Checks for a `config` attribute and common key names.
    2. If that fails, inspects the last linear layer of the model.
    Returns 0 if it cannot be determined.
    """
    # Method 1: Search in the configuration (for Hugging Face models)
    if hasattr(model, 'config'):
        config = model.config
        possible_keys = ['hidden_size', 'd_model', 'n_embd', 'hidden_dim']
        for key in possible_keys:
            if hasattr(config, key):
                d_model = getattr(config, key)
                if isinstance(d_model, int):
                    print(f"Auto-detected d_model='{d_model}' from config.{key}")
                    return d_model

    # Method 2: Inspecting the last line layer (for torchvision and others)
    last_linear_layer = None
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Linear):
            last_linear_layer = m
            break
            
    if last_linear_layer is not None:
        d_model = last_linear_layer.in_features
        print(f"Auto-detected d_model='{d_model}' from last linear layer's in_features.")
        return d_model

    print("Warning: Could not automatically determine d_model. Defaulting to 0.")
    return 0

def generate_artifacts(model_name: str, model: torch.nn.Module, dummy_args: Tuple, d_model: Optional[int] = None, quantization_method: str = 'none', dynamic_shapes=None, store_weights_internally=True, io_counts=(0,0), tokenizer_repo=None, optimize = True, tokenizer_input: str = 'none', optimize_memory_locality: bool = True):
    print("\n" + "="*20 + f" GENERATION ({model_name}) " + "="*20)
    result_manager = ResultManager()

    tokenizer_resources = {} # Final resources for RSRC entry
    
    if tokenizer_repo:
        print(f"Retrieving tokenizer resources from '{tokenizer_repo}'...")
        try:
            from transformers import AutoTokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
            
            # List of standard files that we will try to pack "as is"
            files_to_pack = [
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
                "spiece.model",
                "special_tokens_map.json",
                "tokenizer_config.json"
            ]
            
            print("  - Attempting to pack original resource files...")
            for filename in files_to_pack:
                try:
                    # Downloading a file from the repository
                    downloaded_path = hf_hub_download(repo_id=tokenizer_repo, filename=filename)
                    # Read it as binary data and save it to resources.
                    with open(downloaded_path, 'rb') as f:
                        tokenizer_resources[filename] = f.read()
                    print(f"    - Packed '{filename}' successfully.")
                except Exception:
                    # It's okay if the file doesn't exist (for example, WordPiece doesn't have merges.txt)
                    pass

            # Important fallback: if vocab.json isn't found, generate it from get_vocab() so we always have a dictionary.
            if "vocab.json" not in tokenizer_resources:
                print("  - 'vocab.json' not found. Generating from tokenizer object as a fallback.")
                vocab = hf_tokenizer.get_vocab()
                # Sort the vocab by token ID for deterministic output. This is good practice.
                sorted_vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
                tokenizer_resources['vocab.json'] = json.dumps(sorted_vocab, ensure_ascii=False).encode('utf-8')


            print(f"  - Extracted {len(tokenizer_resources)} total resource files.")

            # If store resources externally, we save them to disk
            if not store_weights_internally:
                tokenizer_dir = os.path.join(result_manager.output_path, f"{model_name}-tokenizer")
                os.makedirs(tokenizer_dir, exist_ok=True)
                for filename, content in tokenizer_resources.items():
                    with open(os.path.join(tokenizer_dir, filename), 'wb') as f:
                        f.write(content)
                print(f"Tokenizer resources are stored externally in {tokenizer_dir}")
                tokenizer_resources = {} # Clean it so as not to write to RSRC

        except Exception as e:
            print(f"!!!!! WARNING: Failed to fetch tokenizer resources for '{tokenizer_repo}'. Error: {e}")
            traceback.print_exc()

    print("Exporting a model to FX graphics...")
    exported_program = torch.export.export(model.eval(), args=dummy_args, dynamic_shapes=dynamic_shapes)
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
                folder = GraphConstantFolder(exported_program, canonical_op_map=ModelProcessor.CANONICAL_OP_MAP)
                folder.fold(optimize_memory_locality=optimize_memory_locality)
                generated_memory_map = getattr(folder, 'generated_memory_map', None)
            except Exception as e:
                print(f"!!!!! ERROR while executing graph optimization: {e}\nContinuing without optimization...")
        else:
            print(f"--- INFO: Optimizer file '{optimizer_file}' not found. Skipping optimization. ---")

    registry_filepath = os.path.join(result_manager.output_path, 'registry.json')
    existing_registry_data = None
    if os.path.exists(registry_filepath) and os.path.getsize(registry_filepath) > 0:
        try:
            with open(registry_filepath, 'r', encoding='utf-8') as f: existing_registry_data = json.load(f)
            print(f"Successfully loaded existing registry from {registry_filepath}")
        except Exception as e: print(f"Failed to read existing registry: {e}")

    # --- (TISA Manifest) ---
    tokenizer_manifest_bytes = None
    if tokenizer_repo:
        print(f"\n--- Compiling the tokenizer from '{tokenizer_repo}' ---")
        try:
            if 'hf_tokenizer' not in locals():
                 hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
            if tokenizer_input == 'none':
                probe_text = "This is a probe text."
            else:
                probe_text = tokenizer_input
            tokenizer_manifest_bytes = TISACompiler.compile_and_calibrate(hf_tokenizer, probe_text)
            print(f"The tokenizer was successfully compiled into a {len(tokenizer_manifest_bytes)}-byte TISA manifest.")
        except Exception as e:
            print(f"!!!!! WARNING: Failed to compile tokenizer for '{tokenizer_repo}'. Error: {e}")
            traceback.print_exc()

    processor = ModelProcessor(existing_registry=existing_registry_data)
    
    processor.process_exported_program(
        exported_program=exported_program, 
        quantization_method=quantization_method
    )
    
    used_mappings = processor.get_used_mappings()
    
    result_manager.save_registry(
        processor.op_string_to_id,
        ModelProcessor.const_to_id,
        processor.perm_tuple_to_id
    )

    final_d_model = 0
    if d_model is not None and isinstance(d_model, int):
        final_d_model = d_model
        print(f"Uses manually specified d_model: {final_d_model}")
    else:
        final_d_model = _get_d_model(model)

    result_manager.save_model_nac(
        model_name, 
        used_mappings,
        processor.precomputed_nodes, 
        processor.param_data_map, 
        processor.param_id_to_name,
        processor.input_node_idx_to_name,
        d_model=final_d_model,
        tokenizer_manifest=tokenizer_manifest_bytes,
        tokenizer_resources=tokenizer_resources,
        quant_method=quantization_method,
        store_weights_internally=store_weights_internally, 
        io_counts=processor.io_counts,
        memory_map=generated_memory_map
    )