# --- START OF FILE NAC_optimizer.py ---

# Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

import sys
import torch
import torch.fx as fx
import operator
import numpy as np
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import InputKind, ExportGraphSignature
from typing import Dict, Any, Set, Tuple, List, Optional
import re

from NAC_kernels import NacKernelBase

TORCH_DTYPE_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.bool: np.bool_,
}

class NacInterpreter(NacKernelBase):
    def __init__(self, canonical_op_map: Dict[str, Tuple[str, Any]] = None):
        super().__init__()
        self.canonical_op_map = canonical_op_map or {}
        
    def _enum_to_numpy_dtype(self, torch_dtype_enum):
        return TORCH_DTYPE_TO_NUMPY.get(torch_dtype_enum, np.float32)
        
    def run_node(self, node, args, kwargs):
        target = node.target
        op_name_raw = str(target) if target is not operator.getitem else "getitem"
        
        normalized_name = re.sub(r'[^a-zA-Z0-9_]+', '_', op_name_raw).strip('_')
        kernel_name = "op_" + normalized_name
        kernel = getattr(self, kernel_name, None)
        
        if kernel is None and op_name_raw in self.canonical_op_map:
            nac_name, permutation = self.canonical_op_map[op_name_raw]
            if permutation:
                try:
                    all_args = list(args) + list(kwargs.values())
                    args = [all_args[i] for i in permutation]
                    kwargs = {} 
                except IndexError:
                     raise RuntimeError(f"Failed to apply permutation {permutation} to '{op_name_raw}'")
            nac_kernel_name = "op_" + nac_name.replace('.', '_')
            kernel = getattr(self, nac_kernel_name, None)
            
        if kernel:
            try:
                return kernel(*args, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Error executing kernel '{kernel_name}' for operation '{target}': {e}") from e
        else:
            raise NotImplementedError(f"The core for operation '{target}' is not implemented.")

class GraphConstantFolder:
    def __init__(self, exported_program: ExportedProgram, bw_graph: Optional[fx.Graph] = None, canonical_op_map: Dict = None):
        self.exported_program = exported_program
        self.graph_module = exported_program.graph_module
        self.graph = self.graph_module.graph
        self.bw_graph = bw_graph
        self.signature = exported_program.graph_signature
        self.interpreter = NacInterpreter(canonical_op_map=canonical_op_map)
        
        self.computed_constants: Dict[str, Any] = {}
        self.folded_constant_counter = 0
        self.user_input_names: Set[str] = set()
        self.generated_memory_map = []
        self.aot_sync_map: Dict[str, Any] = {}

    def _sync_aot_graph(self, old_node: fx.Node, new_node_or_val: Any):
        old_name = old_node.name
        
        if isinstance(new_node_or_val, fx.Node):
            final_target = self.aot_sync_map.get(new_node_or_val.name, new_node_or_val)
        else:
            final_target = new_node_or_val
            
        self.aot_sync_map[old_name] = final_target
        
        for k, v in self.aot_sync_map.items():
            if isinstance(v, fx.Node) and v.name == old_name:
                self.aot_sync_map[k] = final_target

        if self.bw_graph is not None:
            for bw_node in self.bw_graph.nodes:
                if bw_node.op == 'placeholder':
                    if bw_node.meta.get('fw_source_name') == old_name or old_name in bw_node.name:
                        bw_node.meta['synced_target'] = final_target

    def _gather_args(self, node: fx.Node):
        args = [
            self.computed_constants[arg.name] if isinstance(arg, fx.Node) else arg
            for arg in node.args
        ]
        kwargs = {
            k: (self.computed_constants[v.name] if isinstance(v, fx.Node) else v)
            for k, v in node.kwargs.items()
        }
        return args, kwargs

    def _rewrite_getitem_zero(self):
        SAFE_TUPLE_OPS = (
            "aten._native_batch_norm_legit_no_training",
            "aten.native_layer_norm",
            "aten.layer_norm",
            "aten.scaled_dot_product_attention",
        )
        to_erase = []
        for B in list(self.graph.nodes):
            if B.op != "call_function": continue
            if B.target is not operator.getitem: continue
            if len(B.args) != 2: continue
            A, idx = B.args
            if not isinstance(A, fx.Node): continue
            if not isinstance(idx, int) or idx != 0: continue
            if A.op != "call_function": continue
            if len(A.users) != 1 or B not in A.users: continue
            tgt = str(A.target)
            if not any(op in tgt for op in SAFE_TUPLE_OPS): continue
            
            B.replace_all_uses_with(A)
            self._sync_aot_graph(B, A) 
            to_erase.append(B)
            
        for n in to_erase:
            self.graph.erase_node(n)
        if to_erase:
            print(f"[Folder] Rewrite: deleted {len(to_erase)} getitem(x,0)")

    def _eliminate_common_subexpressions(self):
        PURE_OPS = {
            "aten.add", "aten.sub", "aten.mul", "aten.div", "aten.matmul",
            "aten.relu", "aten.gelu", "aten.tanh", "aten.sigmoid",
            "aten.softmax", "aten.layer_norm",
        }
        seen = {}
        to_erase = []
        for node in self.graph.nodes:
            if node.op != "call_function": continue
            tgt = str(node.target)
            if not any(p in tgt for p in PURE_OPS): continue
            key = (node.target, tuple(node.args), tuple(sorted(node.kwargs.items())))
            if key in seen:
                prev = seen[key]
                node.replace_all_uses_with(prev)
                self._sync_aot_graph(node, prev) 
                to_erase.append(node)
            else:
                seen[key] = node
                
        for n in to_erase:
            if len(n.users) == 0:
                self.graph.erase_node(n)
        if to_erase:
            print(f"[Folder] CSE: {len(to_erase)} duplicates removed")

    def _simplify_elementwise(self):
        to_erase = []
        for node in list(self.graph.nodes):
            if node.op != "call_function": continue
            tgt = str(node.target)
            if "out" in node.kwargs: continue
            if len(node.args) < 2: continue
            a = node.args[0]
            b = node.args[1]
            
            if "aten.mul" in tgt and isinstance(b, fx.Node) and b.name in self.computed_constants:
                val = self.computed_constants[b.name]
                if val == 1:
                    node.replace_all_uses_with(a)
                    self._sync_aot_graph(node, a) 
                    to_erase.append(node); continue
                if val == 0:
                    node.replace_all_uses_with(b)
                    self._sync_aot_graph(node, b) 
                    to_erase.append(node); continue
                    
            if "aten.add" in tgt and isinstance(b, fx.Node) and b.name in self.computed_constants:
                if self.computed_constants[b.name] == 0:
                    node.replace_all_uses_with(a)
                    self._sync_aot_graph(node, a) 
                    to_erase.append(node); continue
                    
            if "aten.sub" in tgt and isinstance(b, fx.Node) and b.name in self.computed_constants:
                if self.computed_constants[b.name] == 0:
                    node.replace_all_uses_with(a)
                    self._sync_aot_graph(node, a) 
                    to_erase.append(node); continue
                    
            if "aten.div" in tgt and isinstance(b, fx.Node) and b.name in self.computed_constants:
                if self.computed_constants[b.name] == 1:
                    node.replace_all_uses_with(a)
                    self._sync_aot_graph(node, a) 
                    to_erase.append(node); continue
                    
        for n in to_erase:
            if len(n.users) == 0:
                self.graph.erase_node(n)
        if to_erase:
            print(f"[Folder] Elementwise simplify: {len(to_erase)} nodes")

    def _prune_identity_ops(self):
        IDENTITY_OPS = ("aten.detach.default", "aten.dropout.default", "nac.pass")
        nodes_to_erase = []
        for node in self.graph.nodes:
            if node.op != "call_function": continue
            op_name = str(node.target)
            if any(identity_op in op_name for identity_op in IDENTITY_OPS):
                if node.args and isinstance(node.args[0], fx.Node):
                    input_node = node.args[0]
                    node.replace_all_uses_with(input_node)
                    self._sync_aot_graph(node, input_node) 
                    nodes_to_erase.append(node)

        for node in reversed(nodes_to_erase):
            if not node.users:
                self.graph.erase_node(node)
        if nodes_to_erase:
            print(f"[Folder] Prune Identity Ops: {len(nodes_to_erase)} dummy nodes removed.")

    def _prune_unused_parameters(self):
        print("[Folder] Clearing unused parameters...")
        name_to_node = {n.name: n for n in self.graph.nodes}
        nodes_to_erase = []
        param_keys_to_remove = set()
        for spec in self.signature.input_specs:
            if spec.kind not in (InputKind.PARAMETER, InputKind.BUFFER): continue
            node = name_to_node.get(spec.arg.name)
            if node and len(node.users) == 0:
                nodes_to_erase.append(node)
                param_keys_to_remove.add(spec.target)
                self._sync_aot_graph(node, None) 
                
        for node in self.graph.nodes:
            if node.op == "get_attr" and len(node.users) == 0:
                nodes_to_erase.append(node)
                param_keys_to_remove.add(node.target)
                self._sync_aot_graph(node, None) 
                
        for key in param_keys_to_remove:
            if key in self.exported_program.state_dict: del self.exported_program.state_dict[key]
            if hasattr(self.graph_module, key): delattr(self.graph_module, key)
            
        for node in reversed(nodes_to_erase):
            if len(node.users) == 0: self.graph.erase_node(node)
            
        self.exported_program._graph_signature = ExportGraphSignature(
            input_specs=[s for s in self.signature.input_specs if s.target not in param_keys_to_remove],
            output_specs=self.signature.output_specs,
            **{k: v for k, v in self.signature.__dict__.items() if k not in ("input_specs", "output_specs")},
        )
        self.signature = self.exported_program.graph_signature
        used_attrs = {n.target for n in self.graph.nodes if n.op == "get_attr"}
        for name in list(self.graph_module._buffers.keys()):
            if name not in used_attrs: del self.graph_module._buffers[name]
        print("[Folder] Prune completed.")

    def _fuse_inverse_ops(self):
        INVERSE_PAIRS = {
            "aten.transpose.int": ("aten.transpose.int", False),
            "aten.mul.Tensor": ("aten.div.Tensor", True),
            "aten.div.Tensor": ("aten.mul.Tensor", False),
            "aten.add.Tensor": ("aten.sub.Tensor", True),
            "aten.sub.Tensor": ("aten.add.Tensor", False),
        }
        nodes_to_erase = []
        fused_count = 0
        for B in list(self.graph.nodes):
            if B.op != "call_function": continue
            op_B_name = str(B.target)
            if op_B_name not in INVERSE_PAIRS: continue
            if len(B.args) == 0 or not isinstance(B.args[0], fx.Node): continue
            A = B.args[0]
            if A.op != "call_function": continue
            if len(A.args) == 0 or not isinstance(A.args[0], fx.Node): continue
            
            op_A_name = str(A.target)
            inv_op_A, is_commutative = INVERSE_PAIRS.get(op_A_name, (None, False))
            if inv_op_A != op_B_name: continue
            if len(A.users) != 1: continue
            
            params_A = A.args[1:]
            params_B = B.args[1:]
            if params_A != params_B:
                try:
                    const_params_A = [self.computed_constants.get(p.name, p) if isinstance(p, fx.Node) else p for p in params_A]
                    const_params_B = [self.computed_constants.get(p.name, p) if isinstance(p, fx.Node) else p for p in params_B]
                    if const_params_A != const_params_B: continue
                except Exception: continue

            if not is_commutative:
                if B.args[0] != A or len(B.args) < 2 or B.args[1] not in params_B: continue

            input_to_A = A.args[0]
            B.replace_all_uses_with(input_to_A)
            
            self._sync_aot_graph(B, input_to_A) 
            self._sync_aot_graph(A, input_to_A) 
            
            if B not in nodes_to_erase: nodes_to_erase.append(B)
            if A not in nodes_to_erase: nodes_to_erase.append(A)
            fused_count += 1
        
        for _ in range(2): 
            for node in reversed(nodes_to_erase):
                if not node.users:
                    try:
                        self.graph.erase_node(node)
                        nodes_to_erase.remove(node)
                    except Exception: pass

        if fused_count > 0:
            print(f"[Folder] Fuse Inverse Ops: {fused_count} pair(s) of mutually inverse operations removed.")

    def _optimize_memory_locality(self):
        print("[Folder] Memory locality optimization (late weight materialization)...")
        user_to_attrs_map = {}
        node_list = list(self.graph.nodes)
        
        param_buffer_names = {
            spec.arg.name for spec in self.signature.input_specs
            if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER)
        }
        
        nodes_to_consider = [
            n for n in node_list 
            if (n.op == 'placeholder' and n.target in param_buffer_names) or n.op == 'get_attr'
        ]

        if not nodes_to_consider:
            print("[Folder] No parameter load/buffers nodes found to move.")
            return

        for attr_node in nodes_to_consider:
            if not attr_node.users: continue
            try:
                first_user = min(attr_node.users, key=node_list.index)
                if first_user not in user_to_attrs_map: user_to_attrs_map[first_user] = []
                user_to_attrs_map[first_user].append(attr_node)
            except ValueError: pass

        moved_count = 0
        for user_node in node_list:
            if user_node in user_to_attrs_map:
                attrs_to_move = user_to_attrs_map[user_node]
                attrs_to_move.sort(key=lambda n: node_list.index(n))
                for attr_node in attrs_to_move:
                    user_node.prepend(attr_node)
                    moved_count += 1
        if moved_count > 0:
            print(f"[Folder] Locality optimized: {moved_count} weight loading nodes were moved.")
        self.graph.lint()

    def _generate_memory_map(self) -> List[Dict]:
        print("[Folder] Generating a Memory Management Map (MMAP)...")
        node_list = list(self.graph.nodes)
        node_to_idx = {node: i for i, node in enumerate(node_list)}

        weight_nodes_info = {
            i: node for i, node in enumerate(node_list)
            if node.op == 'placeholder' and any(
                spec.arg.name == node.target and spec.kind in (InputKind.PARAMETER, InputKind.BUFFER)
                for spec in self.signature.input_specs
            )
        }

        usage_info = {
            i: {'node': node, 'users': [node_to_idx[u] for u in node.users if u in node_to_idx], 'is_output_tensor': False} 
            for i, node in enumerate(node_list)
        }
        
        last_use_map = {}
        for i, node in enumerate(node_list):
            for input_node in node.all_input_nodes:
                if input_node in node_to_idx:
                    last_use_map[node_to_idx[input_node]] = max(last_use_map.get(node_to_idx[input_node], -1), i)
        
        output_node = next(n for n in reversed(node_list) if n.op == 'output')
        for input_node in output_node.all_input_nodes:
            if input_node in node_to_idx: usage_info[node_to_idx[input_node]]['is_output_tensor'] = True

        memory_map_dict: Dict[int, List[Dict]] = {i: [] for i in range(len(node_list))}
        preloaded_weight_indices = set()
        
        i = 0
        while i < len(node_list):
            if i in weight_nodes_info and i not in preloaded_weight_indices:
                current_group = [i]
                j = i + 1
                while j in weight_nodes_info and j == current_group[-1] + 1:
                    current_group.append(j); j += 1
                
                first_op_in_group = i
                desired_distance = 3
                lookback_limit = 12
                best_tick = first_op_in_group - 1
                clean_ticks_found = 0
                
                for offset in range(1, lookback_limit + 1):
                    current_tick = first_op_in_group - offset
                    if current_tick < 0: break
                    if any(cmd['action'] == 'PRELOAD' for cmd in memory_map_dict.get(current_tick, [])): break
                    best_tick = current_tick
                    if current_tick not in weight_nodes_info: clean_ticks_found += 1
                    if clean_ticks_found >= desired_distance: break

                best_tick = max(0, best_tick)
                for weight_id in current_group:
                    memory_map_dict[best_tick].append({'action': 'PRELOAD', 'target_id': weight_id})
                    preloaded_weight_indices.add(weight_id)
                i += len(current_group)
            else: i += 1

        resources_in_memory = preloaded_weight_indices.copy()
        for i, info in usage_info.items():
            if i in preloaded_weight_indices or info['node'].op == 'placeholder': continue
            can_forward = False
            if len(info['users']) == 1 and not info['is_output_tensor']:
                user_idx = info['users'][0]
                if all(j in preloaded_weight_indices for j in range(i + 1, user_idx)):
                    can_forward = True
                    memory_map_dict[i].append({'action': 'FORWARD', 'target_id': user_idx})

            if not can_forward and len(info['users']) > 0:
                memory_map_dict[i].append({'action': 'SAVE_RESULT', 'target_id': i})
                resources_in_memory.add(i)

        for resource_id in resources_in_memory:
            if resource_id in last_use_map and not usage_info.get(resource_id, {}).get('is_output_tensor', False):
                free_tick = last_use_map[resource_id] + 1
                memory_map_dict[free_tick].append({'action': 'FREE', 'target_id': resource_id})

        memory_map = []
        action_priority = {'FORWARD': 0, 'FREE': 1, 'SAVE_RESULT': 2, 'PRELOAD': 3}
        for instr_id in sorted(memory_map_dict.keys()):
            if instr_id in preloaded_weight_indices: continue
            commands = memory_map_dict.get(instr_id, [])
            if not commands: continue
            unique_commands = list({(cmd['action'], cmd['target_id']): cmd for cmd in commands}.values())
            unique_commands.sort(key=lambda cmd: (action_priority.get(cmd['action'], 99), cmd['target_id']))
            if unique_commands: memory_map.append({'instr_id': instr_id, 'commands': unique_commands})
        
        print(f"[Folder] Generated {len(memory_map)} entries for MMAP. Skipped {len(preloaded_weight_indices)} weight loading cycles.")
        return memory_map

    def fold(self, optimize_memory_locality: bool = False):
        print("[Folder] Beginning of the convolution of constants...")

        for spec in self.signature.input_specs:
            if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
                if spec.target in self.exported_program.state_dict:
                    self.computed_constants[spec.arg.name] = (self.exported_program.state_dict[spec.target].detach().cpu().numpy())
            elif spec.kind == InputKind.USER_INPUT:
                self.user_input_names.add(spec.arg.name)

        NON_FOLDABLE = ("aten.full", "aten.arange", "aten.linspace", "aten.slice", "aten.view", "aten.reshape")

        print("[Folder] Iterative analysis...")
        while True:
            changed = False
            for node in self.graph.nodes:
                if node.op != "call_function": continue
                if node.name in self.computed_constants: continue
                if any(op in str(node.target) for op in NON_FOLDABLE): continue
                inputs = []
                def collect(n):
                    if isinstance(n, fx.Node): inputs.append(n)
                    return n
                fx.map_arg((node.args, node.kwargs), collect)
                foldable = True
                for inp in inputs:
                    if inp.name in self.user_input_names or inp.name not in self.computed_constants:
                        foldable = False; break
                if not foldable: continue
                args, kwargs = self._gather_args(node)
                try:
                    value = self.interpreter.run_node(node, args, kwargs)
                    self.computed_constants[node.name] = value
                    changed = True
                    self._sync_aot_graph(node, value) 
                except Exception: pass
            if not changed: break
        print("[Folder] Analysis complete.")

        self._rewrite_getitem_zero()
        self._simplify_elementwise()
        self._eliminate_common_subexpressions()
        self._fuse_inverse_ops()
        self._prune_identity_ops()

        if optimize_memory_locality:
            self._optimize_memory_locality()

        print("[Folder] Replacing computed nodes...")
        nodes_to_replace = [n for n in self.graph.nodes if n.op == "call_function" and n.name in self.computed_constants]

        for node in nodes_to_replace:
            value = self.computed_constants[node.name]
            buf = f"_folded_constant_{self.folded_constant_counter}"
            self.folded_constant_counter += 1
            self.graph_module.register_buffer(buf, torch.from_numpy(value))
            with self.graph.inserting_after(node):
                new_node = self.graph.get_attr(buf)
            node.replace_all_uses_with(new_node)
            self._sync_aot_graph(node, new_node) 

        for node in nodes_to_replace:
            if len(node.users) == 0: self.graph.erase_node(node)

        self._prune_unused_parameters()
        self.graph.lint()
        self.graph_module.recompile()
        
        self.generated_memory_map = self._generate_memory_map()
        print("[Folder] Cleaning are complete.")

# --- END OF FILE NAC_optimizer.py ---