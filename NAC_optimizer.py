# Copyright (c) 2025 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

import sys
import torch
import torch.fx as fx
import operator
import numpy as np
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import InputKind, ExportGraphSignature
from typing import Dict, Any, Set, Tuple 
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
    """Performs individual aten operations using NumPy."""
    def _enum_to_numpy_dtype(self, torch_dtype_enum):
        return TORCH_DTYPE_TO_NUMPY.get(torch_dtype_enum, np.float32)
    def run_node(self, node, args, kwargs):
        target = node.target
        op_name_raw = str(target) if target is not operator.getitem else "getitem"
        # 1. First, look for the "raw" aten.* kernel
        normalized_name = re.sub(r'[^a-zA-Z0-9_]+', '_', op_name_raw).strip('_')
        kernel_name = "op_" + normalized_name
        kernel = getattr(self, kernel_name, None)
        # 2. If haven't found it, look for the canonical nac.* kernel through the map
        if kernel is None and op_name_raw in self.canonical_op_map:
            nac_name, permutation = self.canonical_op_map[op_name_raw]
            # Apply the permutation of arguments, if any.
            if permutation:
                try:
                    # Collect all the arguments into one list for indexing
                    all_args = list(args) + list(kwargs.values())
                    args = [all_args[i] for i in permutation]
                    kwargs = {} # After the permutation, kwargs become invalid.
                except IndexError:
                     raise RuntimeError(f"Failed to apply permutation {permutation} to '{op_name_raw}'")
            # Already looking for nac.* kernel
            nac_kernel_name = "op_" + nac_name.replace('.', '_')
            kernel = getattr(self, nac_kernel_name, None)
            if kernel:
                 print(f"[Interpreter] Info: '{op_name_raw}' -> using canonical kernel '{nac_kernel_name}'")
        # 3. Execute the found kernel or return an error
        if kernel:
            try:
                return kernel(*args, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Error executing kernel '{kernel_name}' for operation '{target}': {e}") from e
        else:
            raise NotImplementedError(f"The core for operation '{target}' is not implemented.")

class GraphConstantFolder:
    def __init__(self, exported_program: ExportedProgram, canonical_op_map: Dict = None):
        self.exported_program = exported_program
        self.graph_module = exported_program.graph_module
        self.graph = self.graph_module.graph
        self.signature = exported_program.graph_signature
        self.interpreter = NacInterpreter(canonical_op_map=canonical_op_map)
        self.computed_constants: Dict[str, Any] = {}
        self.folded_constant_counter = 0
        self.user_input_names: Set[str] = set()

    # Utils
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

    # Semantic rewrite: getitem(x, 0)
    def _rewrite_getitem_zero(self):
        """
        Removes getitem(A, 0) ONLY if:
        - A.call_function
        - B = getitem(A,0) single user A
        - there are no other users A between A and B (len(A.users)==1)
        - A from the whitelist of tuple operations
        """
        SAFE_TUPLE_OPS = (
            "aten._native_batch_norm_legit_no_training",
            "aten.native_layer_norm",
            "aten.layer_norm",
            "aten.scaled_dot_product_attention",
        )
        to_erase = []
        for B in list(self.graph.nodes):
            if B.op != "call_function":
                continue
            if B.target is not operator.getitem:
                continue
            if len(B.args) != 2:
                continue
            A, idx = B.args
            if not isinstance(A, fx.Node):
                continue
            if not isinstance(idx, int) or idx != 0:
                continue
            if A.op != "call_function":
                continue
            # KEY CONDITION
            if len(A.users) != 1 or B not in A.users:
                continue
            tgt = str(A.target)
            if not any(op in tgt for op in SAFE_TUPLE_OPS):
                continue
            # Rewrite: B → A
            B.replace_all_uses_with(A)
            to_erase.append(B)
        for n in to_erase:
            self.graph.erase_node(n)
        if to_erase:
            print(f"[Folder] Rewrite: deleted {len(to_erase)} getitem(x,0)")

    # Optimization: Common Subexpression Elimination (CSE)
    def _eliminate_common_subexpressions(self):
        """
        y1 = f(x)
        ...
        y2 = f(x)   → y2 = y1
        Conditions:
        - f ∈ PURE_OPS
        - identical args / kwargs (fx.Node identity)
        - SSA: x does not change (in FX this is guaranteed if there is no inplace)
        """
        PURE_OPS = {
            "aten.add",
            "aten.sub",
            "aten.mul",
            "aten.div",
            "aten.matmul",
            "aten.relu",
            "aten.gelu",
            "aten.tanh",
            "aten.sigmoid",
            "aten.softmax",
            "aten.layer_norm",
        }
        seen = {}
        to_erase = []
        for node in self.graph.nodes:
            if node.op != "call_function":
                continue
            tgt = str(node.target)
            if not any(p in tgt for p in PURE_OPS):
                continue
            key = (
                node.target,
                tuple(node.args),
                tuple(sorted(node.kwargs.items())),
            )
            if key in seen:
                prev = seen[key]
                node.replace_all_uses_with(prev)
                to_erase.append(node)
            else:
                seen[key] = node
        for n in to_erase:
            if len(n.users) == 0:
                self.graph.erase_node(n)
        if to_erase:
            print(f"[Folder] CSE: {len(to_erase)} duplicates removed")

    # Optimization: Elementwise algebraic simplification
    def _simplify_elementwise(self):
        """
        Safe simplification of elementary operations.
        ONLY supported:
        - aten.mul.*
        - aten.add.*
        - aten.sub.*
        - aten.div.*
        Conditions:
        - min 2 positional args
        - no out= in kwargs
        - the constant is known
        """
        to_erase = []
        for node in list(self.graph.nodes):
            if node.op != "call_function":
                continue
            tgt = str(node.target)
            # out-versions are prohibited
            if "out" in node.kwargs:
                continue
            if len(node.args) < 2:
                continue
            a = node.args[0]
            b = node.args[1]
            # ---------------- mul ----------------
            if "aten.mul" in tgt:
                if isinstance(b, fx.Node) and b.name in self.computed_constants:
                    val = self.computed_constants[b.name]
                    # x * 1 → x
                    if val == 1:
                        node.replace_all_uses_with(a)
                        to_erase.append(node)
                        continue
                    # x * 0 → 0
                    if val == 0:
                        node.replace_all_uses_with(b)
                        to_erase.append(node)
                        continue
            # ---------------- add ----------------
            if "aten.add" in tgt:
                if isinstance(b, fx.Node) and b.name in self.computed_constants:
                    val = self.computed_constants[b.name]
                    # x + 0 → x
                    if val == 0:
                        node.replace_all_uses_with(a)
                        to_erase.append(node)
                        continue
            # ---------------- sub ----------------
            if "aten.sub" in tgt:
                if isinstance(b, fx.Node) and b.name in self.computed_constants:
                    val = self.computed_constants[b.name]
                    # x - 0 → x
                    if val == 0:
                        node.replace_all_uses_with(a)
                        to_erase.append(node)
                        continue
            # ---------------- div ----------------
            if "aten.div" in tgt:
                if isinstance(b, fx.Node) and b.name in self.computed_constants:
                    val = self.computed_constants[b.name]
                    # x / 1 → x
                    if val == 1:
                        node.replace_all_uses_with(a)
                        to_erase.append(node)
                        continue
        for n in to_erase:
            if len(n.users) == 0:
                self.graph.erase_node(n)
        if to_erase:
            print(f"[Folder] Elementwise simplify: {len(to_erase)} nodes")

    def _prune_identity_ops(self):
        """
        Removes operations that do nothing (identity/pass-through).
        For example: aten.detach, aten.dropout in eval mode.
        y = dropout(x)  ->  replace all uses of y with x.
        """
        # List of operations that are considered "dummy"
        IDENTITY_OPS = (
            "aten.detach.default",
            "aten.dropout.default",
            "nac.pass", # We also include the canonical nickname
        )
        
        nodes_to_erase = []
        for node in self.graph.nodes:
            if node.op != "call_function":
                continue
            
            # Checking if the operation is a dummy
            op_name = str(node.target)
            if any(identity_op in op_name for identity_op in IDENTITY_OPS):
                # A dummy node must have at least one argument,
                # and this argument must be another node.
                if node.args and isinstance(node.args[0], fx.Node):
                    input_node = node.args[0]
                    # Replace all uses of this node (y)
                    # to its input node (x).
                    node.replace_all_uses_with(input_node)
                    nodes_to_erase.append(node)

        # Remove dummy nodes (in reverse order to avoid breaking connections)
        for node in reversed(nodes_to_erase):
            if not node.users: # Make sure that no one else is using the node.
                self.graph.erase_node(node)
        
        if nodes_to_erase:
            print(f"[Folder] Prune Identity Ops: {len(nodes_to_erase)} dummy nodes removed (dropout, detach, etc.).")

    # Prune unused params / buffers
    def _prune_unused_parameters(self):
        print("[Folder] Clearing unused parameters...")
        name_to_node = {n.name: n for n in self.graph.nodes}
        nodes_to_erase = []
        param_keys_to_remove = set()
        for spec in self.signature.input_specs:
            if spec.kind not in (InputKind.PARAMETER, InputKind.BUFFER):
                continue
            node = name_to_node.get(spec.arg.name)
            if node and len(node.users) == 0:
                nodes_to_erase.append(node)
                param_keys_to_remove.add(spec.target)
        for node in self.graph.nodes:
            if node.op == "get_attr" and len(node.users) == 0:
                nodes_to_erase.append(node)
                param_keys_to_remove.add(node.target)
        for key in param_keys_to_remove:
            if key in self.exported_program.state_dict:
                del self.exported_program.state_dict[key]
            if hasattr(self.graph_module, key):
                delattr(self.graph_module, key)
        for node in reversed(nodes_to_erase):
            if len(node.users) == 0:
                self.graph.erase_node(node)
        self.exported_program._graph_signature = ExportGraphSignature(
            input_specs=[
                s for s in self.signature.input_specs
                if s.target not in param_keys_to_remove
            ],
            output_specs=self.signature.output_specs,
            **{
                k: v for k, v in self.signature.__dict__.items()
                if k not in ("input_specs", "output_specs")
            },
        )
        self.signature = self.exported_program.graph_signature
        # cleaning folded buffers
        used_attrs = {n.target for n in self.graph.nodes if n.op == "get_attr"}
        for name in list(self.graph_module._buffers.keys()):
            if name not in used_attrs:
                del self.graph_module._buffers[name]
        print("[Folder] Cleaning completed.")

    def _fuse_inverse_ops(self):
        """
        Finds and removes pairs of mutually inverse operations.
        Supports:
        - Involutions: transpose(transpose(x, 1, 2), 1, 2) -> x
        - Pairs: div(mul(x, C), C) -> x, sub(add(x, C), C) -> x
        """
        
        # Dictionary of pairs: operation -> (inverse_operation, commutativity_flag)
        # The flag is important for operations like add/mul, where the order of the operands does not matter.
        INVERSE_PAIRS = {
            "aten.transpose.int": ("aten.transpose.int", False),
            "aten.mul.Tensor": ("aten.div.Tensor", True),
            "aten.div.Tensor": ("aten.mul.Tensor", False), # div(y, C) is the inverse of mul(x, C)
            "aten.add.Tensor": ("aten.sub.Tensor", True),
            "aten.sub.Tensor": ("aten.add.Tensor", False), # sub(y, C) is the inverse of add(x, C)
        }

        nodes_to_erase = []
        fused_count = 0

        # Iterate over a copy of the node list since we will be modifying the graph.
        for B in list(self.graph.nodes):
            if B.op != "call_function":
                continue

            op_B_name = str(B.target)
            if op_B_name not in INVERSE_PAIRS:
                continue
            
            # Node B must have at least one argument, and it must be an fx.Node node.
            if len(B.args) == 0 or not isinstance(B.args[0], fx.Node):
                continue
            
            A = B.args[0]  # A - this is the candidate node for the first operation in the pair.
            if A.op != "call_function":
                continue
            
            # The input tensor for A must also be a node
            if len(A.args) == 0 or not isinstance(A.args[0], fx.Node):
                continue

            # --- Key conditions for collapse ---
            op_A_name = str(A.target)
            inv_op_A, is_commutative = INVERSE_PAIRS.get(op_A_name, (None, False))
            
            # 1. Operation B must be the inverse of A
            if inv_op_A != op_B_name:
                continue

            # 2. Node B must be the ONLY user of node A.
            # This ensures that we don't break the graph for other branches.
            if len(A.users) != 1:
                continue
            
            # --- Logic for checking arguments-operands ---
            
            # Сollect all arguments except the first one (which is the input tensor)
            params_A = A.args[1:]
            params_B = B.args[1:]

            # 3. The operand arguments must be "mirror" or the same
            if params_A != params_B:
                # Trying to extract the values ​​of constants if they are "wrapped" in nodes
                try:
                    const_params_A = [self.computed_constants.get(p.name, p) if isinstance(p, fx.Node) else p for p in params_A]
                    const_params_B = [self.computed_constants.get(p.name, p) if isinstance(p, fx.Node) else p for p in params_B]
                    if const_params_A != const_params_B:
                        continue
                except Exception:
                    # If it is not possible to extract the constants, we assume that they do not match
                    continue

            # 4. Additional check for non-commutative operations (sub, div)
            if not is_commutative:
                # We make sure that the operand is in the "correct" place (the second argument)
                # and that the first argument is the result of the previous operation.
                if B.args[0] != A or len(B.args) < 2 or B.args[1] not in params_B:
                     continue

            # --- Collapse ---
            # If all conditions are met, we can safely "collapse" the pair.
            # The input to A (A.args[0]) now becomes the output for anyone who used B.
            input_to_A = A.args[0]
            B.replace_all_uses_with(input_to_A)
            
            # Add both nodes (A and B) to the list for deletion.
            # They will be deleted later when they have no more users.
            if B not in nodes_to_erase:
                nodes_to_erase.append(B)
            if A not in nodes_to_erase:
                nodes_to_erase.append(A)
            
            fused_count += 1
        
        # Delete nodes that no longer have users.
        # Go through it several times in case deleting one node "frees" another.
        for _ in range(2): 
            for node in reversed(nodes_to_erase):
                if not node.users:
                    try:
                        self.graph.erase_node(node)
                        nodes_to_erase.remove(node)
                    except Exception as e:
                        print(f"Info: Could not erase node {node.name} during inverse op fusion: {e}")

        if fused_count > 0:
            print(f"[Folder] Fuse Inverse Ops: {fused_count} pair(s) of mutually inverse operations removed.")

    # Main fold
    def fold(self):
        print("[Folder] Beginning of the convolution of constants...")

        # --- STEP 1: init ---
        for spec in self.signature.input_specs:
            if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
                if spec.target in self.exported_program.state_dict:
                    self.computed_constants[spec.arg.name] = (
                        self.exported_program.state_dict[spec.target]
                        .detach()
                        .cpu()
                        .numpy()
                    )
            elif spec.kind == InputKind.USER_INPUT:
                self.user_input_names.add(spec.arg.name)

        NON_FOLDABLE = (
            "aten.full",
            "aten.arange",
            "aten.linspace",
            "aten.slice",
            "aten.view",
            "aten.reshape",
        )

        # --- STEP 2: constant folding ---
        print("[Folder] Iterative analysis...")
        while True:
            changed = False
            for node in self.graph.nodes:
                if node.op != "call_function":
                    continue
                if node.name in self.computed_constants:
                    continue
                if any(op in str(node.target) for op in NON_FOLDABLE):
                    continue
                inputs = []
                def collect(n):
                    if isinstance(n, fx.Node):
                        inputs.append(n)
                    return n
                fx.map_arg((node.args, node.kwargs), collect)
                foldable = True
                for inp in inputs:
                    if (
                        inp.name in self.user_input_names
                        or inp.name not in self.computed_constants
                    ):
                        foldable = False
                        break
                if not foldable:
                    continue
                args, kwargs = self._gather_args(node)
                try:
                    value = self.interpreter.run_node(node, args, kwargs)
                    self.computed_constants[node.name] = value
                    changed = True
                except Exception:
                    pass
            if not changed:
                break
        print("[Folder] Analysis complete.")

        # --- STEP 3: semantic rewrite ---
        self._rewrite_getitem_zero()
        # --- STEP 4: Elementwise algebraic simplification ---
        self._simplify_elementwise()
        # --- STEP 5: Common Subexpression Elimination (CSE) ---
        self._eliminate_common_subexpressions()
        # --- STEP 6: Remove pairs of mutually inverse operations
        self._fuse_inverse_ops()
        # --- STEP 7: Removing operations that do nothing
        self._prune_identity_ops()
        # --- STEP 8: replace folded ---
        print("[Folder] Replacing computed nodes...")

        nodes_to_replace = [
            n for n in self.graph.nodes
            if n.op == "call_function" and n.name in self.computed_constants
        ]

        for node in nodes_to_replace:
            value = self.computed_constants[node.name]
            buf = f"_folded_constant_{self.folded_constant_counter}"
            self.folded_constant_counter += 1

            self.graph_module.register_buffer(buf, torch.from_numpy(value))
            with self.graph.inserting_after(node):
                new_node = self.graph.get_attr(buf)

            node.replace_all_uses_with(new_node)

        for node in nodes_to_replace:
            if len(node.users) == 0:
                self.graph.erase_node(node)

        # --- STEP 9: prune ---
        self._prune_unused_parameters()
        self.graph.lint()
        self.graph_module.recompile()

        print("[Folder] Cleaning are complete.")