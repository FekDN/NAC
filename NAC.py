# Copyright (c) 2025 Feklin Dmitry (FeklinDN@gmail.com)
#
# ==============================================================================
# Neural Architecture Code (NAC) v1.2 Specification
# ==============================================================================
# The NAC format is designed for the compact, unified, and machine-readable
# representation of neural network computation graphs. Each graph is a sequence
# of nodes, where each node is described by a variable-length command.
#
# 1. Node Structure 
# Every node in the binary stream consists of a fixed-length header (5 bytes)
# and an optional variable-length data block. The structure is determined by the
# value of the first byte (`A`).
#
# There are four command types:
# 1. Fundamental Operation (standard `ABCD[]` command)
# 2. Pattern Invocation (command with `<PATTERN_PREFIX>`)
# 3. Copy Command (command with `<COPY_PREFIX>`)
# 4. Control Flow Command (command with `<CONTROL_FLOW_PREFIX>`)
#
# 2. Byte `A`: Prefix and Operation ID (1 byte, Unsigned) 
# Byte `A` defines the command type and/or serves as an index for a fundamental operation.
#
#   - `0`: `<NONE>` - Reserved. Should not appear in valid graphs.
#   - `1`: `<PAD>` - Reserved for data alignment if needed.
#   - `2`: `<DATA_INPUT>` - A node representing an entry point for user-provided data (e.g., `input_ids`).
#   - `3`: `<PARAM_INPUT>` - A node representing an entry point for model parameters (weights, biases). Mostly legacy; `<CONST_REF>` is preferred.
#   - `4`: `<OUTPUT>` - A node whose inputs are the outputs of the entire graph.
#   - `5`: `<CONST_REF>` - A node representing a reference to a parameter, identified by its name ID (e.g., `get_attr` in torch.fx).
#   - `6`: `<PATTERN_PREFIX>` - Command Prefix. Indicates this node is a high-level pattern (macro) invocation.
#   - `7`: `<COPY_PREFIX>` - Command Prefix. Indicates this node is a compression command for repeating nodes.
#   - `8`: `<CONTROL_FLOW_PREFIX>` - Command Prefix. Defines a structured control flow block (e.g., if/while).
#   - `9`: `<RESERVED>` - Reserved for future command types.
#   - `10-255`: Fundamental Operation ID - A direct index into the `registry.json["canonical_to_index"]` dictionary.
#
# 3. Command Types and Formats 
#
# 3.1. Command: Fundamental Operation (`A >= 10`) 
# The standard command for executing a single mathematical operation.
#
#   - Format: `A B C D[]`
#   - Length: 5 bytes + `2 * num_inputs`
#   - Fields:
#     - `A` (1 byte, `10-255`): Operation ID from `registry.json["canonical_to_index"]`.
#     - `B` (1 byte, `1-255`): Call variation ID from `registry.json["variation_to_index"]`. Describes the canonical permutation of tensor inputs.
#     - `C` (2 bytes, `signed`): Index for a constant or constant group.
#       - `C = 0`: No constants are used.
#       - `C > 0`: A direct ID for a literal constant from `registry.json["constants"]`.
#       - `C < 0`: An ID for a group of constants. The `group_id` is `-(C + 1)`. The group definition is in `registry.json["index_to_constant_group"]`.
#     - `D[]` (variable): List of input dependencies.
#       - `num_inputs` (1 byte): The number of inputs.
#       - `input_id` (2 bytes, `signed`) x `num_inputs`: Source node identifiers. Can be an absolute ID or a relative offset (see Section 5).
#
# 3.2. Command: Pattern Invocation (`A = 6`) 
# This command executes an entire subgraph (a pattern) defined in the registry.
#
#   - Format: `A E₀ E₁ unused D[]`
#   - Length: 5 bytes + `2 * num_inputs`
#   - Fields:
#     - `A` (1 byte): Always `6` (`<PATTERN_PREFIX>`).
#     - `E₀` (1 byte, occupies the `B` field): The lower byte of the 2-byte pattern ID.
#     - `E₁` (1 byte, 1st byte of the `C` field): The upper byte of the 2-byte pattern ID.
#       - The full `pattern_id` is reassembled as `(E₁ << 8) | E₀`, allowing for 65,536 unique pattern IDs (0 to 65535).
#     - `unused` (1 byte, 2nd byte of the `C` field): Reserved, should be `0`.
#     - `D[]` (variable): List of absolute IDs for arguments (inputs) passed to the pattern. The structure is identical to `D[]` for a fundamental operation.
#
# 3.3. Command: Copy (`A = 7`) 
# This command is used to compress sequences of identical, input-less nodes
# (Run-Length Encoding), such as repeated `get_attr` calls.
#
#   - Format: `A B C D[]`
#   - Length: 5 bytes + `2 * num_inputs` (which is `5 + 2*2 = 9` bytes total)
#   - Fields:
#     - `A` (1 byte): Always `7` (`<COPY_PREFIX>`).
#     - `B` (1 byte): The Operation ID (`A` component) of the node to be copied.
#     - `C` (2 bytes, `signed`): The number of times the node should be repeated.
#     - `D[]` (4 bytes): Repurposed to store the `B` and `C` components of the node being copied.
#       - `num_inputs` (1 byte): Always `2`.
#       - `input_1` (2 bytes, `signed`): The `B` component (Variation ID) of the template node.
#       - `input_2` (2 bytes, `signed`): The `C` component (Constant ID) of the template node.
#
# 3.4. Command: Control Flow (`A = 8`)
# Defines a structured control flow construct like `IF-ELSE` or `WHILE`. The logical
# blocks (`then`, `else`, `loop_body`) must be laid out linearly in the NAC stream
# immediately following this command node.
#
#   - Format: `A B C D[]`
#   - Length: 5 bytes + `2 * num_inputs`
#   - Fields:
#     - `A` (1 byte): Always `8` (`<CONTROL_FLOW_PREFIX>`).
#     - `B` (1 byte): Sub-opcode defining the control flow type.
#       - `1`: `IF-ELSE`.
#       - `2`: `WHILE`.
#     - `C` (2 bytes, `signed`): Unused, should be `0`.
#     - `D[]` (variable): Contains metadata describing the layout of the following blocks and their external dependencies. All IDs are absolute.
#       - For `IF-ELSE (B=1)`:
#         - `D[0]`: `len_then_block` (number of nodes in the `then` block).
#         - `D[1]`: `len_else_block` (number of nodes in the `else` block).
#         - `D[2]`: `condition_node_id` (absolute ID of the node whose boolean result is the condition).
#         - `D[3...]`: Absolute IDs of external nodes required by the `then`/`else` blocks.
#       - For `WHILE (B=2)`:
#         - `D[0]`: `len_condition_graph` (number of nodes in the condition-evaluation subgraph).
#         - `D[1]`: `len_loop_body_graph` (number of nodes in the loop body subgraph).
#         - `D[2...]`: Absolute IDs of nodes providing the initial state for the loop variables.
#
# 4. Relation to `registry.json` 
# The NAC format is not self-contained. It requires a corresponding `registry.json` file
# for interpretation. The registry acts as a dictionary, mapping the integer IDs used
# in the binary format to their full string definitions, constant values, parameter
# names, and pattern definitions.
#
# 5. Addressing Model: Absolute vs. Relative
# NAC uses a hybrid addressing model for node dependencies specified in the `D[]` component.
# The type of addressing depends on the context (scope) of the link.
#
#   - 5.1. Global Scope (Absolute Addressing):
#     - **When:** Links between nodes in the main graph, or links that cross the boundary
#       into a logical block (e.g., inputs to a pattern or a control flow structure).
#     - **Format:** A non-negative signed 2-byte integer (`0` to `32767`) representing the
#       absolute index of the source node from the beginning of the graph.
#
#   - 5.2. Block Scope (Relative Addressing):
#     - **When:** Links between nodes that are *inside* a self-contained logical block.
#       This applies to nodes within a pattern definition or within the body of a
#       control flow construct (`then`, `else`, `loop_body`, `condition_body`).
#     - **Format:** A negative signed 2-byte integer (`-1`, `-2`, etc.) representing
#       the offset from the *current* node's position.
#     - **Example:** `D=[-1]` refers to the immediately preceding node within the same block.
#     - **Benefit:** This makes logical blocks (patterns, loops) modular and relocatable. They can
#       be moved within the graph without needing to recompute all internal dependency links.
#
# ==============================================================================

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
LOCAL_FILES_ONLY=True
import torch
import torch.nn as nn
import torch.fx as fx
import torch.export
import operator
from transformers import (
    AutoModelForSequenceClassification, AutoModelForMaskedLM, 
    AutoModelForTokenClassification, AutoModelForCausalLM, AutoConfig
)
from huggingface_hub import snapshot_download
from typing import Dict, Any, List, Tuple, Set, Optional
import json
import traceback
import shutil
import base64
from collections import defaultdict 

CACHE_DIR_NAME = "scientific_models_cache"
os.environ['HF_HOME'] = os.path.abspath(CACHE_DIR_NAME)

# Assume OperationRegistry is defined elsewhere.
class OperationRegistry:
    def __init__(self):
        # Placeholder for registry attributes
        self.canonical_to_index = {}
        self.index_to_canonical = {}
        self.variation_to_index = {}
        self.index_to_constant = {}
        self.constant_to_index = {}
        self.constant_group_to_index = {}
        self.index_to_constant_group = {}
        self.param_name_to_index = {}
        self.index_to_param_name = {}
        self.patterns = {}
        self.index_to_pattern = {}
        self.pattern_to_index = {}
    def init_special_tokens(self): pass
    def _get_const_key(self, const): pass

class DatabaseManager:
    """
    Manages the persistence layer for the entire knowledge base.
    This class is responsible for saving all components of the learned knowledge
    (operations, constants, parameter names, patterns) to disk and loading them back.
    It acts as the interface between the in-memory `OperationRegistry` object and the
    `registry.json` file on disk.
    """
    def __init__(self, db_path: str = './model_knowledge_base'):
        # [CONSTRUCTOR]
        # Purpose: Initializes the manager and ensures the database directory exists.
        # Step 1: Store the provided path to the database directory.
        self.db_path = db_path
        # Step 2: Create the directory if it doesn't already exist.
        # `exist_ok=True` prevents an error if the directory is already present.
        os.makedirs(self.db_path, exist_ok=True)
        # Step 3: Print a confirmation message with the absolute path for clarity.
        print(f"Knowledge base will be saved to: {os.path.abspath(self.db_path)}")

    def save_registry(self, registry: 'OperationRegistry'):
        # [SAVE METHOD]
        # Purpose: Serializes the current state of the `OperationRegistry` object into a JSON file.
        try:
            # Step 1: Construct a dictionary (`registry_data`) that will be written to JSON.
            # This dictionary mirrors the structure of the `OperationRegistry` object.
            registry_data = {
                # Operation Mappings 
                # 'canonical_to_index': Maps a canonical operation signature (e.g., "aten.add.Tensor:node_args(2):kwargs()") to a unique integer ID.
                'canonical_to_index': registry.canonical_to_index,
                # 'index_to_canonical': The reverse mapping, from ID back to the signature string.
                'index_to_canonical': registry.index_to_canonical,

                # Variation Mapping 
                # 'variation_to_index': Maps an input permutation signature (e.g., "in0,in1") to a unique integer ID.
                'variation_to_index': registry.variation_to_index,

                # Constant Value Mappings 
                # 'constants': The primary storage for actual constant values (numbers, dtypes, etc.).
                # Each constant is serialized into a typed dictionary (e.g., {'type': 'int', 'value': 1}) using the helper method.
                'constants': {idx: self._serialize_const(const) for idx, const in registry.index_to_constant.items()},
                # 'index_to_constant' (string version): A human-readable version for debugging. It's not used during loading.
                'index_to_constant': {idx: str(const) for idx, const in registry.index_to_constant.items()},

                # Constant Group Mappings 
                # 'constant_group_to_index': Maps a sorted string of constant assignments (e.g., "arg1=10;arg2=True") to a group ID.
                'constant_group_to_index': registry.constant_group_to_index,
                # 'index_to_constant_group': The reverse mapping, from group ID back to the string.
                'index_to_constant_group': {idx: grp for idx, grp in registry.index_to_constant_group.items()},

                # Parameter Name Mappings (Crucial for Reconstruction) 
                # 'param_name_to_index': Maps a parameter's string name (e.g., "layer.0.weight") to a unique integer ID.
                'param_name_to_index': registry.param_name_to_index,
                # 'index_to_param_name': The reverse mapping, from ID back to the parameter name.
                'index_to_param_name': registry.index_to_param_name,

                # Structural Patterns 
                # 'patterns': Maps a pattern ID to its Base64-encoded binary representation.
                'patterns': registry.patterns,
            }
            # Step 2: Define the full path for the output JSON file.
            filepath = os.path.join(self.db_path, 'registry.json')
            # Step 3: Open the file in write mode with UTF-8 encoding.
            with open(filepath, 'w', encoding='utf-8') as f:
                # Step 4: Dump the prepared dictionary to the file. `indent=2` makes it human-readable.
                json.dump(registry_data, f, indent=2)
            print("Operation and constant registry saved.")
        except TypeError as e:
            # Catch potential serialization errors if an unsupported type is encountered.
            print(f"!!!!! CRITICAL SERIALIZATION ERROR: {e}")

    def load_registry(self, registry: 'OperationRegistry'):
        # [LOAD METHOD]
        # Purpose: Populates an `OperationRegistry` object with data from the `registry.json` file.
        filepath = os.path.join(self.db_path, 'registry.json')
        # Step 1: Check if the registry file exists and is not empty.
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            # If not, it's a fresh start. Initialize the registry with special tokens and return.
            print("Registry file not found or is empty. Creating a new knowledge base.")
            registry.init_special_tokens()
            return

        print(f"Attempting to load registry from {filepath}...")
        try:
            # Step 2: Open and load the JSON data from the file.
            with open(filepath, 'r', encoding='utf-8') as f: registry_data = json.load(f)
            
            # Step 3: Populate the `OperationRegistry` object field by field.
            # Use `.get(key, {})` to prevent errors if a key is missing in an older registry file.

            # Load Operation Mappings 
            registry.canonical_to_index = registry_data.get('canonical_to_index', {})
            # Keys in JSON are always strings, so we must convert them back to integers for the reverse map.
            registry.index_to_canonical = {int(k): v for k, v in registry_data.get('index_to_canonical', {}).items()}
            # If the registry was empty, initialize special tokens.
            if not registry.index_to_canonical: registry.init_special_tokens()

            # Load Variation Mapping 
            registry.variation_to_index = registry_data.get('variation_to_index', {})
            
            # Load Constant Values 
            # Clear existing dictionaries to ensure a clean load.
            registry.constant_to_index = {}
            registry.index_to_constant = {}
            # Iterate through the 'constants' dictionary, which contains typed value objects.
            for idx_str, const_info in registry_data.get('constants', {}).items():
                idx = int(idx_str)
                # Deserialize the typed dictionary back into a Python/Torch object (e.g., torch.float32).
                const_val = self._deserialize_const(const_info)
                # Generate a hashable key for the constant to use in the reverse mapping.
                key = registry._get_const_key(const_val)
                # Populate both forward and reverse mappings for constants.
                registry.index_to_constant[idx] = const_val
                registry.constant_to_index[key] = idx
            
            # Load Constant Groups 
            registry.constant_group_to_index = registry_data.get('constant_group_to_index', {})
            registry.index_to_constant_group = {int(k): v for k, v in registry_data.get('index_to_constant_group', {}).items()}
            
            # Load Parameter Names 
            registry.param_name_to_index = registry_data.get('param_name_to_index', {})
            registry.index_to_param_name = {int(k): v for k, v in registry_data.get('index_to_param_name', {}).items()}

            # Load Patterns 
            registry.patterns = {int(k): v for k, v in registry_data.get('patterns', {}).items()}
            # Rebuild the reverse mappings for patterns for efficient runtime lookups.
            # This is an in-memory convenience and is not saved to disk.
            registry.pattern_to_index = {v: k for k, v in registry.patterns.items()} # Base64 -> ID
            registry.index_to_pattern = {k: v for k, v in registry.patterns.items()} # ID -> Base64

            print(f"Registry loaded successfully. {len(registry.canonical_to_index)} ops, {len(registry.constant_to_index)} consts, {len(registry.param_name_to_index)} param names, {len(registry.patterns)} patterns.")
        except Exception as e: 
            # If any error occurs during loading (e.g., corrupted JSON), start with a fresh registry.
            print(f"!!!!! Error loading registry: {e}. Continuing with a new registry.")
            registry.init_special_tokens()

    def save_model_signatures(self, signatures: Dict[str, str], filename: str):
        # [SAVE SIGNATURES METHOD]
        # Purpose: Saves the final, compressed graph signatures to a separate JSON file.
        filepath = os.path.join(self.db_path, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(signatures, f, indent=2)
            print(f"\nSuccessfully saved {len(signatures)} signatures to file: {filename}")
        except Exception as e:
            print(f"!!!!! Error writing the final signatures file {filename}: {e}")

    def _serialize_const(self, const: Any) -> Dict:
        # [HELPER: SERIALIZE]
        # Purpose: Converts a Python or PyTorch object into a JSON-serializable dictionary.
        # This is necessary because objects like `torch.dtype` cannot be directly saved in JSON.
        # The method identifies the type of the constant and stores it along with its value.
        if isinstance(const, (int, float, str, bool, type(None))): return {'type': type(const).__name__, 'value': const}
        if isinstance(const, (list, tuple)): return {'type': type(const).__name__, 'value': list(const)}
        if isinstance(const, slice): return {'type': 'slice', 'value': [const.start, const.stop, const.step]}
        if isinstance(const, torch.dtype): return {'type': 'torch.dtype', 'value': str(const)}
        if isinstance(const, torch.device): return {'type': 'torch.device', 'value': str(const)}
        if isinstance(const, torch.layout): return {'type': 'torch.layout', 'value': str(const)}
        if isinstance(const, torch.memory_format): return {'type': 'torch.memory_format', 'value': str(const)}
        # If an unknown type is encountered, raise an error to prevent silent data corruption.
        raise TypeError(f"Object of type {type(const).__name__} is not JSON serializable")

    def _deserialize_const(self, const_info: Dict) -> Any:
        # [HELPER: DESERIALIZE]
        # Purpose: Reconstructs a Python or PyTorch object from its serialized dictionary representation.
        # It reads the 'type' field and uses it to correctly cast or recreate the 'value'.
        const_type, const_val = const_info['type'], const_info['value']
        if const_type in ['int', 'float', 'str', 'bool', 'NoneType']: return const_val
        if const_type == 'list': return const_val
        if const_type == 'tuple': return tuple(const_val)
        if const_type == 'immutable_list': return tuple(const_val) # For backward compatibility
        if const_type == 'slice': return slice(*const_val)
        # For Torch types, it reconstructs them by accessing the `torch` module.
        if const_type == 'torch.dtype': return getattr(torch, const_val.split('.')[-1])
        if const_type == 'torch.device': return torch.device(const_val)
        if const_type == 'torch.layout': return getattr(torch, const_val.split('.')[-1])
        if const_type == 'torch.memory_format': return getattr(torch, const_val.split('.')[-1])
        # If an unknown type is found in the JSON, raise an error.
        raise TypeError(f"Unknown constant type '{const_type}' during deserialization")

class GraphRepresentation:
    """
    Represents a computation graph as a binary sequence and provides a method to encode it into Base64.
    This class is the core component for converting a high-level graph (like torch.fx.Graph)
    into a compact, serializable, and comparable binary format.

    It supports two main command types:
    1. Standard Operation (ABCD format): Encodes a regular computation node.
    2. Pattern Invocation (AEE format): Encodes a call to a pre-defined subgraph (a pattern).
    """
    def __init__(self, name: str, registry: 'OperationRegistry'):
        # [CONSTRUCTOR]
        # Purpose: Initializes a new, empty graph representation.
        # Step 1: Store the human-readable name of this graph part (e.g., "bert_base_model").
        self.name = name
        # Step 2: Keep a reference to the global OperationRegistry to resolve special token IDs.
        self.registry = registry
        # Step 3: Initialize an empty, mutable byte array. This will be incrementally built
        #         as nodes are added to the graph.
        self.binary_stream = bytearray()
        # Step 4: Create a mapping from an fx.Node's unique string name to its integer ID (its position)
        #         within this binary stream. This is crucial for resolving input dependencies (the 'D' part).
        self.fx_node_to_id: Dict[str, int] = {}
        # Step 5: Initialize a counter for the number of nodes added. This also serves as the ID for the next node.
        self.node_count = 0

    def add_op_node(self, A: int, B: int, C: int, D: List[int], fx_node: fx.Node):
        # [ADD STANDARD NODE]
        # Purpose: Encodes a standard operation node and appends it to the binary stream.
        
        # Step 1: Assign the current node count as the unique ID for this new node.
        node_id = self.node_count
        # Step 2: Map the fx.Node's string name to this new integer ID. Future nodes that
        #         use `fx_node` as an input will look up this ID.
        self.fx_node_to_id[fx_node.name] = node_id
        
        # Step 3: Append the components to the binary stream, following the ABCD format.
        # A (1 byte): Operation ID. `& 0xFF` ensures it's treated as an unsigned byte.
        self.binary_stream.append(A & 0xFF)
        # B (1 byte): Variation ID.
        self.binary_stream.append(B & 0xFF)
        # C (2 bytes): Constant ID. Encoded as a 2-byte signed integer in big-endian format.
        self.binary_stream.extend(C.to_bytes(2, 'big', signed=True))
        # D (variable bytes): List of input node IDs. Handled by a helper method.
        self._encode_D(D)
        
        # Step 4: Increment the node counter for the next node.
        self.node_count += 1

    def add_pattern_node(self, pattern_id: int, D: List[int], name: str):
        # [ADD PATTERN NODE]
        # Purpose: Encodes a pattern invocation (a macro-node) and appends it to the binary stream.
        
        # Step 1: Assign a unique ID to this pattern node, similar to a standard node.
        node_id = self.node_count
        # Step 2: Map the given name for this node instance to its ID.
        self.fx_node_to_id[name] = node_id

        # Step 3: Encode the node using the AEE format.
        # A (1 byte): The special prefix indicating a pattern. Fetched from the registry.
        A = self.registry.canonical_to_index["<PATTERN_PREFIX>"]
        
        # E (2 bytes): The Pattern ID. We split the 16-bit ID into two 8-bit bytes (E0 and E1).
        # This allows us to fit the pattern ID into the existing B and C fields of the binary format.
        E0 = pattern_id & 0xFF       # Lower 8 bits of the pattern ID.
        E1 = (pattern_id >> 8) & 0xFF # Upper 8 bits of the pattern ID.

        # Step 4: Append the components to the binary stream.
        self.binary_stream.append(A & 0xFF) # The <PATTERN_PREFIX> ID.
        self.binary_stream.append(E0)       # Lower byte of pattern ID is stored in the 'B' field.
        self.binary_stream.append(E1)       # Upper byte of pattern ID is stored in the first half of the 'C' field.
        self.binary_stream.append(0)        # The second half of the 'C' field is unused, so we pad it with zero.
        
        # D (variable bytes): List of input node IDs for the entire pattern.
        self._encode_D(D)
        
        # Step 5: Increment the node counter.
        self.node_count += 1

    def _encode_D(self, D: List[int]):
        # [HELPER: ENCODE INPUTS]
        # Purpose: Encodes the list of input dependencies (D) into binary format.
        
        # Step 1: Get the number of inputs. This is crucial for the decoder to know how many IDs to read.
        num_inputs = len(D)
        # Step 2: Append the number of inputs as a single byte.
        self.binary_stream.append(num_inputs & 0xFF)
        
        # Step 3: Iterate through each input ID in the list.
        for input_id in D:
            # Step 4: Convert each ID to a 2-byte signed integer (big-endian) and append it.
            # `signed=True` is critical for supporting relative addressing (negative offsets)
            # and for the COPY command, which can store negative constant group IDs.
            self.binary_stream.extend(input_id.to_bytes(2, 'big', signed=True))

    def to_base64(self) -> str:
        # [EXPORT METHOD]
        # Purpose: Converts the entire accumulated binary stream into a URL-safe text string.
        # Step 1: Use the standard `base64.b64encode` function on the internal bytearray.
        # Step 2: Decode the resulting bytes into a standard ASCII string for JSON serialization.
        return base64.b64encode(self.binary_stream).decode('ascii')

    def get_node_count(self) -> int:
        # [GETTER METHOD]
        # Purpose: Returns the total number of nodes encoded in this representation.
        return self.node_count

class OperationRegistry:
    """
    Acts as a centralized database for mapping graph components to unique integer IDs.
    This class is the "vocabulary" of the graph language. It learns and stores every unique
    operation, input variation, constant value, and parameter name encountered across all models.
    This centralized mapping is essential for creating consistent and comparable binary signatures.
    """
    def __init__(self):
        # [CONSTRUCTOR]
        # Purpose: Initializes all the mapping dictionaries required for the vocabulary.

        # Dictionaries for Operations (Component 'A') 
        # Maps a canonical operation signature string to its integer ID.
        self.canonical_to_index: Dict[str, int] = {}
        # The reverse map: integer ID back to the signature string.
        self.index_to_canonical: Dict[int, str] = {}

        # Dictionary for Input Variations (Component 'B') 
        # Maps a string representing the permutation of inputs to an integer ID.
        self.variation_to_index: Dict[str, int] = {}

        # Dictionaries for "Literal" Constants (Component 'C') 
        # These are for actual values like numbers, booleans, dtypes, etc.
        # Maps a constant's hashable key to its integer ID.
        self.constant_to_index: Dict[Any, int] = {}
        # The reverse map: integer ID back to the actual constant value.
        self.index_to_constant: Dict[int, Any] = {}
        # Maps a string representing a group of constants to a group integer ID.
        self.constant_group_to_index: Dict[str, int] = {}
        # The reverse map: group ID back to the group string.
        self.index_to_constant_group: Dict[int, str] = {}

        # Dictionaries for Parameter Names (also Component 'C', but for <CONST_REF>) 
        # These are for symbolic references to model weights, not literal values.
        # Maps a parameter's string name (e.g., "layer.0.weight") to its integer ID.
        self.param_name_to_index: Dict[str, int] = {}
        # The reverse map: integer ID back to the parameter name.
        self.index_to_param_name: Dict[int, str] = {}
        
        # Dictionaries for Structural Patterns 
        # Maps a pattern's integer ID to its Base64-encoded binary string.
        self.patterns: Dict[int, str] = {}
        # In-memory reverse maps for efficient lookups.
        self.pattern_to_index: Dict[str, int] = {}  # Base64 -> ID
        self.index_to_pattern: Dict[int, str] = {}  # ID -> Base64

        # Configuration 
        # A set of operations where the order of tensor inputs does not change the result.
        self.COMMUTATIVE_OPS: Set[str] = {'aten.add.Tensor', 'aten.mul.Tensor'}
        # Reserve the first 10 IDs for special, globally defined tokens.
        self.RESERVED_RANGE = 10
        # Initialize the special tokens upon creation.
        self.init_special_tokens()

    def init_special_tokens(self):
        """
        Reserves the first few integer IDs for global tokens that have a special
        meaning across all graphs (e.g., inputs, outputs, control codes).
        """
        special_tokens = {
            0: "<NONE>",
            1: "<PAD>",
            2: "<DATA_INPUT>",
            3: "<PARAM_INPUT>",
            4: "<OUTPUT>",
            5: "<CONST_REF>",
            6: "<PATTERN_PREFIX>",
            7: "<COPY_PREFIX>",
            8: "<CONTROL_FLOW_PREFIX>",
            9: "<RESERVED>"
        }
        # Populate the canonical maps with these special tokens.
        for idx, token in special_tokens.items():
            if token not in self.canonical_to_index:
                self.canonical_to_index[token] = idx
                self.index_to_canonical[idx] = token
        # Define a global ID for "no constant".
        self.NONE_CONST_IDX = 0

    def _register_param_name(self, param_name: str) -> int:
        """Registers a parameter name (from a `get_attr` node) and returns its unique ID."""
        # Step 1: Check if this parameter name has already been seen.
        if param_name not in self.param_name_to_index:
            # Step 2: If it's new, calculate a new ID.
            # The ID starts after the reserved range to avoid collisions.
            new_index = max(self.RESERVED_RANGE, len(self.param_name_to_index) + self.RESERVED_RANGE)
            # Step 3: Add the new name and ID to both the forward and reverse maps.
            self.index_to_param_name[new_index] = param_name
            self.param_name_to_index[param_name] = new_index
        # Step 4: Return the ID for this parameter name.
        return self.param_name_to_index[param_name]

    def _register_constant(self, const_val: Any) -> int:
        """Registers a literal constant value (like a number or dtype) and returns its unique ID."""
        # Step 1: Get a hashable key for the constant value.
        key = self._get_const_key(const_val)
        # Step 2: Check if this constant has been seen before.
        if key not in self.constant_to_index:
            # Step 3: If new, calculate a new ID, starting after the reserved range.
            new_index = max(self.RESERVED_RANGE, len(self.constant_to_index) + self.RESERVED_RANGE)
            # Step 4: Add the new constant and its ID to both maps.
            self.index_to_constant[new_index] = const_val
            self.constant_to_index[key] = new_index
        # Step 5: Return the ID for this constant.
        return self.constant_to_index[key]

    def _get_const_key(self, const_val: Any) -> Any:
        """Helper to create a hashable key from a potentially unhashable constant value."""
        if isinstance(const_val, list): return tuple(const_val) # Convert lists to tuples.
        # If the type is not inherently hashable (but not a list), convert it to a string.
        if not isinstance(const_val, (int, float, str, bool, type(None), torch.dtype, torch.device, tuple, torch.layout, torch.memory_format, slice)):
            return str(const_val)
        return const_val

    def get_indices_ABCD(self, node: fx.Node) -> Tuple[int, int, int]:
        """
        The main method for converting an `fx.Node` into its integer-based (A, B, C) representation.
        It dispatches based on node type and computes the ID for each component.
        """
        # Handle Special Node Types 
        if node.op == 'get_attr':
            # This node is a reference to a model parameter.
            # Step 1: Register the parameter's name (node.target) to get its unique ID.
            param_id = self._register_param_name(node.target)
            # Step 2: Assemble the (A, B, C) tuple.
            A = self.canonical_to_index["<CONST_REF>"] # 'A' is the special "get parameter" operation.
            B = 1                                     # 'B' is 1, as there are no input variations.
            C = param_id                              # 'C' is the ID of the parameter name.
            return A, B, C

        # Note: 'placeholder' nodes are handled in the `Decomposer` because it has the context
        # to distinguish between data inputs and parameter inputs from `torch.export`.
        if node.op == 'output':
            A = self.canonical_to_index["<OUTPUT>"] # 'A' is the special output operation.
            B = 1
            C = self.NONE_CONST_IDX                 # Outputs have no constants.
            return A, B, C

        # Ignore nodes that are not part of the computation graph's logic.
        if node.op not in ('call_function', 'call_method', 'call_module'):
            return self.canonical_to_index["<NONE>"], 0, 0

        # Component A: Canonical Operation Signature 
        op_name = str(node.target)
        input_nodes = [arg for arg in node.all_input_nodes]
        num_node_args = len(input_nodes)
        kwarg_names = sorted(node.kwargs.keys())
        # The signature uniquely identifies an operation by its name, number of tensor args, and kwarg names.
        canonical_signature = f"{op_name}:node_args({num_node_args}):kwargs{tuple(kwarg_names)}"
        
        # Register the signature if it's new.
        if canonical_signature not in self.canonical_to_index:
            new_A_index = max(self.RESERVED_RANGE, len(self.canonical_to_index))
            self.canonical_to_index[canonical_signature] = new_A_index
            self.index_to_canonical[new_A_index] = canonical_signature
        A = self.canonical_to_index[canonical_signature]

        # Component B: Input Variation Signature 
        arg_to_input_map = {}
        for i, arg in enumerate(node.args):
            if isinstance(arg, fx.Node): arg_to_input_map[f'arg{i}'] = arg.name
        for key, val in node.kwargs.items():
            if isinstance(val, fx.Node): arg_to_input_map[key] = val.name

        # For commutative ops, sort by the actual input node names to create a canonical ordering.
        if op_name in self.COMMUTATIVE_OPS:
            variation_parts = sorted(arg_to_input_map.values())
        else: # Otherwise, sort by formal argument names to preserve order.
            variation_parts = [arg_to_input_map[key] for key in sorted(arg_to_input_map.keys())]

        # Normalize the signature by replacing concrete node names (e.g., "add_5") with generic ones ("in0").
        input_node_names = [n.name for n in input_nodes]
        name_to_canonical_input = {name: f'in{i}' for i, name in enumerate(input_node_names)}
        canonical_variation_parts = [name_to_canonical_input.get(name, str(name)) for name in variation_parts]
        variation_signature = ','.join(canonical_variation_parts)

        # Register the variation signature if it's new.
        if not variation_signature: B = 1 # Default for no variation.
        elif variation_signature not in self.variation_to_index:
            new_B_index = max(self.RESERVED_RANGE, len(self.variation_to_index) + self.RESERVED_RANGE)
            self.variation_to_index[variation_signature] = new_B_index
            B = new_B_index
        else:
            B = self.variation_to_index[variation_signature]

        # Component C: Constant Arguments 
        # Helper to check if an argument tree contains any `fx.Node`.
        def _contains_node(arg_to_check: Any) -> bool:
            found_node = False
            def check(val):
                nonlocal found_node
                if isinstance(val, fx.Node): found_node = True
            fx.map_arg(arg_to_check, check)
            return found_node

        # Collect all arguments that are NOT `fx.Node`s.
        constants_map = {}
        for i, arg in enumerate(node.args):
            if not _contains_node(arg): constants_map[f'arg{i}'] = self._register_constant(arg)
        for key, val in node.kwargs.items():
            if not _contains_node(val): constants_map[key] = self._register_constant(val)
        
        # Determine the final C value based on the number of constants found.
        if not constants_map:
            C = self.NONE_CONST_IDX
        elif len(constants_map) == 1:
            # If there's only one constant, C is its direct ID.
            C = list(constants_map.values())[0]
        else:
            # If there are multiple constants, create a group.
            group_key = ";".join([f"{k}={v}" for k, v in sorted(constants_map.items())])
            if group_key not in self.constant_group_to_index:
                new_group_index = max(self.RESERVED_RANGE, len(self.constant_group_to_index) + self.RESERVED_RANGE)
                self.constant_group_to_index[group_key] = new_group_index
                self.index_to_constant_group[new_group_index] = group_key
            group_index = self.constant_group_to_index[group_key]
            # A negative value for C indicates it's a group ID, not a direct constant ID.
            C = -(group_index + 1)
            
        return A, B, C

class Decoder:
    """
    Decodes a Base64-encoded binary graph signature back into a "raw" list of node dictionaries.
    This class is the inverse of `GraphRepresentation`. It parses the compact binary stream
    and reconstructs a structured, human-readable list of nodes that can be used for
    analysis, compression, or graph reconstruction.
    """
    def __init__(self, registry: 'OperationRegistry'):
        # [CONSTRUCTOR]
        # Purpose: Initializes the decoder with necessary special token IDs from the registry.
        self.registry = registry
        # Pre-fetch the integer IDs for special command prefixes for efficient checking in the loop.
        self.pattern_prefix_id = registry.canonical_to_index.get("<PATTERN_PREFIX>")
        self.copy_prefix_id = registry.canonical_to_index.get("<COPY_PREFIX>")

    def decode_raw(self, b64_signature: str, is_pattern_definition: bool = False) -> List[Dict[str, Any]]:
        """
        The main decoding method.
        `is_pattern_definition=True` is a flag that indicates the binary stream represents a
        pattern definition, where the input dependencies ('D') are relative offsets, not absolute IDs.
        While this implementation now reads 'D' as signed by default, the flag remains for semantic clarity
        and potential future use.
        """
        # Step 1: Decode the Base64 string into a raw byte sequence.
        binary_stream = base64.b64decode(b64_signature)
        # Step 2: Initialize an empty list to store the parsed node dictionaries.
        parsed_nodes = []
        # Step 3: Initialize a pointer to keep track of the current position in the binary stream.
        pointer = 0
        
        # Step 4: Loop through the binary stream until the pointer reaches the end.
        while pointer < len(binary_stream):
            # Step 5: Read the first byte, which is always the 'A' component (Operation ID).
            A = int.from_bytes(binary_stream[pointer : pointer+1], 'big')
            pointer += 1
            
            # This flag is used to handle the special case of the <COPY_PREFIX> command,
            # where the 'D' field is repurposed to store template information.
            read_d_later = False
            
            # Step 6: Dispatch based on the value of 'A'.
            if A == self.pattern_prefix_id:
                # This is a Pattern Invocation node (AEE format) 
                # Read the 2-byte pattern ID, which is split across the B and C fields.
                E0 = int.from_bytes(binary_stream[pointer : pointer+1], 'big'); pointer += 1
                E1 = int.from_bytes(binary_stream[pointer : pointer+1], 'big'); pointer += 1
                _ = int.from_bytes(binary_stream[pointer : pointer+1], 'big'); pointer += 1 # Skip the unused padding byte.
                # Reconstruct the 16-bit pattern ID from its two bytes.
                pattern_id = (E1 << 8) | E0
                # Create the node dictionary for the pattern.
                node = {'A': A, 'pattern_id': pattern_id}
            
            elif A == self.copy_prefix_id:
                # This is a Copy (Run-Length Encoding) node 
                # The 'B' field contains the 'A' component of the node to be copied.
                template_A = int.from_bytes(binary_stream[pointer : pointer+1], 'big'); pointer += 1
                # The 'C' field contains the number of times to repeat the node.
                count = int.from_bytes(binary_stream[pointer : pointer+2], 'big', signed=True); pointer += 2
                # Create the initial node dictionary. The rest of the template will be read from 'D'.
                node = {'A': A, 'count': count, 'template': {'A': template_A}}
                # Set the flag to trigger special 'D' field parsing.
                read_d_later = True

            else:
                # This is a Standard Operation node (ABCD format) 
                # Read the 'B' (Variation ID) and 'C' (Constant ID) components.
                B = int.from_bytes(binary_stream[pointer : pointer+1], 'big'); pointer += 1
                C = int.from_bytes(binary_stream[pointer : pointer+2], 'big', signed=True); pointer += 2
                # Create the standard node dictionary.
                node = {'A': A, 'B': B, 'C': C}

            # Step 7: Read the 'D' component (input dependencies), which is common to all node types.
            # First, read the single byte that specifies the number of inputs.
            num_inputs = int.from_bytes(binary_stream[pointer : pointer+1], 'big')
            pointer += 1
            D = []
            # Loop `num_inputs` times to read each 2-byte input ID.
            for _ in range(num_inputs):
                # Read the 2-byte value as a signed integer. This is crucial:
                # - For standard nodes, it correctly reads positive absolute IDs.
                # - For pattern definitions, it correctly reads negative/positive relative offsets.
                # - For COPY nodes, it correctly reads the signed template C value.
                input_val = int.from_bytes(binary_stream[pointer : pointer+2], 'big', signed=True)
                D.append(input_val)
                pointer += 2
            
            # Step 8: Finalize the node dictionary based on the 'read_d_later' flag.
            if read_d_later:
                # Special handling for COPY node's 'D' field 
                # The 'D' field was repurposed to store the template's B and C values.
                if len(D) >= 2:
                    node['template']['B'] = D[0] # The first value is the template B.
                    node['template']['C'] = D[1] # The second value is the template C.
                # A COPY node itself has no real inputs, so we set its 'D' field to empty.
                node['D'] = []
            else:
                # For all other node types, the read 'D' list is the actual list of inputs.
                node['D'] = D
                
            # Step 9: Add the fully constructed node dictionary to the results list.
            parsed_nodes.append(node)
            
        # Step 10: Return the complete list of parsed nodes.
        return parsed_nodes

class Decomposer:
    """
    Orchestrates the conversion of a PyTorch `nn.Module` into a `GraphRepresentation`.
    Its main responsibilities are:
    1. Extracting meaningful sub-modules (e.g., encoder, head) from a larger model.
    2. Tracing these sub-modules into a `torch.fx.GraphModule`.
    3. Iterating through the traced graph's nodes.
    4. Converting each node into its (A, B, C, D) integer representation.
    5. Assembling the final binary `GraphRepresentation`.

    NOTE: The previous Run-Length Encoding (RLE) stage has been removed, as it was
    ineffective for compressing sequences of `<CONST_REF>` nodes, which now have
    unique `C` values for each distinct parameter. The primary compression now
    happens later in the `Compressor` stage via pattern matching.
    """
    def __init__(self, registry: OperationRegistry, extractor_map: Dict):
        # [CONSTRUCTOR]
        # Purpose: Initializes the decomposer with necessary tools.
        # Step 1: Store a reference to the global OperationRegistry to resolve/register IDs.
        self.registry = registry
        # Step 2: Store a map of model types to functions that can extract specific parts.
        self.extractor_map = extractor_map

    def _get_model_parts(self, model: nn.Module, model_meta: Dict) -> Dict[str, nn.Module]:
        """Helper function to extract named sub-modules from a given model."""
        parts = {}
        # Step 1: Attempt to extract the 'encoder' part.
        encoder_key = model_meta.get("encoder_extractor")
        if encoder_key and encoder_key in self.extractor_map:
            encoder_module = self.extractor_map[encoder_key](model)
            if encoder_module: parts['base_model'] = encoder_module
        
        # Step 2: Attempt to extract the 'head' part.
        head_key = model_meta.get("head_extractor")
        if head_key and head_key in self.extractor_map:
            head_module = self.extractor_map[head_key](model)
            if head_module and head_module is not model: parts['head'] = head_module
        
        # Step 3: If no specific parts could be extracted, fall back to using the entire model.
        if not parts:
            print("  -> WARNING: Could not extract specific parts. Decomposing the entire model.")
            parts['base_model'] = model
        return parts

    def run(self, model: nn.Module, model_name: str, model_meta: Dict) -> Dict[str, 'GraphRepresentation']:
        print(f"\n--- Decomposition: {model_name} ---")
        model_parts = self._get_model_parts(model, model_meta)
        results = {}

        try:
            # For torchvision models, the .config attribute may not exist.
            config = getattr(model, 'config', None)
        except AttributeError:
            print("  -> WARNING: The model does not have a .config attribute.")
            config = None

        for part_name, part_module in model_parts.items():
            if not part_module:
                print(f"  Part '{part_name}' not found or empty. Skipping.")
                continue
        
            print(f"  Tracing part '{part_name}'...")
            gm = None
    
            try:
                # Pass model_meta to determine the input type
                dummy_args, dummy_kwargs = self._get_dummy_inputs(part_name, part_module, config, model_meta)
        
                if dummy_args is None and dummy_kwargs is None:
                    print("    -> Failed to create inputs for `torch.export`. Falling back to `symbolic_trace`.")
                    gm = fx.symbolic_trace(part_module)
                else:
                    print("    -> Strategy: `torch.export` (universal).")
                    exported_program = torch.export.export(part_module, args=dummy_args, kwargs=dummy_kwargs)
                    gm = exported_program.graph_module
            except Exception as e:
                print(f"    !!!!! Traceback error in '{part_name}': {e}")
                traceback.print_exc()
                continue

            if gm:
                # --- STAGE 1: Initial Node List Generation ---
                # Convert every fx.Node into a dictionary representation.
                raw_nodes = []
                temp_fx_map = {} # Maps fx.Node.name to its integer index in `raw_nodes`.
            
                # Identify which placeholders are for data vs. parameters.
                num_data_inputs = len(dummy_args) + len(dummy_kwargs)
                placeholders = [p for p in gm.graph.nodes if p.op == 'placeholder']
                input_names_from_export = [p.name for p in placeholders[:num_data_inputs]]

                for node_id, node in enumerate(gm.graph.nodes):
                    temp_fx_map[node.name] = node_id
                    # Dispatch based on the fx.Node's operation type.
                    if node.op == 'placeholder':
                        if node.name in input_names_from_export:
                            # This is a true data input (e.g., input_ids).
                            A = self.registry.canonical_to_index["<DATA_INPUT>"]
                            B, C = 1, self.registry.NONE_CONST_IDX
                        else:
                            # This is a parameter lifted by `torch.export`. Treat it as a constant reference.
                            A = self.registry.canonical_to_index["<CONST_REF>"]
                            B = 1
                            C = self.registry._register_param_name(node.name) # C holds the ID of the parameter name.
                    else:
                        # For all other nodes (`call_function`, `get_attr`), use the registry to get (A,B,C).
                        A, B, C = self.registry.get_indices_ABCD(node)
                
                    if A == self.registry.canonical_to_index["<NONE>"]: continue
                
                    # Resolve input dependencies: get integer IDs of input nodes.
                    inputs = [temp_fx_map[n.name] for n in node.all_input_nodes if n.name in temp_fx_map]
                    raw_nodes.append({'A': A, 'B': B, 'C': C, 'D': inputs, 'fx_node': node, 'original_id': node_id})

                # --- STAGE 2: Run-Length Encoding (RLE) Compression ---
                # Find and compress sequences of identical, input-less nodes (e.g., repeated `get_attr`).
                compressed_nodes = []
                old_to_new_id_map = {} # Maps original node ID to its new ID in `compressed_nodes`.
                i = 0
                while i < len(raw_nodes):
                    current_node = raw_nodes[i]
                    new_node_id = len(compressed_nodes)
                
                    # RLE is only applied to nodes without inputs.
                    if not current_node['D']:
                        count = 1
                        j = i + 1
                        # Check subsequent nodes for an identical (A, B, C) signature.
                        while (j < len(raw_nodes) and not raw_nodes[j]['D'] and 
                               raw_nodes[j]['A'] == current_node['A'] and
                               raw_nodes[j]['B'] == current_node['B'] and
                               raw_nodes[j]['C'] == current_node['C']):
                            count += 1
                            j += 1
                    
                        # Apply compression if the sequence is long enough.
                        if count > 2:
                            op_to_copy_str = self.registry.index_to_canonical.get(current_node['A'], str(current_node['A']))
                            print(f"    -> Compressing a sequence of {count} '{op_to_copy_str}' nodes")
                        
                            # Map all original nodes in the sequence to the single new compressed node's ID.
                            for k in range(i, i + count):
                                old_to_new_id_map[raw_nodes[k]['original_id']] = new_node_id
                        
                            # Create the special <COPY_PREFIX> node.
                            copy_node_data = {
                                'A': self.registry.canonical_to_index["<COPY_PREFIX>"],
                                'count': count,
                                'template': { # Store the full (A,B,C) signature of the node being copied.
                                    'A': current_node['A'],
                                    'B': current_node['B'],
                                    'C': current_node['C']
                                },
                                'D': [],
                                'name': f"copy_{op_to_copy_str}_x{count}"
                            }
                            compressed_nodes.append(copy_node_data)
                            i += count # Jump the pointer past the entire sequence.
                            continue

                    # If no compression occurred, just copy the node.
                    old_to_new_id_map[current_node['original_id']] = new_node_id
                    compressed_nodes.append(current_node)
                    i += 1

                # --- STAGE 3: Remap Input Dependencies ---
                # Update the 'D' fields in all nodes to point to the new, post-compression IDs.
                for node in compressed_nodes:
                    if node['A'] == self.registry.canonical_to_index["<COPY_PREFIX>"]: continue
                    old_inputs = node.get('D', [])
                    new_inputs = [old_to_new_id_map[old_id] for old_id in old_inputs if old_id in old_to_new_id_map]
                    node['D'] = sorted(list(set(new_inputs)))

                # --- STAGE 4: Final Binary Encoding ---
                # Convert the list of compressed node dictionaries into a GraphRepresentation object.
                graph_repr = GraphRepresentation(f"{model_name}_{part_name}", self.registry)
                for node_data in compressed_nodes:
                    A = node_data['A']
                    D = node_data['D']
                
                    if A == self.registry.canonical_to_index["<COPY_PREFIX>"]:
                        # Encode the COPY node using the special binary format.
                        template = node_data['template']
                        count = node_data['count']
                        B = template['A'] # The 'B' field stores the template's operation ID.
                        C = count         # The 'C' field stores the repetition count.
                        # The template's B and C are "hidden" in the D field for storage.
                        D_extra = [template['B'], template['C']]
                        fx_node_mock = type('obj', (object,), {'name': node_data['name']})()
                        graph_repr.add_op_node(A, B, C, D_extra, fx_node_mock)
                    else:
                        # Encode a standard node.
                        B = node_data['B']
                        C = node_data['C']
                        fx_node = node_data['fx_node']
                        graph_repr.add_op_node(A, B, C, D, fx_node)
            
                results[part_name] = graph_repr
                print(f"    {part_name.capitalize()}: Original {len(gm.graph.nodes)} nodes -> compressed (RLE) to {graph_repr.get_node_count()} nodes.")
    
        return results

    def _get_dummy_inputs(self, part_name: str, module: nn.Module, config: Optional[Any], model_meta: Dict) -> Tuple:
        dummy_args, dummy_kwargs = (), {}
        model_type = model_meta.get("model_type", "nlp") # By default consider the NLP model

        # Logic for CV models
        if model_type == "vision":
            if part_name == 'base_model':
                # ResNet input on CIFAR-10: (batch, channels, height, width)
                dummy_args = (torch.randn(1, 3, 32, 32),)
            elif part_name == 'head':
                # The output of the avgpool layer before fc in ResNet-18 is (batch, 512, 1, 1),
                # which then compresses to (batch, 512)
                dummy_args = (torch.randn(1, 512),)
            return dummy_args, dummy_kwargs
        
        # Old logic for NLP models
        if part_name == 'base_model':
            dummy_kwargs = {"input_ids": torch.zeros((1, 64), dtype=torch.long)}
            if config and hasattr(config, 'type_vocab_size'):
                dummy_kwargs["token_type_ids"] = torch.zeros((1, 64), dtype=torch.long)
            if not (config and hasattr(config, 'is_decoder') and config.is_decoder):
                 dummy_kwargs["attention_mask"] = torch.ones((1, 64), dtype=torch.long)
            if hasattr(module, 'forward') and 'use_cache' in module.forward.__code__.co_varnames:
                dummy_kwargs["use_cache"] = False
        elif part_name == 'head':
            if config and hasattr(config, 'hidden_size'):
                dummy_args = (torch.randn(1, 64, config.hidden_size),)
            else:
                dummy_args = (torch.randn(1, 64, 768),)
        else:
            return None, None
        return dummy_args, dummy_kwargs

class KnowledgeBase:
    """
    A read-only container for accessing the contents of a saved `registry.json`.
    Its purpose is to load the knowledge base from disk once and provide a simple,
    structured interface for other components (like a future `Visualizer` or `Interpreter`)
    to look up information about operation names, constants, etc.

    NOTE: This class is currently somewhat redundant as other components often load the
    `OperationRegistry` directly. It could be expanded into a more comprehensive query engine
    for the knowledge base in the future.
    """
    def __init__(self, registry_path: str):
        # [CONSTRUCTOR]
        # Purpose: Loads the registry file from disk and populates the instance attributes.
        print(f"Loading knowledge base from {registry_path}...")
        # Step 1: Validate that the registry file actually exists.
        if not os.path.exists(registry_path):
            raise FileNotFoundError(f"Registry file not found: {registry_path}")
        
        # Step 2: Open the file and load its contents as a JSON object.
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry_data = json.load(f)
        
        # Step 3: Populate the instance attributes by extracting data from the loaded JSON.
        # Use `.get(key, {})` to provide a default empty dictionary, ensuring backward
        # compatibility if the file format changes.

        # Load Operation Mappings 
        # Keys from JSON are strings, so they must be cast back to integers.
        self.index_to_canonical: Dict[int, str] = {int(k): v for k, v in registry_data.get('index_to_canonical', {}).items()}
        
        # Load Variation Mappings 
        self.variation_to_index: Dict[str, int] = registry_data.get('variation_to_index', {})
        # Create a reverse mapping for convenience (e.g., for decoding/visualization).
        self.index_to_variation: Dict[int, str] = {v: k for k, v in self.variation_to_index.items()}
        
        # Load Constant Mappings 
        # NOTE: This loads the string-representation of constants, which is mainly for debugging/display.
        # The `Reconstructor` uses the fully deserialized constants from `OperationRegistry`.
        self.index_to_constant: Dict[int, str] = {int(k): v for k, v in registry_data.get('index_to_constant', {}).items()}
        self.index_to_constant_group: Dict[int, str] = {int(k): v for k, v in registry_data.get('index_to_constant_group', {}).items()}
        
        print("Knowledge base loaded successfully.")

class Tokenizer:
    """
    Converts a binary graph stream into a sequence of integer tokens.
    This class serves a similar purpose to a text tokenizer in NLP: it builds a vocabulary
    of unique "words" (where each "word" is a unique binary representation of a graph node)
    and maps each word to a unique integer ID.

    It supports two modes:
    1. `normalize=True`: Converts absolute input IDs to relative offsets, making patterns
       position-independent and discoverable. Used for pattern mining.
    2. `normalize=False`: Keeps the binary representation as-is. Used for looking up
       already-normalized tokens, like those from a pattern definition.
    """
    def __init__(self):
        # [CONSTRUCTOR]
        # Purpose: Initializes the vocabulary dictionaries.
        
        # Maps a binary token (bytes) to its unique integer ID.
        self.token_to_id: Dict[bytes, int] = {}
        # The reverse mapping: integer ID back to the binary token.
        self.id_to_token: Dict[int, bytes] = {}
        # A counter to assign the next available integer ID.
        self.next_id = 0

    def tokenize_stream(self, binary_stream: bytes, normalize: bool = True) -> List[int]:
        """
        The main method for tokenizing a binary graph.
        """
        # Step 1: Slice the continuous binary stream into a list of individual node tokens.
        raw_tokens = self._slice_stream_into_tokens(binary_stream)
        
        token_ids = []
        # Step 2: Process each raw binary token.
        for i, token_bytes in enumerate(raw_tokens):
            if normalize:
                # Step 3a: If normalization is enabled, process the token to ensure its
                # input dependencies are relative offsets.
                processed_token_bytes = self._normalize_token(token_bytes, current_node_id=i)
            else:
                # Step 3b: If normalization is disabled, use the raw token as is. This is
                # used when we want to find the ID for an already-normalized binary token.
                processed_token_bytes = token_bytes
            
            # Step 4: Check if this processed token is already in our vocabulary.
            if processed_token_bytes not in self.token_to_id:
                # If it's a new "word", add it to the vocabulary and assign it a new ID.
                self.token_to_id[processed_token_bytes] = self.next_id
                self.id_to_token[self.next_id] = processed_token_bytes
                self.next_id += 1
            
            # Step 5: Append the integer ID for this token to the final sequence.
            token_ids.append(self.token_to_id[processed_token_bytes])
            
        return token_ids

    def _slice_stream_into_tokens(self, binary_stream: bytes) -> List[bytes]:
        """Helper function to parse a byte stream into a list of variable-length node tokens."""
        tokens = []
        pointer = 0
        while pointer < len(binary_stream):
            # Step 1: A node's header (A, B, C, num_inputs) is always 5 bytes.
            if pointer + 5 > len(binary_stream): break
            
            # Step 2: The 5th byte (at index `pointer + 4`) specifies the number of inputs.
            num_inputs = binary_stream[pointer + 4]
            # Each input ID is 2 bytes, so calculate the total length of the 'D' part.
            d_len = num_inputs * 2
            # The total token length is the header (5 bytes) + the 'D' part.
            token_len = 5 + d_len

            # Step 3: Check if the stream contains the full token.
            if pointer + token_len > len(binary_stream): break

            # Step 4: Slice the full token from the stream and add it to our list.
            token = binary_stream[pointer : pointer + token_len]
            tokens.append(token)
            # Step 5: Advance the pointer by the length of the token we just read.
            pointer += token_len
        return tokens

    def _normalize_token(self, token_bytes: bytes, current_node_id: int) -> bytes:
        """
        Ensures a token's input dependencies ('D' field) are relative offsets.
        This function is idempotent: it correctly handles both absolute IDs (from raw graphs)
        and existing relative offsets (from pattern definitions), producing a consistent,
        normalized binary token.
        """
        # Step 1: Unpack the fixed-size header.
        header = token_bytes[:5]
        num_inputs = header[4]
        
        # Step 2: Special handling for COPY nodes.
        A = header[0]
        # Lazily initialize the copy prefix ID to avoid requiring the registry in the constructor.
        if not hasattr(self, '_copy_prefix_id'):
            temp_registry = OperationRegistry()
            temp_registry.init_special_tokens()
            self._copy_prefix_id = temp_registry.canonical_to_index.get('<COPY_PREFIX>')

        # The 'D' field of a COPY node contains template data, not input IDs.
        # It must not be normalized. Return the token unmodified.
        if A == self._copy_prefix_id:
            return token_bytes

        # If the node has no inputs, there's nothing to normalize.
        if num_inputs == 0:
            return token_bytes

        # Step 3: Prepare to build a new, normalized 'D' component.
        d_bytes = token_bytes[5:]
        new_d_bytes = bytearray()

        # Step 4: Iterate through each 2-byte value in the 'D' field.
        for i in range(num_inputs):
            # Always read as a signed integer, as it could be a positive absolute ID or a negative relative offset.
            input_val = int.from_bytes(d_bytes[i*2 : i*2+2], 'big', signed=True)
            
            # Step 5: Determine if normalization is needed.
            if input_val < 0:
                # If the value is already negative, it's a relative offset from a pattern definition.
                # No conversion is needed; the value is already "normalized".
                relative_offset = input_val
            else:
                # If the value is non-negative, it's an absolute ID from a raw graph.
                # Convert it to a relative offset.
                # Example: current node is #15, its input is node #12 -> offset = 12 - 15 = -3.
                relative_offset = input_val - current_node_id
            
            # Step 6: Append the calculated (or original) relative offset to the new 'D' component.
            new_d_bytes.extend(relative_offset.to_bytes(2, 'big', signed=True))
            
        # Step 7: Concatenate the original header with the new, normalized 'D' component to form the final token.
        return bytes(header) + bytes(new_d_bytes)

class PatternMiner:
    """
    This class is responsible for discovering meaningful, recurring subgraphs (patterns)
    from a collection of tokenized graph sequences. It implements algorithms for finding,
    ranking, and selecting the most significant patterns.
    """
    def __init__(self, kb: KnowledgeBase):
        # [CONSTRUCTOR]
        # Purpose: Initializes the miner.
        # The KnowledgeBase (`kb`) is currently unused but is kept for potential future
        # enhancements, such as using semantic information from the KB to guide mining.
        self.kb = kb

    def find_and_rank_patterns(
        self, 
        sequences: List[Tuple[int, ...]], 
        max_len: int = 30, 
        min_len: int = 5
    ) -> List[Dict]:
        """
        Finds all possible subsequences (n-grams) within the given length constraints,
        counts their frequencies, and ranks them by a heuristic score.
        This is a brute-force approach to identify all potential patterns.
        """
        print(f"Searching for patterns with lengths from {min_len} to {max_len}...")
        # Use a defaultdict to easily count occurrences of each n-gram.
        ngram_counts = defaultdict(int)
        
        # Step 1: Iterate through every provided sequence of token IDs.
        for seq in sequences:
            # Step 2: For each sequence, iterate through all possible pattern lengths.
            for length in range(min_len, max_len + 1):
                if len(seq) < length:
                    continue # Skip if the sequence is shorter than the desired pattern length.
                # Step 3: Slide a window of the current `length` across the sequence.
                for i in range(len(seq) - length + 1):
                    # Extract the subsequence (the n-gram).
                    ngram = seq[i : i + length]
                    # Increment the count for this specific n-gram.
                    ngram_counts[ngram] += 1
        
        # Step 4: Rank the discovered patterns.
        ranked_patterns = []
        for ngram, freq in ngram_counts.items():
            # We are only interested in patterns that appear more than once.
            if freq > 1:
                # The scoring metric is `frequency * length`. This prioritizes patterns
                # that are both common and long, as they offer the best compression potential.
                score = freq * len(ngram)
                ranked_patterns.append({
                    'sequence': list(ngram), # The pattern as a list of token IDs.
                    'frequency': freq,
                    'length': len(ngram),
                    'score': score
                })
        
        # Step 5: Sort the patterns in descending order based on their score.
        ranked_patterns.sort(key=lambda x: x['score'], reverse=True)
        print(f"Found {len(ranked_patterns)} unique, recurring patterns.")
        return ranked_patterns

    def select_best_non_overlapping_patterns(self, ranked_patterns: List[Dict]) -> List[Dict]:
        """
        Selects the best patterns from a ranked list, ensuring that no selected
        pattern is a sub-pattern of another, more significant (higher-ranked) pattern.
        This prunes the vast number of discovered patterns to a smaller, more meaningful set.
        """
        print("Selecting the best non-overlapping patterns...")
        best_patterns = []
        # A set to keep track of patterns that have been "covered" by being part of a larger one.
        covered_patterns = set()

        # Iterate through the patterns, starting from the highest-scored.
        for pattern_data in ranked_patterns:
            current_pattern_tuple = tuple(pattern_data['sequence'])
            
            # If this pattern has already been identified as a sub-pattern of a
            # previously selected larger pattern, skip it.
            if current_pattern_tuple in covered_patterns:
                continue

            # This pattern is a "maximal" one so far, so add it to our final list.
            best_patterns.append(pattern_data)
            
            # Now, find all possible sub-patterns within the one we just selected and
            # add them to the `covered_patterns` set. This ensures they won't be
            # selected later on their own.
            for length in range(pattern_data['length'] - 1, 0, -1):
                for i in range(pattern_data['length'] - length + 1):
                    sub_pattern = current_pattern_tuple[i : i + length]
                    covered_patterns.add(sub_pattern)
        
        print(f"Selected {len(best_patterns)} key patterns after eliminating overlaps.")
        return best_patterns

    def save_patterns_to_registry(self, patterns: List[Dict], id_to_token: Dict[int, bytes], registry: 'OperationRegistry', db_manager: 'DatabaseManager'):
        """
        Saves the selected patterns to the `OperationRegistry` if they are not already present.
        """
        # Determine the next available ID for a new pattern, starting after the reserved range.
        next_pattern_id = registry.RESERVED_RANGE
        if registry.patterns:
            max_id = max(registry.patterns.keys(), default=registry.RESERVED_RANGE - 1)
            next_pattern_id = max(registry.RESERVED_RANGE, max_id + 1)

        new_patterns_added = 0
        for data in patterns:
            # Step 1: "De-tokenize" the pattern's integer sequence back into its binary representation.
            pattern_binary = b"".join([id_to_token[token_id] for token_id in data['sequence']])
            # Step 2: Encode the binary representation into Base64 for JSON serialization.
            pattern_b64 = base64.b64encode(pattern_binary).decode('ascii')
            
            # Step 3: Check if this exact pattern already exists in the registry to avoid duplicates.
            if pattern_b64 not in registry.pattern_to_index:
                # The pattern ID is encoded as a 2-byte value, so it cannot exceed 65535.
                if next_pattern_id > 65535: break
                
                print(f"  -> Adding new pattern ID={next_pattern_id}: len={data['length']}, freq={data['frequency']}, score={data['score']}")
                
                # Step 4: Add the new pattern to all relevant registry mappings.
                registry.patterns[next_pattern_id] = pattern_b64
                registry.pattern_to_index[pattern_b64] = next_pattern_id
                registry.index_to_pattern[next_pattern_id] = pattern_b64
                
                next_pattern_id += 1
                new_patterns_added += 1
        
        # Step 5: If any new patterns were added, save the updated registry to disk.
        if new_patterns_added > 0:
            db_manager.save_registry(registry)
            print(f"Registry updated. Added {new_patterns_added} new patterns.")
        else:
            print("No new unique patterns were found to add.")

    def select_best_iteratively(self, sequences: List[Tuple[int, ...]], min_len=5, max_len=40, min_score=50) -> List[Dict]:
        """
        An iterative greedy algorithm to find the best patterns. In each step, it finds the
        single highest-scoring pattern, adds it to the results, and then "replaces" all
        occurrences of that pattern in the sequences with a placeholder. This helps uncover
        nested or hierarchical patterns.
        """
        print("Iteratively selecting best patterns by 'score' metric...")
        best_patterns = []
        # Create a mutable copy of the sequences.
        working_sequences = [list(seq) for seq in sequences]
        
        iteration = 1
        while True:
            print(f"\n--- Pattern Selection Iteration #{iteration} ")
            
            # Step 1: Find all n-grams and their frequencies in the current state of the sequences.
            ngram_counts = defaultdict(int)
            for seq in working_sequences:
                for length in range(min_len, max_len + 1):
                    if len(seq) < length: continue
                    for i in range(len(seq) - length + 1):
                        ngram = tuple(seq[i : i + length])
                        # Ignore n-grams that contain the placeholder from a previous iteration.
                        if -1 in ngram:
                            continue
                        ngram_counts[ngram] += 1
            
            if not ngram_counts:
                print("No more recurring patterns found. Search complete.")
                break

            # Step 2: Find the single best pattern in this iteration based on the score.
            best_current_pattern = None
            max_score = -1
            for ngram, freq in ngram_counts.items():
                if freq > 1:
                    score = freq * len(ngram)
                    if score > max_score:
                        max_score = score
                        best_current_pattern = ngram
            
            # Step 3: Stop if no significant patterns are left.
            if best_current_pattern is None or max_score < min_score:
                print(f"Best remaining pattern has score={max_score}, which is below the threshold of {min_score}. Search complete.")
                break
            
            # Step 4: Add the best pattern found in this iteration to our results.
            pattern_len = len(best_current_pattern)
            pattern_freq = ngram_counts[best_current_pattern]
            print(f"Selected best pattern: length={pattern_len}, frequency={pattern_freq}, score={max_score}")
            best_patterns.append({
                'sequence': list(best_current_pattern),
                'frequency': pattern_freq,
                'length': pattern_len,
                'score': max_score
            })

            # Step 5: Replace all occurrences of the found pattern with a placeholder (-1).
            placeholder = -1
            new_sequences = []
            for seq in working_sequences:
                new_seq = []
                i = 0
                while i < len(seq):
                    if i + pattern_len <= len(seq) and tuple(seq[i:i+pattern_len]) == best_current_pattern:
                        new_seq.append(placeholder)
                        i += pattern_len # Jump the pointer past the pattern.
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_sequences.append(new_seq)
            # Update the working sequences for the next iteration.
            working_sequences = new_sequences
            
            iteration += 1

        print(f"\nSelected {len(best_patterns)} key patterns.")
        return best_patterns

class Compressor:
    def __init__(self, registry: OperationRegistry, tokenizer: 'Tokenizer'):
        self.registry = registry
        self.tokenizer = tokenizer
        
        self.token_seq_to_pattern_id: Dict[Tuple[int, ...], int] = {}
        for pattern_id, b64_pattern in registry.index_to_pattern.items():
            binary_pattern = base64.b64decode(b64_pattern)
            # The pattern in the registry is ALREADY normalized. We need to get its
            # canonical ID sequence from our universal tokenizer.
            token_ids = tuple(self.tokenizer.tokenize_stream(binary_pattern, normalize=False))
            self.token_seq_to_pattern_id[token_ids] = pattern_id

        self.sorted_patterns = sorted(
            self.token_seq_to_pattern_id.items(),
            key=lambda item: len(item[0]),
            reverse=True
        )

    def compress_graph(self, graph_name: str, raw_nodes: List[Dict]) -> 'GraphRepresentation':
        """
        Compresses a "raw" graph (list of node dictionaries) in a single pass.
        """
        if not self.sorted_patterns:
            print("    -> No patterns found for compression.")
            return self._re_encode(graph_name, raw_nodes) # Re-encode without changes.
        initial_node_count = len(raw_nodes)

        # SINGLE-PASS GREEDY ALGORITHM 
        
        # Step 1: Convert the input graph into a normalized token sequence.
        # This is done only once at the beginning for efficiency.
        initial_binary_stream = self._nodes_to_stream(raw_nodes)
        token_ids = self.tokenizer.tokenize_stream(initial_binary_stream, normalize=True)

        final_nodes = [] # This will hold the new, compressed list of nodes.
        # This map tracks how original node indices map to new indices after compression.
        old_to_new_id_map = {}
        
        # `i` is a pointer to our current position in the original `raw_nodes` list.
        i = 0
        while i < len(raw_nodes):
            match_found = False
            # Step 2: Try to match the longest patterns first at the current position `i`.
            for pattern_token_seq, pattern_id in self.sorted_patterns:
                pattern_len = len(pattern_token_seq)
                
                # Check if the token sequence starting at `i` matches the current pattern.
                if i + pattern_len <= len(token_ids) and tuple(token_ids[i:i+pattern_len]) == pattern_token_seq:
                    print(f"    -> Collapsing pattern ID={pattern_id} (length {pattern_len}) at position {i}")
                    
                    # A match is found! Create a single macro-node to replace the sequence.
                    new_node_id = len(final_nodes)
                    
                    # All original nodes covered by this pattern now map to this single new node ID.
                    for j in range(i, i + pattern_len):
                        old_to_new_id_map[j] = new_node_id
                    
                    # Step 3: Determine the inputs for this new macro-node.
                    # An input is any node that was an input to the original sequence
                    # but was NOT part of the sequence itself.
                    nodes_in_pattern_ids = set(range(i, i + pattern_len))
                    macro_inputs_old_ids = []
                    for j in range(i, i + pattern_len):
                        for input_id in raw_nodes[j]['D']:
                            if input_id not in nodes_in_pattern_ids:
                                macro_inputs_old_ids.append(input_id)
                    # Deduplicate and sort for a canonical representation.
                    macro_inputs_old_ids = sorted(list(set(macro_inputs_old_ids)))
                    
                    # Create the macro-node dictionary.
                    macro_node = {
                        'A': self.registry.canonical_to_index["<PATTERN_PREFIX>"], 
                        'pattern_id': pattern_id, 
                        'D_old': macro_inputs_old_ids, # Temporarily store old input IDs.
                        'name': f'pattern_{pattern_id}_at_{new_node_id}'
                    }
                    final_nodes.append(macro_node)
                    
                    # Advance the pointer `i` past the entire matched sequence.
                    i += pattern_len
                    match_found = True
                    break # Since we found the longest possible match, break to the outer loop.
            
            if not match_found:
                # If no pattern matched at position `i`, simply copy the original node.
                new_node_id = len(final_nodes)
                old_to_new_id_map[i] = new_node_id
                final_nodes.append(raw_nodes[i])
                i += 1

        # Step 4: Final pass to remap all input dependencies to their new IDs.
        for node in final_nodes:
            # Get the list of original input IDs (either from 'D_old' for patterns or 'D' for others).
            old_inputs = node.pop('D_old', node.get('D', []))
            # Look up each old ID in the map to find its new, compressed ID.
            new_inputs = [old_to_new_id_map[old_id] for old_id in old_inputs if old_id in old_to_new_id_map]
            node['D'] = sorted(list(set(new_inputs)))

        final_node_count = len(final_nodes)
        compression_ratio = initial_node_count / final_node_count if final_node_count > 0 else 0
        print(f"    -> Graph compressed from {initial_node_count} to {final_node_count} nodes (x{compression_ratio:.1f} compression).")
        
        # Step 5: Convert the final list of node dictionaries back into a binary GraphRepresentation.
        return self._re_encode(graph_name, final_nodes)

    def _tokens_to_nodes(self, token_ids: List[int]) -> List[Dict]:
        """(Helper) Converts a sequence of token IDs back into a list of raw node dictionaries."""
        nodes = []
        for tid in token_ids:
            binary_token = self.tokenizer.id_to_token[tid]
            A, B, C, D = self._decode_token(binary_token)
            nodes.append({'A': A, 'B': B, 'C': C, 'D': D})
        return nodes

    def _nodes_to_stream(self, nodes: List[Dict]) -> bytes:
        """
        (Helper) Converts a list of node dictionaries into a single binary stream.
        This is the inverse of the Decoder's main functionality.
        """
        #  Start with an empty, mutable byte array.
        binary_stream = bytearray()
        #  Pre-fetch special token IDs for efficiency.
        pattern_prefix_id = self.registry.canonical_to_index.get("<PATTERN_PREFIX>")
        copy_prefix_id = self.registry.canonical_to_index.get("<COPY_PREFIX>")

        #  Iterate through each node dictionary in the list.
        for node in nodes:
            A = node['A']
            
            #  Dispatch based on the node's operation type 'A'.
            if A == pattern_prefix_id:
                # ENCODE A PATTERN NODE 
                #  Extract the pattern ID from the dictionary.
                pattern_id = node['pattern_id']
                #  Split the 16-bit ID into two 8-bit bytes (E0, E1).
                E0 = pattern_id & 0xFF
                E1 = (pattern_id >> 8) & 0xFF
                #  Append the binary components in A-E-E format.
                binary_stream.append(A)
                binary_stream.append(E0)
                binary_stream.append(E1)
                binary_stream.append(0) # Padding byte for the second half of 'C'.
                
                #  Encode the 'D' component (inputs to the pattern).
                D = node.get('D', [])
                binary_stream.append(len(D))
                for input_id in D:
                    # Input IDs for patterns are absolute node indices and are always non-negative. it should be `signed=False`.
                    binary_stream.extend(input_id.to_bytes(2, 'big', signed=True))

            elif A == copy_prefix_id:
                # ENCODE A COPY NODE 
                #  Extract the template and count from the dictionary.
                template = node['template']
                count = node['count']
                
                #  Append the binary components using the special COPY format.
                binary_stream.append(A)              # 'A' = <COPY_PREFIX> ID.
                binary_stream.append(template['A'])  # 'B' field = template's 'A'.
                binary_stream.extend(count.to_bytes(2, 'big', signed=True)) # 'C' field = repetition count.
                
                #  The 'D' field is repurposed to store the template's 'B' and 'C'.
                D_extra = [template['B'], template['C']]
                binary_stream.append(len(D_extra))
                #  Encode both values as 2-byte signed integers for consistency,
                # as template['C'] can be negative (representing a constant group).
                binary_stream.extend(D_extra[0].to_bytes(2, 'big', signed=True))
                binary_stream.extend(D_extra[1].to_bytes(2, 'big', signed=True))

            else:
                # ENCODE A STANDARD NODE 
                #  Append the A, B, and C components directly from the dictionary.
                binary_stream.append(node['A'])
                binary_stream.append(node['B'])
                binary_stream.extend(node['C'].to_bytes(2, 'big', signed=True))
                
                #  Encode the 'D' component (standard input dependencies).
                D = node.get('D', [])
                binary_stream.append(len(D))
                for input_id in D:
                    # Input IDs for standard nodes are absolute node indices, which are always non-negative. it should be `signed=False`.
                    binary_stream.extend(input_id.to_bytes(2, 'big', signed=True))

        #  Return the final, immutable byte string.
        return bytes(binary_stream)

    def _re_encode(self, graph_name: str, nodes: List[Dict]) -> 'GraphRepresentation':
        """
        (Helper) Converts the final (compressed) list of node dictionaries into a GraphRepresentation object.
        This is the last step of compression, creating the final binary artifact.
        """
        #  Initialize a new, empty GraphRepresentation to build into.
        new_graph_repr = GraphRepresentation(graph_name, self.registry)
        
        #  Pre-fetch special token IDs.
        pattern_prefix_id = self.registry.canonical_to_index.get("<PATTERN_PREFIX>")
        copy_prefix_id = self.registry.canonical_to_index.get("<COPY_PREFIX>")
        
        #  Iterate through the final list of nodes.
        for i, node_data in enumerate(nodes):
            A = node_data['A']
            D = node_data.get('D', []) # `D` will be empty for COPY nodes.
            
            #  Dispatch based on the node's operation type 'A'.
            if A == pattern_prefix_id:
                # RE-ENCODE A PATTERN NODE 
                pattern_id = node_data['pattern_id']
                name = node_data.get('name', f'pattern_{pattern_id}_at_{i}')
                #  Use the dedicated method on GraphRepresentation to add a pattern node.
                new_graph_repr.add_pattern_node(pattern_id, D, name)

            elif A == copy_prefix_id:
                # RE-ENCODE A COPY NODE 
                template = node_data['template']
                count = node_data['count']
                
                #  Prepare the components for the special binary format.
                B_encoded = template['A']
                C_encoded = count
                #  The template's 'B' and 'C' are passed via the 'D' argument.
                D_encoded_extra = [template['B'], template['C']]
                
                name = node_data.get('name', f'copy_at_{i}')
                #  A mock fx.Node is needed to satisfy the `add_op_node` interface.
                fx_node_mock = type('obj', (object,), {'name': name})()
                #  Use the standard `add_op_node` method, which will correctly handle
                # the repurposed B, C, and D fields based on the `A` value being <COPY_PREFIX>.
                new_graph_repr.add_op_node(A, B_encoded, C_encoded, D_encoded_extra, fx_node_mock)

            else:
                # RE-ENCODE A STANDARD NODE 
                B = node_data['B']
                C = node_data['C']
                name = node_data.get('name', f'node_{new_graph_repr.get_node_count()}')
                fx_node_mock = type('obj', (object,), {'name': name})()
                #  Call the standard method to add a regular operation node.
                new_graph_repr.add_op_node(A, B, C, D, fx_node_mock)
                
        #  Return the fully constructed GraphRepresentation object.
        return new_graph_repr

    def _decode_token(self, token: bytes) -> Tuple:
        """
        (Helper) Decodes a single binary "word" (a node's binary representation) into its (A, B, C, D) components.
        This is a minimal decoder used for internal debugging or simple transformations, not the main `Decoder` class.
        """
        #  Unpack the fixed-size header.
        A, B = token[0], token[1]
        C = int.from_bytes(token[2:4], 'big', signed=True)
        num_inputs = token[4]
        
        #  Unpack the variable-size 'D' component.
        D = []
        ptr = 5
        for _ in range(num_inputs):
            # It should read `signed=True` to be consistent with the encoding format.
            # However, for the limited use case within this class, `signed=False` (or just `big`) might have worked by chance.
            # For correctness, it should align with how 'D' is written.
            D.append(int.from_bytes(token[ptr:ptr+2], 'big', signed=True)) 
            ptr += 2
        return A, B, C, D

class Analyzer:
    """
    Orchestrates the main workflow for pattern mining, graph compression, and rule extraction.
    This class ties together all the other components to perform a full analysis cycle on a
    set of decomposed graphs.
    """
    def __init__(self, registry: 'OperationRegistry', db_manager: 'DatabaseManager'):
        # [CONSTRUCTOR]
        # Purpose: Initializes the analyzer with necessary dependencies.
        self.registry = registry
        self.db_manager = db_manager
        self.decoder = Decoder(registry)

    def run(self, flat_graphs: Dict[str, 'GraphRepresentation']):
        """Executes the full analysis and compression pipeline."""
        print("\n" + "="*80)
        print("           STARTING ANALYSIS AND COMPRESSION (ITERATIVE MODE)")
        print("="*80)

        # STAGE 1: Create and Train a Universal Tokenizer 
        print("\n--- STAGE 1: Creating and Training a Universal Tokenizer ")
        # Create a SINGLE tokenizer instance that will be used for the entire process.
        # This ensures that token IDs are consistent between pattern mining and compression.
        universal_tokenizer = Tokenizer()
        
        # "Train" the tokenizer by feeding it all possible "words" (binary node tokens)
        # from all available graphs.
        for graph_repr in flat_graphs.values():
            binary_stream = base64.b64decode(graph_repr.to_base64())
            # We tokenize twice: once normalized and once unnormalized. This guarantees
            # that the tokenizer's vocabulary contains every possible binary token variant
            # it might encounter later.
            universal_tokenizer.tokenize_stream(binary_stream, normalize=True)
            universal_tokenizer.tokenize_stream(binary_stream, normalize=False)
        
        print(f"Universal tokenizer created. Vocabulary contains {len(universal_tokenizer.token_to_id)} unique 'words'.")

        # STAGE 2: Mine for New Patterns 
        print("\n--- STAGE 2: Mining for New Patterns ")
        all_tokenized_sequences = []
        # Define the set of non-computational tokens to be ignored during pattern mining.
        # Patterns should represent computational logic, not data loading or graph structure.
        ignored_token_ids = {
            self.registry.canonical_to_index.get(token)
            for token in ["<DATA_INPUT>", "<PARAM_INPUT>", "<CONST_REF>", "<OUTPUT>", "<NONE>"]
        }

        print("Tokenizing graphs for analysis (using the universal tokenizer)...")
        for name, graph_repr in flat_graphs.items():
            binary_stream = base64.b64decode(graph_repr.to_base64())
            # Get the normalized token ID sequence from the pre-trained tokenizer.
            full_token_ids = universal_tokenizer.tokenize_stream(binary_stream, normalize=True)
            
            # To filter the sequence, we need to inspect the 'A' component of each token.
            # First, get the binary representation for each token ID.
            binary_tokens = [universal_tokenizer.id_to_token[tid] for tid in full_token_ids]
            
            # Build a new sequence containing only the IDs of computational nodes.
            filtered_sequence = []
            for i, binary_token in enumerate(binary_tokens):
                A = binary_token[0] # The 'A' component is always the first byte.
                if A not in ignored_token_ids:
                    filtered_sequence.append(full_token_ids[i])

            if filtered_sequence:
                all_tokenized_sequences.append(tuple(filtered_sequence))
            
        # Run the pattern mining algorithm on the filtered, computation-only sequences.
        miner = PatternMiner(None)
        best_patterns = miner.select_best_iteratively(all_tokenized_sequences, min_len=8, max_len=40, min_score=100)
        
        if best_patterns:
            # Save any new patterns found to the registry.
            miner.save_patterns_to_registry(best_patterns, universal_tokenizer.id_to_token, self.registry, self.db_manager)
            print("\nReloading registry with new patterns...")
            self.db_manager.load_registry(self.registry)
        else:
            print("\nNo significant new patterns were found.")

        # STAGE 3: Final Compression 
        print("\n--- STAGE 3: Final Compression and Saving ")
        # Create the compressor, passing it the SAME universal tokenizer used for mining.
        # This ensures token ID consistency.
        final_compressor = Compressor(self.registry, universal_tokenizer)
        compressed_signatures = {}
        
        for name, graph_repr in flat_graphs.items():
            print(f"  Final compression of {name}...")
            # The compressor operates on the full, unfiltered graph.
            raw_nodes = self.decoder.decode_raw(graph_repr.to_base64())
            compressed_graph_repr = final_compressor.compress_graph(name, raw_nodes)
            compressed_signatures[name] = compressed_graph_repr.to_base64()

        # Save the final compressed signatures to disk.
        self.db_manager.save_model_signatures(compressed_signatures, 'compressed_signatures.json')
        
        print("\nOverwriting .b64 files with final compressed versions...")
        for name, compressed_b64 in compressed_signatures.items():
            filename = name.replace('/', '_') + '.b64'
            filepath = os.path.join(self.db_manager.db_path, filename)
            with open(filepath, 'w', encoding='ascii') as f:
                f.write(compressed_b64)
        print("Files .b64 updated.")

        # STAGE 4: Rule Extraction 
        # This stage operates on the compressed graphs to find higher-level rules.
        print("\n" + "="*80)
        print("                      STARTING RULE EXTRACTION")
        print("="*80)
        
        compressed_graphs_decoded = {}
        for name, b64_string in compressed_signatures.items():
            compressed_graphs_decoded[name] = self.decoder.decode_raw(b64_string)
            
        rule_miner = RuleMiner(self.registry)
        found_rules = rule_miner.find_rules(compressed_graphs_decoded)
        
        rule_miner.save_rules_to_registry(found_rules, os.path.join(self.db_manager.db_path, 'registry.json'))

class VerificationDecoder:
    """
    A specialized version of the Decoder designed for human-readable debugging and verification.
    Its primary purpose is to parse a binary signature and print a formatted, easy-to-understand
    representation of each node, including special handling for `PATTERN` and `COPY` commands.
    It does not return a data structure; it prints directly to the console.
    """
    def __init__(self, registry: 'OperationRegistry'):
        # [CONSTRUCTOR]
        # Purpose: Initializes the decoder with a reference to the OperationRegistry.
        self.registry = registry
        # The pattern prefix ID is stored for convenience, though it's also fetched locally in the main method.
        self.pattern_prefix_id = registry.canonical_to_index.get("<PATTERN_PREFIX>")

    def decode_and_print(self, model_part_name: str, b64_signature: str):
        """
        Decodes a Base64 signature and prints a formatted representation of its contents.
        """
        #  Print a header for the current decoding task.
        print("\n" + "-"*80)
        print(f"DECODING COMPRESSED SIGNATURE: {model_part_name}")
        print("-"*80)

        # Step 1: Decode the Base64 string into a raw byte sequence.
        try:
            binary_stream = base64.b64decode(b64_signature)
        except Exception as e:
            print(f"  Base64 decoding error: {e}")
            return

        # Step 2: Initialize a pointer and a node counter.
        pointer = 0
        node_id_counter = 0

        #  Pre-fetch special token IDs for efficiency inside the loop.
        pattern_prefix_id = self.registry.canonical_to_index.get("<PATTERN_PREFIX>")
        copy_prefix_id = self.registry.canonical_to_index.get("<COPY_PREFIX>")

        # Step 3: Loop through the binary stream until the end is reached.
        while pointer < len(binary_stream):
            try:
                # Step 4: Read the 'A' component (Operation ID).
                A = int.from_bytes(binary_stream[pointer : pointer+1], 'big')
                pointer += 1
                
                #  Initialize string representations for each component for printing.
                # Look up the human-readable name for the operation ID.
                op_str = self.registry.index_to_canonical.get(A, f"<UNKNOWN_A:{A}>")
                var_str = "N/A"
                const_str = "N/A"
                D = []

                # Step 5: Dispatch based on the operation ID to parse the rest of the node.
                if A == pattern_prefix_id:
                    # This is a PATTERN node 
                    #  Read the two bytes of the pattern ID (E0, E1).
                    E0 = int.from_bytes(binary_stream[pointer : pointer+1], 'big'); pointer += 1
                    E1 = int.from_bytes(binary_stream[pointer : pointer+1], 'big'); pointer += 1
                    _ = int.from_bytes(binary_stream[pointer : pointer+1], 'big'); pointer += 1 # Skip padding.
                    #  Reconstruct the 16-bit pattern ID.
                    pattern_id = (E1 << 8) | E0
                    op_str = f"<PATTERN ID={pattern_id}>"

                elif A == copy_prefix_id:
                    # This is a COPY node 
                    #  The 'B' field holds the ID of the operation to copy.
                    op_to_copy_id = int.from_bytes(binary_stream[pointer : pointer+1], 'big'); pointer += 1
                    #  The 'C' field holds the repetition count.
                    count = int.from_bytes(binary_stream[pointer : pointer+2], 'big', signed=True); pointer += 2
                    
                    #  Look up the name of the operation being copied for a user-friendly message.
                    op_to_copy_str = self.registry.index_to_canonical.get(op_to_copy_id, f"UNK_OP:{op_to_copy_id}")
                    op_str = f"<COPY '{op_to_copy_str}' x {count}>"

                else:
                    # This is a STANDARD node 
                    #  Read the 'B' (Variation) and 'C' (Constant) components.
                    B = int.from_bytes(binary_stream[pointer : pointer+1], 'big'); pointer += 1
                    C = int.from_bytes(binary_stream[pointer : pointer+2], 'big', signed=True); pointer += 2
                    
                    #  Prepare human-readable strings for B and C.
                    var_str = str(B)
                    if C == 0: const_str = "None"
                    elif C > 0: const_str = f"c[{C}]" # Positive C is a direct constant ID.
                    else: group_idx = -(C + 1); const_str = f"g[{group_idx}]" # Negative C is a group ID.

                # Step 6: Read the 'D' component (inputs), which is common to all formats.
                num_inputs = int.from_bytes(binary_stream[pointer : pointer+1], 'big'); pointer += 1
                for _ in range(num_inputs):
                    #  Read each 2-byte input ID as a signed integer. This is important
                    # because pattern definitions use negative relative offsets, and the COPY command
                    # uses this field to store its (potentially negative) template C value.
                    input_id = int.from_bytes(binary_stream[pointer : pointer+2], 'big', signed=True); pointer += 2
                    D.append(input_id)

                # Step 7: Print the formatted output for the current node.
                print(f"Node {node_id_counter:<3} | Inputs: {str(D):<15} | Op: {op_str:<60} | Var: {var_str:<15} | Const: {const_str}")
                node_id_counter += 1

            except IndexError:
                #  Catch errors if the stream ends unexpectedly.
                print(f"Error: Unexpected end of stream at node {node_id_counter}.")
                break
            except Exception as e:
                #  Catch any other parsing errors.
                print(f"Unknown error while parsing node {node_id_counter}: {e}")
                traceback.print_exc()
                break

class RuleMiner:
    """
    Analyzes a collection of compressed graphs to extract high-level, stable "rules"
    about model architecture. These rules represent common idioms and structural
    relationships found across different models.
    
    It searches for three types of rules:
    1.  **Composition Rules**: Frequent combinations of (Operation, Variation, Constant).
    2.  **Transition Rules**: Probabilistic sequences, e.g., "Node X is almost always followed by Node Y".
    3.  **Structural Rules**: Multi-input relationships, e.g., "Node Z is always formed from inputs X and Y".
    """
    def __init__(self, registry: 'OperationRegistry'):
        # [CONSTRUCTOR]
        # Purpose: Initializes the miner with a reference to the global registry.
        self.registry = registry
        self.decoder = Decoder(registry) # A decoder might be needed for future, more complex rule analysis.

    def find_rules(self, compressed_graphs: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """The main method to orchestrate the search for all rule types."""
        #  Execute each rule-finding method sequentially.
        transition_rules = self._find_transition_rules(compressed_graphs)
        composition_rules = self._find_composition_rules(compressed_graphs)
        structural_rules = self._find_structural_rules(compressed_graphs)
        
        #  Aggregate the results into a single dictionary for serialization.
        return {
            "transition_rules": transition_rules,
            "composition_rules": composition_rules,
            "structural_rules": structural_rules
        }

    def _find_structural_rules(self, compressed_graphs: Dict[str, List[Dict]]):
        """
        Searches for complex structural rules of the form (Ancestor_1, Ancestor_2, ...) -> Target.
        This identifies nodes that are consistently formed by a specific combination of input node types.
        """
        print("\nSearching for structural rules...")
        #  `structural_contexts` will store counts of each unique context.
        # A context is a tuple: (target_signature, (parent_1_sig, offset_1), (parent_2_sig, offset_2), ...)
        structural_contexts = defaultdict(int)

        #  Iterate through every node in every graph.
        for name, graph in compressed_graphs.items():
            for target_id, target_node in enumerate(graph):
                
                # Step 1: For the current `target_node`, identify all its direct parents.
                parents = []
                for input_id in target_node.get('D', []):
                    if input_id < len(graph): # Sanity check
                        parent_node = graph[input_id]
                        parent_sig = self._get_node_signature(parent_node)
                        #  Record the parent's signature and its relative position to the target.
                        parents.append( (parent_sig, input_id - target_id) )
                
                # Step 2: A structural dependency is only interesting if there's more than one parent.
                if len(parents) > 1:
                    # Step 3: Create a canonical key for this context.
                    # The key consists of the target node's signature followed by a sorted list of its parents.
                    # Sorting the parents makes the key independent of the input order (e.g., add(x, y) vs add(y, x)).
                    context_key = tuple(
                        [self._get_node_signature(target_node)] + sorted(parents)
                    )
                    # Step 4: Increment the count for this observed structural context.
                    structural_contexts[context_key] += 1
        
        # Step 5: Filter for strong, frequently occurring rules.
        strong_rules = {}
        for context, count in structural_contexts.items():
            if count > 10: # Threshold: the context must appear more than 10 times.
                target_sig = context[0]
                parent_sigs = context[1:]
                
                #  Format the rule into a human-readable string.
                parents_str = ", ".join([f"'{p[0]}' (at rel {p[1]})" for p in parent_sigs])
                rule_str = f"Node '{target_sig}' is formed by inputs from [{parents_str}]"
                
                strong_rules[rule_str] = {'count': count}

        print(f"Found {len(strong_rules)} strong structural rules.")
        return strong_rules

    def _find_transition_rules(self, compressed_graphs: Dict[str, List[Dict]]):
        """
        Searches for simple transition rules of the form (Source_Signature) -> (Target_Signature),
        also considering the specific connection pattern ('D' signature).
        """
        print("\nSearching for transition (syntactic) rules...")
        
        # Step 1: Build a map from each source node type to all the target nodes it connects to.
        # Structure: source_signature -> [ (target_node_dict, source_id, target_id), ... ]
        source_to_targets = defaultdict(list)
        for name, graph in compressed_graphs.items():
            for target_node_id, target_node in enumerate(graph):
                for source_node_id in target_node.get('D', []):
                    if source_node_id < len(graph):
                        source_node = graph[source_node_id]
                        source_signature = self._get_node_signature(source_node)
                        source_to_targets[source_signature].append( (target_node, source_node_id, target_node_id) )

        # Step 2: Count the occurrences of each unique transition.
        # Structure: (source_sig, target_sig, connection_sig) -> { "count": N, "examples": [...] }
        transitions = defaultdict(lambda: {"count": 0, "examples": []})
        for source_signature, targets in source_to_targets.items():
            for target_node, source_node_id, target_node_id in targets:
                target_signature = self._get_node_signature(target_node)
                
                #  Create a canonical signature for the 'D' connection pattern.
                # This describes how the source and other inputs connect to the target.
                d_signature_parts = []
                for idx, input_id in enumerate(target_node.get('D', [])):
                    if input_id == source_node_id:
                        d_signature_parts.append(f"in{idx}=SOURCE")
                    else:
                        d_signature_parts.append(f"in{idx}=rel_{input_id - target_node_id}")
                d_signature = ",".join(d_signature_parts)
                
                rule_key = (source_signature, target_signature, d_signature)
                transitions[rule_key]["count"] += 1
                
                #  Store one example of the nodes involved in this rule for visualization.
                if len(transitions[rule_key]["examples"]) < 1:
                    # This search is inefficient but acceptable for grabbing a single example.
                    graph_name = [name for name, graph in compressed_graphs.items() if target_node in graph][0]
                    source_node_ex = compressed_graphs[graph_name][source_node_id]
                    transitions[rule_key]["examples"].append( (source_node_ex, target_node) )

        # Step 3: Filter for high-probability rules and identify exceptions.
        strong_rules = {}
        #  Pre-calculate the total number of times each source signature appears.
        source_counts = {k: len(v) for k, v in source_to_targets.items()}
        for (source_sig, target_sig, d_sig), data in transitions.items():
            count = data["count"]
            total_source_count = source_counts.get(source_sig, 0)
            if total_source_count == 0: continue
            
            #  Calculate the probability: P(target | source) = count(source->target) / count(source).
            probability = count / total_source_count
            
            #  A rule is considered "strong" if its probability is > 80% and it has been seen at least 5 times.
            if probability > 0.8 and total_source_count > 5:
                rule_str = f"After '{source_sig}' -> '{target_sig}' with connection '{d_sig}'"
                if not data["examples"]: continue
                
                #  Generate a Base64 representation of the 2-node rule for visualization/testing.
                source_node_ex, target_node_ex = data["examples"][0]
                rule_b64 = self._generate_rule_b64(source_node_ex, target_node_ex)
                rule_data = {'probability': f"{probability:.2f}", 'count': count, 'b64_repr': rule_b64}

                #  If the rule is not 100% certain, list the alternative outcomes (exceptions).
                if probability < 1.0:
                    exceptions = []
                    all_transitions_from_source = [key for key in transitions if key[0] == source_sig]
                    for key in all_transitions_from_source:
                        if key != (source_sig, target_sig, d_sig):
                            ex_data = transitions[key]
                            exceptions.append(f"-> '{key[1]}' with conn '{key[2]}' ({ex_data['count']} times)")
                    if exceptions:
                        rule_data['exceptions'] = exceptions
                strong_rules[rule_str] = rule_data
        
        print(f"Found {len(strong_rules)} strong transition rules.")
        return strong_rules

    def _get_node_signature(self, node: Dict) -> str:
        """Helper to create a unique, readable string signature for a node dictionary."""
        A = node.get('A')
        if A == self.registry.canonical_to_index.get("<PATTERN_PREFIX>"):
            return f"<PATTERN ID={node.get('pattern_id', '?')}>"
        else:
            return self.registry.index_to_canonical.get(A, str(A))

    def _generate_rule_b64(self, source_node: Dict, target_node: Dict) -> str:
        """Helper to create a binary Base64 representation of a simple two-node rule."""
        temp_graph = GraphRepresentation("rule", self.registry)
        
        #  Add the source node with no inputs.
        fx_mock_0 = type('obj', (object,), {'name': 'source'})()
        if source_node.get('A') == self.registry.canonical_to_index.get("<PATTERN_PREFIX>"):
            temp_graph.add_pattern_node(source_node['pattern_id'], [], fx_mock_0.name)
        else:
            temp_graph.add_op_node(source_node['A'], source_node['B'], source_node['C'], [], fx_mock_0)
        
        #  Add the target node, with its input hardcoded to point to the source (ID=0).
        new_D = [0]
        fx_mock_1 = type('obj', (object,), {'name': 'target'})()
        if target_node.get('A') == self.registry.canonical_to_index.get("<PATTERN_PREFIX>"):
            temp_graph.add_pattern_node(target_node['pattern_id'], new_D, fx_mock_1.name)
        else:
            temp_graph.add_op_node(target_node['A'], target_node['B'], target_node['C'], new_D, fx_mock_1)
        
        return temp_graph.to_base64()

    def _find_composition_rules(self, compressed_graphs: Dict[str, List[Dict]]):
        """
        Searches for frequently occurring combinations of (A, B, C). These represent
        common "semantic idioms" or fully configured operations.
        """
        print("\nSearching for composition (semantic) rules...")
        compositions = defaultdict(int)

        for graph in compressed_graphs.values():
            for node in graph:
                #  Ignore special nodes like patterns, copy, inputs, etc.
                if node.get('A', 0) < 10: continue
                
                #  Create a key from the (A, B, C) tuple and count its occurrences.
                # Note: This won't work correctly for COPY nodes which lack 'B' and 'C'. A check is needed.
                if 'B' not in node or 'C' not in node: continue
                rule_key = (node['A'], node['B'], node['C'])
                compositions[rule_key] += 1
        
        #  Filter for frequently occurring compositions.
        strong_rules = {}
        for (A, B, C), count in compositions.items():
            if count > 20: # Threshold
                op_str = self.registry.index_to_canonical.get(A, str(A))
                const_str = "None"
                if C > 0: const_str = f"c[{C}]"
                elif C < 0: const_str = f"g[{-(C+1)}]"
                rule_str = f"Composition: Op='{op_str}', Var={B}, Const='{const_str}'"
                strong_rules[rule_str] = {'count': count}

        print(f"Found {len(strong_rules)} strong composition rules.")
        return strong_rules

    def save_rules_to_registry(self, rules: Dict, registry_path: str):
        """Adds the found rules to the main registry.json file."""
        print("\nSaving rules to registry...")
        #  Open the file in read/write mode ('r+').
        with open(registry_path, 'r+') as f:
            #  Load the existing data.
            registry_data = json.load(f)
            #  Add or overwrite the 'rules' section.
            registry_data['rules'] = rules
            #  Go back to the beginning of the file to overwrite it.
            f.seek(0)
            json.dump(registry_data, f, indent=2)
            #  Truncate the file in case the new content is smaller than the old.
            f.truncate()
        print("Registry updated with new 'rules' section.")

class Reconstructor:
    """
    Reconstructs a `torch.fx.GraphModule` from a "flat" list of node dictionaries.
    This class is the inverse of the `Decomposer`. It takes the abstract, integer-based
    representation of a graph and translates it back into an executable PyTorch model.
    """
    def __init__(self, registry: 'OperationRegistry', state_dict_path: str = None):
        """
        Initializes the Reconstructor with the necessary vocabulary and an optional state_dict.
        """
        #  Store a reference to the live `OperationRegistry` object.
        self.registry = registry
        
        #  Create direct references to the required mapping dictionaries from the registry
        # for convenient access during the build process.
        self.index_to_constant = self.registry.index_to_constant         # Maps ID -> literal constant value
        self.index_to_constant_group = self.registry.index_to_constant_group # Maps Group ID -> string of constants
        self.index_to_param_name = self.registry.index_to_param_name     # Maps ID -> parameter name string

        #  Optionally load a state dictionary. This is not used in the graph construction
        # itself but would be necessary to populate the reconstructed module with actual weights.
        if state_dict_path:
            self.state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        else:
            self.state_dict = {}

    def build_module(self, flat_nodes: List[Dict]) -> Optional[fx.GraphModule]:
        """The main reconstruction method."""
        #  Step 1: Create an empty `fx.Graph`. This is the canvas on which we will draw the new graph.
        graph = fx.Graph()
        #  This dictionary will map the integer ID of a node in our `flat_nodes` list
        # to the actual `fx.Node` object created in the new graph. It's crucial for wiring inputs.
        reconstructed_nodes: Dict[int, fx.Node] = {}
        
        data_input_counter = 0
        
        #  Step 2: Iterate through each node dictionary from the "unrolled" flat list.
        for node_id, p_node in enumerate(flat_nodes):
            #  Get the operation signature (e.g., "<DATA_INPUT>", "aten.add.Tensor:...") from the 'A' component.
            A = p_node['A']
            op_signature = self.registry.index_to_canonical.get(A)
            
            new_node = None # Will hold the newly created fx.Node.

            #  Step 3: Dispatch based on the operation signature to create the appropriate fx.Node.
            if op_signature == "<DATA_INPUT>":
                #  This represents a data input to the model. Create a `placeholder` node.
                new_node = graph.placeholder(f"data_input_{data_input_counter}")
                data_input_counter += 1
            
            elif op_signature == "<PARAM_INPUT>":
                 #  This is a legacy type for a parameter input. Also create a `placeholder`.
                 new_node = graph.placeholder(f"param_input_{node_id}")

            elif op_signature == "<CONST_REF>":
                #  This represents a reference to a model parameter (weight/buffer).
                # This must be converted into a `get_attr` node.
                param_id = p_node['C']
                #  Look up the parameter's string name using its ID in the correct dictionary.
                target_name = self.index_to_param_name.get(param_id)
                
                if target_name and isinstance(target_name, str):
                    #  If the name is found, create the `get_attr` node.
                    new_node = graph.get_attr(target_name)
                else:
                    #  If the ID is invalid, print a warning and skip this node.
                    print(f"WARNING: Could not find parameter name for CONST_REF with ID {param_id}. Skipping node.")
                    continue

            elif op_signature == "<OUTPUT>":
                #  This is the graph's output node.
                #  Gather all its input nodes from the `reconstructed_nodes` map.
                output_args = tuple(reconstructed_nodes[i] for i in p_node['D'] if i in reconstructed_nodes)
                if not output_args:
                    print("WARNING: Output node has no inputs. Skipping.")
                    continue
                #  Create the `output` node in the graph. FX handles single vs. multiple outputs.
                graph.output(output_args[0] if len(output_args) == 1 else output_args)
                continue # No further processing needed for the output node.

            else: # This handles all other callable nodes (call_function, call_method, etc.).
                #  Step 3a: Resolve the operation signature string into a callable Python object.
                target = self._resolve_target(op_signature)
                if target is None:
                    print(f"WARNING: Could not resolve target for '{op_signature}'. Skipping node {node_id}.")
                    continue
                
                #  Step 3b: Gather the node's inputs. Tensor inputs are those that are themselves nodes.
                tensor_inputs = [reconstructed_nodes[i] for i in p_node['D'] if i in reconstructed_nodes]
                #  Constant inputs are retrieved from the 'C' component.
                constant_inputs = self._get_constants_from_C(p_node['C'])
                
                #  Step 3c: Assemble the final arguments (`args`) and keyword arguments (`kwargs`) for the call.
                final_args = []
                final_kwargs = {}

                if 'getitem' in op_signature: # Special handling for `getitem`.
                    if tensor_inputs: final_args.append(tensor_inputs[0])
                    final_args.append(constant_inputs.get('arg1', constant_inputs.get('arg_const')))
                else: # Generic handling.
                    final_kwargs.update(constant_inputs)
                    final_args.extend(tensor_inputs)

                #  Step 3d: Create the `call_function` node in the graph.
                new_node = graph.call_function(target, args=tuple(final_args), kwargs=final_kwargs)

            #  Step 4: If a new node was created, add it to our mapping dictionary.
            if new_node:
                reconstructed_nodes[node_id] = new_node

        #  Step 5: Finalize the `fx.GraphModule`.
        try:
            graph.lint() # Check for errors in the constructed graph.

            #  Create an empty nn.Module. This will serve as the "owner" of the parameters.
            reconstructed_module = torch.nn.Module()

            #  Iterate through the final graph to find all `get_attr` nodes.
            for node in graph.nodes:
                if node.op == 'get_attr':
                    #  For each `get_attr` node, we must add a corresponding `nn.Parameter` to our
                    # `reconstructed_module`. This is crucial for `fx.GraphModule` to be valid.
                    # The actual tensor values don't matter at this stage, only their existence.
                    reconstructed_module.register_parameter(
                        node.target, 
                        torch.nn.Parameter(torch.randn(1))
                    )
            
            #  Step 6: Create the final `GraphModule` by combining the module (with its parameters) and the graph.
            return fx.GraphModule(reconstructed_module, graph)

        except Exception as e:
            print(f"!!!!! Error creating GraphModule: {e}")
            traceback.print_exc()
            return None

    def _resolve_target(self, op_signature: str) -> Any:
        """(Helper) Converts an operation signature string into a callable object."""
        op_name = op_signature.split(':')[0]
        if op_name.startswith('aten.'): # Handle PyTorch ATen operators.
            parts = op_name.split('.')
            obj = torch.ops.aten
            for part in parts[1:]:
                obj = getattr(obj, part, None)
                if obj is None: return None
            return obj
        elif 'getitem' in op_name: # Handle the special case of getitem.
            return operator.getitem
        return None

    def _get_constants_from_C(self, C: int) -> Dict[str, Any]:
        """(Helper) Retrieves constant values based on the 'C' component ID."""
        if C == 0: return {}
        if C > 0: # C is a direct ID for a single constant.
            const_val = self.index_to_constant.get(C)
            # This is a heuristic to guess the argument name (currently unused).
            if 'linear' in self.registry.index_to_canonical.get(C, ''):
                return {'weight': const_val} if isinstance(const_val, torch.Tensor) else {'bias': const_val}
            return {'arg_const': const_val}
        
        # Negative C indicates it's an ID for a group of constants.
        group_idx = -(C + 1)
        group_str = self.index_to_constant_group.get(group_idx)
        if not group_str: return {}
        
        # Parse the group string (e.g., "key1=10;key2=25") to build a kwargs dictionary.
        constants = {}
        for part in group_str.split(';'):
            try:
                key, val_idx_str = part.split('=')
                val_idx = int(val_idx_str)
                constants[key] = self.index_to_constant.get(val_idx)
            except (ValueError, KeyError):
                print(f"WARNING: Failed to parse constant group part: '{part}'")
        return constants

def run_reconstruction_test(db_path: str):
    """
    Loads compressed signatures, "unrolls" or "expands" them into a flat list of
    fundamental operations, and then reconstructs them into `torch.fx.GraphModule`
    objects to verify the integrity of the entire compression/decompression pipeline.
    """
    print("\n" + "#"*80)
    print("                    PHASE 4: RECONSTRUCTION VERIFICATION")
    print("#"*80)

    # Step 1: Set up file paths and dependencies.
    registry_file = os.path.join(db_path, 'registry.json')
    signatures_file = os.path.join(db_path, 'compressed_signatures.json')

    # Exit early if there are no compressed signatures to test.
    if not os.path.exists(signatures_file):
        print(f"Compressed signatures file not found: {signatures_file}. Test skipped.")
        return

    # Initialize all necessary components: registry, db_manager, decoder, and reconstructor.
    registry = OperationRegistry()
    db_manager = DatabaseManager(db_path)
    db_manager.load_registry(registry)
    
    decoder = Decoder(registry)
    reconstructor = Reconstructor(registry)

    # Load the compressed signatures from the JSON file.
    with open(signatures_file, 'r', encoding='utf-8') as f:
        signatures = json.load(f)

    def expand_graph_recursively(nodes: List[Dict]) -> List[Dict]:
        """
        Takes a list of nodes that may contain high-level macro-nodes (`PATTERN` and `COPY`)
        and iteratively expands them until only fundamental operations remain.
        """
        copy_prefix_id = registry.canonical_to_index.get('<COPY_PREFIX>')
        pattern_prefix_id = registry.canonical_to_index.get('<PATTERN_PREFIX>')

        # Loop as long as there are any macro-nodes left to expand in the graph.
        while any(node.get('A') in (pattern_prefix_id, copy_prefix_id) for node in nodes):
            print(f"  -> Expansion iteration, current graph size: {len(nodes)}")
            
            # PHASE 1: EXPANSION
            # Expand all macros, but keep dependencies as old IDs in a temporary field.
            expanded_nodes_pass1 = []
            id_map = {}  # Maps old ID to one or more new IDs.

            for old_id, node in enumerate(nodes):
                node_A = node.get('A')
                
                if node_A == pattern_prefix_id:
                    pattern_id = node['pattern_id']
                    pattern_b64 = registry.patterns.get(pattern_id)
                    if not pattern_b64: continue
                    
                    pattern_nodes = decoder.decode_raw(pattern_b64, is_pattern_definition=True)
                    base_offset = len(expanded_nodes_pass1)

                    internal_to_external_input_map = {
                        rel_id: node['D'][i]
                        for i, rel_id in enumerate(sorted(list(set(d for pn in pattern_nodes for d in pn.get('D', []) if d < 0))))
                        if i < len(node['D'])
                    }

                    output_source_internal_id = -1
                    for pn in pattern_nodes:
                        if pn.get('A') == registry.canonical_to_index.get('<OUTPUT>'):
                            if pn.get('D'):
                                output_source_internal_id = pn['D'][0]
                                break
                    
                    if output_source_internal_id != -1:
                        # Map old macro ID to the *relative* internal ID of its output node
                        id_map[old_id] = output_source_internal_id + base_offset

                    for internal_id, pn in enumerate(pattern_nodes):
                        if pn.get('A') in (registry.canonical_to_index.get(t) for t in ["<DATA_INPUT>", "<PARAM_INPUT>", "<OUTPUT>"]):
                            continue
                        
                        # Store old dependencies in a temp field
                        old_deps = []
                        for d in pn.get('D', []):
                            if d < 0: # External input
                                if d in internal_to_external_input_map:
                                    old_deps.append(internal_to_external_input_map[d])
                            else: # Internal connection (relative to pattern start)
                                old_deps.append(d + base_offset) # This is an ID relative to the *new* list
                        pn['D_old'] = old_deps
                        pn.pop('D', None)
                        expanded_nodes_pass1.append(pn)

                elif node_A == copy_prefix_id:
                    count = node['count']
                    template = node['template']
                    new_node_base_id = len(expanded_nodes_pass1)
                    
                    # For a COPY node, any node depending on it should probably only get the *first* instance.
                    # This is a simplification but more robust than passing a list.
                    id_map[old_id] = new_node_base_id
                    
                    for i in range(count):
                        new_node = {'A': template['A'], 'B': template['B'], 'C': template['C'], 'D_old': []}
                        expanded_nodes_pass1.append(new_node)
                else:
                    id_map[old_id] = len(expanded_nodes_pass1)
                    node['D_old'] = node.get('D', [])
                    node.pop('D', None)
                    expanded_nodes_pass1.append(node)

            # PHASE 2: RE-WIRING
            # Now that id_map is complete and all nodes are expanded, fix all dependencies.
            for node in expanded_nodes_pass1:
                new_D = []
                old_deps = node.pop('D_old', [])
                for old_d in old_deps:
                    # In this pass, old_d is either an old absolute ID or a new absolute ID (for intra-pattern links)
                    mapped_id = id_map.get(old_d)
                    if mapped_id is not None:
                        new_D.append(mapped_id)
                    else:
                        # If not in map, it might be a new intra-pattern ID already
                        if old_d < len(expanded_nodes_pass1):
                           new_D.append(old_d)

                node['D'] = sorted(list(set(new_D)))
            
            nodes = expanded_nodes_pass1
        return nodes

    # Main test loop 
    # Select the first two signatures from the file for a quick test.
    models_to_reconstruct = list(signatures.keys())[:2]
    
    for model_name in models_to_reconstruct:
        print("\n" + "="*80)
        print(f"Reconstructing signature for: {model_name}")
        print("="*80)
        
        # Step A: Decode the Base64 string into a list of (potentially compressed) nodes.
        b64_string = signatures[model_name]
        compressed_nodes = decoder.decode_raw(b64_string)
        
        # Step B: Expand all PATTERN and COPY macros into fundamental operations.
        flat_nodes = expand_graph_recursively(compressed_nodes)
        
        print(f"Graph '{model_name}' expanded from {len(compressed_nodes)} to {len(flat_nodes)} fundamental operations.")

        # Step C: Pass the flat list of nodes to the Reconstructor to build an `fx.GraphModule`.
        reconstructed_gm = reconstructor.build_module(flat_nodes)

        # Step D: If reconstruction was successful, print the generated Python code for verification.
        if reconstructed_gm:
            print("\n--- Reconstructed torch.fx.GraphModule: \n")
            code_lines = reconstructed_gm.code.split('\n')
            # Truncate long code listings for readability.
            if len(code_lines) > 30:
                print('\n'.join(code_lines[:15]))
                print(f"\n... (and {len(code_lines) - 30} more lines of code) ...\n")
                print('\n'.join(code_lines[-15:]))
            else:
                print(reconstructed_gm.code)

def display_found_patterns(db_path: str):
    """
    Loads the registry from disk and decodes the definitions of the first few
    discovered patterns for human inspection and analysis.
    """
    print("\n" + "#"*80)
    print("                  PHASE 2.5: ANALYZING DISCOVERED PATTERNS")
    print("#"*80)

    # Step 1: Set up dependencies and check for the registry file.
    registry_file = os.path.join(db_path, 'registry.json')
    if not os.path.exists(registry_file):
        print("Registry file not found. Pattern analysis skipped.")
        return

    # Step 2: Load the full knowledge base into an OperationRegistry object.
    registry = OperationRegistry()
    db_manager = DatabaseManager(db_path)
    db_manager.load_registry(registry)

    # Step 3: Check if any patterns have been discovered and saved.
    if not registry.patterns:
        print("No patterns found in the registry.")
        return

    # Step 4: Initialize the special VerificationDecoder for pretty-printing.
    decoder = VerificationDecoder(registry)

    # Step 5: Iterate through the patterns, sorted by their ID for consistent output.
    for pattern_id, b64_string in sorted(registry.patterns.items()):
        # For readability, only decode and print the first few patterns.
        # Here, we arbitrarily choose to show patterns with ID < 15 (since IDs start at 10).
        if pattern_id >= 15:
            print("\n" + "-"*80)
            print(f"... and {len(registry.patterns) - (15 - registry.RESERVED_RANGE)} more patterns.")
            break
        
        # Step 6: Call the decoder to print a formatted representation of the pattern.
        # IMPORTANT NOTE: Inside a pattern definition, the 'D' field contains RELATIVE offsets,
        # not absolute IDs. The VerificationDecoder, designed for general graphs, will print these
        # negative numbers as-is, which is the correct representation for a pattern's internal structure.
        decoder.decode_and_print(f"PATTERN ID={pattern_id}", b64_string)

def run_verification_test(db_path: str):
    """
    Loads the final compressed signatures and decodes a few of them using the
    VerificationDecoder to provide a human-readable check on the final compression results.
    """
    print("\n" + "#"*80)
    print("                      PHASE 3: COMPRESSION VERIFICATION")
    print("#"*80)

    # Step 1: Set up file paths and check for the signatures file.
    registry_file = os.path.join(db_path, 'registry.json')
    signatures_file = os.path.join(db_path, 'compressed_signatures.json')

    if not os.path.exists(signatures_file):
        print(f"Compressed signatures file not found: {signatures_file}. Test skipped.")
        return

    # Step 2: Load the registry. The decoder needs it to map integer IDs back to names.
    registry = OperationRegistry()
    db_manager = DatabaseManager(db_path)
    db_manager.load_registry(registry)
    
    # Step 3: Initialize the VerificationDecoder.
    decoder = VerificationDecoder(registry)

    # Step 4: Load the dictionary of compressed signatures.
    with open(signatures_file, 'r', encoding='utf-8') as f:
        signatures = json.load(f)

    # Step 5: Decode and print the first 3 signatures from the file for a quick spot-check.
    count = 0
    for name, b64_string in signatures.items():
        if count >= 3:
            break
        # Call the decoder. It will parse the Base64 string and print a formatted
        # representation of the compressed graph, including any `PATTERN` or `COPY` macro-nodes.
        decoder.decode_and_print(name, b64_string)
        count += 1

import torchvision.models as models

def main():
    # 1. Setting up---
    cache_dir = os.path.abspath(CACHE_DIR_NAME)
    
    # 2. Definition of extractors 
    def get_hf_bert_encoder(model): return getattr(model, 'bert', None)
    def get_hf_roberta_encoder(model): return getattr(model, 'roberta', None)
    def get_hf_esm_encoder(model): return getattr(model, 'esm', None)
    def get_hf_transformer_encoder(model): return getattr(model, 'transformer', None)
    def get_hf_distilbert_encoder(model): return getattr(model, 'distilbert', None)
    def get_hf_classifier_head(model): return getattr(model, 'classifier', None)
    def get_hf_lm_head(model: nn.Module) -> Optional[nn.Module]:
        if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'decoder'):
            print("  -> Found RobertaLMHead, extracting only 'decoder' (linear layer).")
            return model.lm_head.decoder 
        for attr in ['lm_head', 'cls', 'output_projection']:
            if hasattr(model, attr):
                print(f"  -> LM head found: '{attr}'")
                return getattr(model, attr)
        return None
    
    # Extractors for ResNet 
    def get_resnet_features(model):
        # Returns everything except the last layer (fc)
        return nn.Sequential(*list(model.children())[:-1])
    def get_resnet_fc(model):
        return getattr(model, 'fc', None)

    # Combining extractors into one dictionary
    extractor_map = {
        "bert": get_hf_bert_encoder, "roberta": get_hf_roberta_encoder, 
        "esm": get_hf_esm_encoder, "distilbert": get_hf_distilbert_encoder,
        "transformer": get_hf_transformer_encoder,
        "classifier": get_hf_classifier_head, "lm_head": get_hf_lm_head,
        "cls": get_hf_lm_head,
        "resnet_features": get_resnet_features,
        "resnet_fc": get_resnet_fc,
    }

    # 3. Reliable model loading feature (with modifications for TorchVision) 
    def _robust_load_model(task_name: str, expert_config: dict):
        # Adding a branch for models not from Hugging Face 
        if expert_config.get("source") == "torchvision":
            print("\n" + "="*50 + f"\nLoading: {task_name} (from torchvision)\n" + "="*50)
            # Load the model using the function specified in 'task_class'
            model = expert_config["task_class"](weights=None, num_classes=10) # num_classes=10 для CIFAR-10
            print("  -> The model has been loaded successfully.")
            return model.eval()

        # Old logic for hugging face models
        print("\n" + "="*50 + f"\nLoading: {task_name} ({expert_config['model_id']})\n" + "="*50)
        model_id = expert_config["model_id"]
        local_model_path = snapshot_download(repo_id=model_id, cache_dir=cache_dir, local_files_only=True)
        safetensors_path = os.path.join(local_model_path, "model.safetensors")
        bin_path = os.path.join(local_model_path, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            print(f"  -> Found 'model.safetensors'. Loading via 'from_pretrained'.")
            model = expert_config["task_class"].from_pretrained(local_model_path, local_files_only=True, trust_remote_code=True)
        elif os.path.exists(bin_path):
            print(f"  -> Found 'pytorch_model.bin'. Manual download.")
            model_config_hf = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
            model = expert_config["task_class"].from_config(model_config_hf)
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"No weights ('model.safetensors' or 'pytorch_model.bin') found in {local_model_path}")
        print("  -> The model has been loaded successfully.")
        return model.eval()

    # 4. List of models to be processed 
    all_expert_tasks = {
        # NLP models
        "biobert": {"model_id": "dmis-lab/biobert-v1.1", "task_class": AutoModelForSequenceClassification, "encoder_extractor": "bert", "head_extractor": "classifier"},
        "pubmed_bert": {"model_id": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", "task_class": AutoModelForMaskedLM, "encoder_extractor": "bert", "head_extractor": "cls"},
        "clinicalbert": {"model_id": "emilyalsentzer/Bio_ClinicalBERT", "task_class": AutoModelForSequenceClassification, "encoder_extractor": "bert", "head_extractor": "classifier"},
        "chemberta": {"model_id": "DeepChem/ChemBERTa-77M-MTR", "task_class": AutoModelForMaskedLM, "encoder_extractor": "roberta", "head_extractor": "lm_head"},
        "esm2": {"model_id": "facebook/esm2_t30_150M_UR50D", "task_class": AutoModelForMaskedLM, "encoder_extractor": "esm", "head_extractor": "lm_head"},
        "biobert_ner": {"model_id": "alvaroalon2/biobert_diseases_ner", "task_class": AutoModelForTokenClassification, "encoder_extractor": "bert", "head_extractor": "classifier"},
        "scibert": {"model_id": "allenai/scibert_scivocab_uncased", "task_class": AutoModelForSequenceClassification, "encoder_extractor": "bert", "head_extractor": "classifier"},
        "gatortron": {"model_id": "ufnlp/gatortron-base", "task_class": AutoModelForMaskedLM, "encoder_extractor": "bert", "head_extractor": "cls"},
        "biogpt": {"model_id": "microsoft/biogpt", "task_class": AutoModelForCausalLM, "encoder_extractor": "transformer", "head_extractor": "lm_head"},
        "biomed_ner_distilbert": {"model_id": "d4data/biomedical-ner-all", "task_class": AutoModelForTokenClassification, "encoder_extractor": "distilbert", "head_extractor": "classifier"},
        "biomed_roberta": {"model_id": "allenai/biomed_roberta_base", "task_class": AutoModelForSequenceClassification, "encoder_extractor": "roberta", "head_extractor": "classifier"},
        "radbert": {"model_id": "zzxslp/RadBERT-RoBERTa-4m", "task_class": AutoModelForSequenceClassification, "encoder_extractor": "roberta", "head_extractor": "classifier"},
        "s_biobert": {"model_id": "pritamdeka/S-BioBert-snli-multinli-stsb", "task_class": AutoModelForSequenceClassification, "encoder_extractor": "bert", "head_extractor": "classifier"},
        "gpt_neo_125m": {"model_id": "EleutherAI/gpt-neo-125M", "task_class": AutoModelForCausalLM, "encoder_extractor": "transformer", "head_extractor": "lm_head"},
        "prot_bert": {"model_id": "Rostlab/prot_bert", "task_class": AutoModelForMaskedLM, "encoder_extractor": "bert", "head_extractor": "cls"},
        
        # CV model
        "resnet18_cifar10": {
            "source": "torchvision", 
            "task_class": models.resnet18, 
            "encoder_extractor": "resnet_features", 
            "head_extractor": "resnet_fc"
        },
    }

    # 5. Main conveyor 
    
    # Initialization
    db_manager = DatabaseManager()
    registry = OperationRegistry()
    db_manager.load_registry(registry)
    # We pass the updated extractor_map
    decomposer = Decomposer(registry, extractor_map)
    
    all_flat_graphs: Dict[str, 'GraphRepresentation'] = {}
    existing_signatures = {}
    signatures_filepath = os.path.join(db_manager.db_path, 'compressed_signatures.json')
    if os.path.exists(signatures_filepath):
        try:
            with open(signatures_filepath, 'r', encoding='utf-8') as f:
                existing_signatures = json.load(f)
            print(f"{len(existing_signatures)} existing signatures found. Only new models will be processed.")
        except json.JSONDecodeError:
            print("The compressed_signatures.json file is corrupted. All models will be reprocessed.")
            
    # Phase 1: Decomposition
    print("\n" + "#"*80)
    print("                      PHASE 1: MODEL DECOMPOSITION")
    print("#"*80)
    models_to_process = False
    for name, meta in all_expert_tasks.items():
        already_processed = any(key.startswith(f"{name}_") for key in existing_signatures.keys())

        if already_processed:
            print(f"\n--- Model '{name}' has already been processed. Skipping. ")
            continue

        models_to_process = True
        try:
            model = _robust_load_model(name, meta)
            if model:
                # Passing additional information about the model type to the decomposer
                # This is needed so that _get_dummy_inputs knows which inputs to create.
                run_meta = meta.copy()
                if meta.get("source") == "torchvision":
                    run_meta["model_type"] = "vision"
                
                model_graphs = decomposer.run(model, name, run_meta)
                for part_name, graph_repr in model_graphs.items():
                    all_flat_graphs[f"{name}_{part_name}"] = graph_repr
        except Exception as e:
            print(f"!!!!! CRITICAL ERROR while processing {name}: {e}")
            traceback.print_exc()
            
    # Phase 2: Analysis and Compression
    if all_flat_graphs:
        print("\nSaving a complete register of operations and constants before analysis...")
        db_manager.save_registry(registry)
        analyzer = Analyzer(registry, db_manager)
        analyzer.run(all_flat_graphs)
    elif not models_to_process:
        print("\nAll models have already been processed. There are no new models for analysis.")
    else:
        print("\nUnable to create graphs for analysis. Phase 2 skipped.")

    # Launching a verification test
    display_found_patterns(db_manager.db_path)
    run_verification_test(db_manager.db_path)
    run_reconstruction_test(db_manager.db_path)
    print("The work has been completed successfully.")

if __name__ == "__main__":
    main()