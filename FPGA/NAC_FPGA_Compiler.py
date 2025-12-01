import os
import json
import struct
import torch
import shutil
from typing import Dict, List, Any, Tuple, Optional

# Import NAC classes (assuming NAC.py is present)
try:
    from NAC import OperationRegistry, GraphRepresentation, Decoder
except ImportError:
    print("CRITICAL: NAC.py not found.")
    exit(1)

# ==============================================================================
# HARDWARE CONSTANTS & MAPPINGS
# ==============================================================================
# These definitions must be kept in sync with nac_hw_defines.vh
HW_OPCODES = {
    "NOP": 0, "ADD": 1, "SUB": 2, "MUL": 3, "DIV": 4, "NEG": 5, "POW": 6,
    "SIN": 7, "COS": 8, "ERF": 9, "RELU": 10, "GELU": 11, "TANH": 12, "SOFTMAX": 13,
    "LINEAR": 14, "MATMUL": 15, "SUM": 16, "MEAN": 17, "LAYER_NORM": 18,
    "EQ": 19, "NE": 20, "GT": 21, "LT": 22, "MASKED_FILL": 23,
    "ARANGE": 24, "FULL": 25, "TRIU": 26, "COPY": 27, "EMBED": 28, "CONV2D": 29,
    "BATCH_NORM": 30, "MAX_POOL2D": 31, "ADAPTIVE_AVG_POOL2D": 32,
    "VIEW": 40, "TRANSPOSE": 41, "PERMUTE": 42, "UNSQUEEZE": 43,
    "CAT": 44, "SPLIT": 45
}

CANONICAL_TO_HW_MAP = {
    # Math and Compute Opcodes
    "add": HW_OPCODES["ADD"], "sub": HW_OPCODES["SUB"], "mul": HW_OPCODES["MUL"],
    "div": HW_OPCODES["DIV"], "neg": HW_OPCODES["NEG"], "pow": HW_OPCODES["POW"],
    "relu": HW_OPCODES["RELU"], "gelu": HW_OPCODES["GELU"], "tanh": HW_OPCODES["TANH"],
    "softmax": HW_OPCODES["SOFTMAX"], "sin": HW_OPCODES["SIN"], "cos": HW_OPCODES["COS"],
    "erf": HW_OPCODES["ERF"], "linear": HW_OPCODES["LINEAR"], "matmul": HW_OPCODES["MATMUL"],
    "bmm": HW_OPCODES["MATMUL"], "sum": HW_OPCODES["SUM"], "mean": HW_OPCODES["MEAN"],
    "layer_norm": HW_OPCODES["LAYER_NORM"], "batch_norm": HW_OPCODES["BATCH_NORM"],
    "conv2d": HW_OPCODES["CONV2D"], "max_pool2d": HW_OPCODES["MAX_POOL2D"],
    "adaptive_avg_pool2d": HW_OPCODES["ADAPTIVE_AVG_POOL2D"], "embedding": HW_OPCODES["EMBED"],
    # Logic and Generator Opcodes
    "eq": HW_OPCODES["EQ"], "ne": HW_OPCODES["NE"], "gt": HW_OPCODES["GT"],
    "lt": HW_OPCODES["LT"], "masked_fill": HW_OPCODES["MASKED_FILL"],
    "arange": HW_OPCODES["ARANGE"], "full": HW_OPCODES["FULL"], "triu": HW_OPCODES["TRIU"],
    # Data Movement Opcodes (map to COPY or NOP)
    "get_attr": HW_OPCODES["COPY"], "dropout": HW_OPCODES["COPY"],
    "<CONST_REF>": HW_OPCODES["NOP"],
    # Metadata Pseudo-Opcodes
    "view": HW_OPCODES["VIEW"], "transpose": HW_OPCODES["TRANSPOSE"],
    "permute": HW_OPCODES["PERMUTE"], "unsqueeze": HW_OPCODES["UNSQUEEZE"],
    "cat": HW_OPCODES["CAT"], "split": HW_OPCODES["SPLIT"]
}

STREAMING_OPS = {
    HW_OPCODES["LINEAR"], HW_OPCODES["MATMUL"], HW_OPCODES["EMBED"],
    HW_OPCODES["CONV2D"], HW_OPCODES["BATCH_NORM"], HW_OPCODES["LAYER_NORM"]
}

# ==============================================================================
# SHARED UTILS
# ==============================================================================
def float_to_q16_16(val: float) -> int:
    """Converts a float to a 32-bit Q16.16 fixed-point integer."""
    scaled = int(val * 65536.0)
    limit = 2147483647
    return max(min(scaled, limit), -limit - 1) & 0xFFFFFFFF

def get_hw_opcode_and_flags(canonical_name: str) -> Tuple[int, int]:
    """Finds the best hardware opcode for a given canonical name."""
    name_lower = canonical_name.lower()
    opcode = HW_OPCODES["NOP"]
    best_match_key = ""
    # Find the longest (most specific) matching key
    for key in CANONICAL_TO_HW_MAP.keys():
        if key in name_lower:
            if len(key) > len(best_match_key):
                best_match_key = key
    
    if best_match_key:
        opcode = CANONICAL_TO_HW_MAP[best_match_key]

    # Flags are stored in the upper bits. Bit 6 = Streaming Op
    flags = (1 << 6) if opcode in STREAMING_OPS else 0
    return opcode, flags

def serialize_nodes(nodes: List[Dict], code_buffer: bytearray, const_map: Dict[int, int] = None):
    """Serializes a list of NAC nodes into a binary stream for the hardware."""
    for node in nodes:
        A = node['A']
        
        # 1. Pattern Call (A=6)
        if 'pattern_id' in node:
            pid = node['pattern_id']
            # Header format: A (1B), E0 (1B), E1+padding (2B), D_len (1B)
            header = struct.pack('<BBHB', A, pid & 0xFF, (pid >> 8) & 0xFF, len(node.get('D', [])))
            code_buffer.extend(header)
            for inp in node.get('D', []):
                code_buffer.extend(struct.pack('>H', inp)) # Big Endian Unsigned Short
        
        # 2. Copy / RLE (A=7)
        elif 'template' in node:
            tpl = node['template']
            count = node['count']
            B_field = tpl['A']
            
            tpl_c = tpl['C']
            if const_map and tpl_c > 0 and tpl_c in const_map:
                tpl_c = const_map[tpl_c]
            
            # Header format: A(1B), Template_A(1B), Count(2B, signed), D_len(1B=2)
            header = struct.pack('<BBhB', A, B_field, count, 2)
            code_buffer.extend(header)
            # Payload D = [Template_B, Template_C], both signed shorts
            code_buffer.extend(struct.pack('>h', tpl['B']))
            code_buffer.extend(struct.pack('>h', tpl_c))

        # 3. Standard Operation
        else:
            C = node.get('C', 0)
            # Remap original constant ID to the new hardware slot ID
            if const_map and C > 0 and C in const_map:
                C = const_map[C]
                
            inputs = node.get('D', [])
            # Header format: A(1B), B(1B), C(2B, signed), D_len(1B)
            header = struct.pack('<BBhB', A, node.get('B', 0), C, len(inputs))
            code_buffer.extend(header)
            for inp in inputs:
                code_buffer.extend(struct.pack('>H', inp)) # Unsigned Short

# ==============================================================================
# CLASS 1: SYSTEM COMPILER (Universal, Run Once)
# ==============================================================================
class SystemCompiler:
    """Compiles the universal system firmware (LUTs, patterns)."""
    def __init__(self, registry: OperationRegistry, output_dir: str, system_ddr_base: int):
        self.registry = registry
        self.out_dir = output_dir
        self.base_addr = system_ddr_base
        
        self.OFF_OPMAP = 0x0000
        self.OFF_VARMAP = 0x1000
        self.OFF_REGISTRY = 0x2000
        self.OFF_PATTERNS = 0x4000
        
        if os.path.exists(self.out_dir): shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir)

    def compile(self):
        print(f"[SystemCompiler] Building NAC System Firmware (Base: 0x{self.base_addr:08X})...")
        self._compile_luts()
        self._compile_patterns()
        self._gen_header()
        print(f"[SystemCompiler] Done. System binaries in '{self.out_dir}'")

    def _compile_luts(self):
        # OpMap (256 entries of 1 byte)
        op_data = bytearray()
        for i in range(256):
            if i in self.registry.index_to_canonical:
                name = self.registry.index_to_canonical[i]
                op, fl = get_hw_opcode_and_flags(name)
                val = (fl & 0xC0) | (op & 0x3F) # Flags in bits 7:6, opcode in 5:0
            else:
                val = 0
            op_data.append(val)
        with open(os.path.join(self.out_dir, 'sys_opmap.bin'), 'wb') as f: f.write(op_data)
            
        # VarMap (256 entries of 1 byte)
        var_data = bytearray()
        for i in range(256):
            val = 0
            var_str = next((k for k, v in self.registry.variation_to_index.items() if v == i), "")
            s1, s2 = 0, 1
            if var_str:
                parts = var_str.split(',')
                try: 
                    if len(parts) > 0: s1 = int(parts[0].replace('in', ''))
                    if len(parts) > 1: s2 = int(parts[1].replace('in', ''))
                except (ValueError, IndexError):
                    pass
            val = ((s2 & 0xF) << 4) | (s1 & 0xF)
            var_data.append(val)
        with open(os.path.join(self.out_dir, 'sys_varmap.bin'), 'wb') as f: f.write(var_data)

    def _compile_patterns(self):
        decoder = Decoder(self.registry)
        code_stream = bytearray()
        jump_table = [0] * 256
        
        for pid in sorted(self.registry.patterns.keys()):
            if pid >= 256: continue
            
            abs_addr = self.base_addr + self.OFF_PATTERNS + len(code_stream)
            jump_table[pid] = abs_addr
            
            raw = decoder.decode_raw(self.registry.patterns[pid], is_pattern_definition=True)
            serialize_nodes(raw, code_buffer=code_stream, const_map=None)
            
            # Append RETURN instruction (A=0, all other fields zero)
            code_stream.extend(b'\x00\x00\x00\x00\x00')
            
        with open(os.path.join(self.out_dir, 'sys_patterns.bin'), 'wb') as f: f.write(code_stream)
        
        reg_data = bytearray()
        for addr in jump_table:
            reg_data.extend(struct.pack('<I', addr))
        with open(os.path.join(self.out_dir, 'sys_registry.bin'), 'wb') as f: f.write(reg_data)

    def _gen_header(self):
        with open(os.path.join(self.out_dir, 'nac_system_info.txt'), 'w') as f:
            f.write(f"SYSTEM_BASE_ADDR=0x{self.base_addr:08X}\n")

# ==============================================================================
# CLASS 2: MODEL COMPILER (Runs Per Model)
# ==============================================================================
class ModelCompiler:
    """Compiles a specific model graph into a loadable binary package."""
    def __init__(self, registry: OperationRegistry, output_dir: str):
        self.registry = registry
        self.out_dir = output_dir
        self.weights_blob = bytearray()
        self.const_map = {} # Maps original const ID -> new hardware slot ID
        self.next_slot = 0
        self.ALIGNMENT = 2048 # Hardware memory alignment (e.g., for DMA)
        
        if os.path.exists(self.out_dir): shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir)

    def compile(self, graph: GraphRepresentation, dummy_input: Optional[torch.Tensor]):
        print(f"[ModelCompiler] Building Model Package: {graph.name}...")
        self._compile_weights()
        self._compile_code(graph)
        self._compile_input(dummy_input)
        print(f"[ModelCompiler] Done. Model binaries in '{self.out_dir}'")

    def _compile_weights(self):
        """Compiles all constants (tensors, scalars, lists) into a single aligned binary blob."""
        all_ids = sorted(self.registry.index_to_constant.keys())

        for cid in all_ids:
            val = self.registry.index_to_constant[cid]
            blob = None
            
            if isinstance(val, (int, float)):
                blob = struct.pack('<i', float_to_q16_16(float(val)))
            elif isinstance(val, torch.Tensor):
                t_max = val.abs().max()
                scale = 127.0 / t_max if t_max > 0 else 1.0
                t_int8 = (val * scale).round().clamp(-128, 127).to(torch.int8)
                blob = t_int8.numpy().tobytes()
            elif isinstance(val, (list, tuple)):
                try:
                    # Pack lists/tuples of integers (for view, permute, etc.)
                    blob = struct.pack('<' + 'i'*len(val), *[int(v) for v in val])
                except (ValueError, TypeError):
                    continue # Skip if not a list of integers
            
            if blob:
                # Align the start of the current data block
                offset = len(self.weights_blob)
                if offset % self.ALIGNMENT != 0:
                    pad = self.ALIGNMENT - (offset % self.ALIGNMENT)
                    self.weights_blob.extend(b'\x00' * pad)

                # The hardware slot ID is the aligned address divided by alignment size
                slot = len(self.weights_blob) // self.ALIGNMENT
                self.const_map[cid] = slot
                self.weights_blob.extend(blob)
        
        # Final padding for the entire blob
        if len(self.weights_blob) % self.ALIGNMENT != 0:
             pad = self.ALIGNMENT - (len(self.weights_blob) % self.ALIGNMENT)
             self.weights_blob.extend(b'\x00' * pad)

        with open(os.path.join(self.out_dir, 'model_weights.bin'), 'wb') as f:
            f.write(self.weights_blob)

    def _compile_code(self, graph):
        decoder = Decoder(self.registry)
        raw_nodes = decoder.decode_raw(graph.to_base64())
        code_stream = bytearray()
        
        # Serialize, replacing abstract const IDs with concrete slot IDs
        serialize_nodes(raw_nodes, code_stream, self.const_map)
        
        # Append an infinite loop (JMP to self) or a STOP instruction at the end
        code_stream.extend(b'\x00\x00\x00\x00\x00') 
        
        with open(os.path.join(self.out_dir, 'model_code.bin'), 'wb') as f:
            f.write(code_stream)

    def _compile_input(self, tensor):
        if tensor is None:
            blob = b'\x00' * self.ALIGNMENT
        else:
            if tensor.dtype == torch.float32:
                # Quantize float inputs (e.g., images) to UINT8
                blob = tensor.mul(255).round().clamp(0, 255).to(torch.uint8).numpy().tobytes()
            else: # For NLP, inputs are typically int32/int64
                blob = tensor.flatten().to(torch.int32).numpy().tobytes()

            if len(blob) % self.ALIGNMENT != 0:
                pad = self.ALIGNMENT - (len(blob) % self.ALIGNMENT)
                blob += b'\x00' * pad
            
        with open(os.path.join(self.out_dir, 'model_input.bin'), 'wb') as f:
            f.write(blob)

# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def run_full_compilation(registry: OperationRegistry, 
                         graphs: Dict[str, GraphRepresentation],
                         sys_base_addr: int = 0x10000000):
    
    # 1. System Compilation (Once)
    sys_compiler = SystemCompiler(registry, "./nac_dist/system", sys_base_addr)
    sys_compiler.compile()
    
    # 2. Model Compilation (For each graph)
    for name, graph in graphs.items():
        dummy = None
        if "resnet" in name:
            dummy = torch.randn(1, 3, 32, 32)
        elif any(s in name for s in ["bert", "roberta", "gpt"]):
            dummy = torch.randint(0, 30000, (1, 64), dtype=torch.long)
        
        model_dir = os.path.join("./nac_dist", name.replace("/", "_"))
        mod_compiler = ModelCompiler(registry, model_dir)
        mod_compiler.compile(graph, dummy)
        
    print("\n[INFO] Distribution ready in ./nac_dist/")
    print("       - /system/ : Universal NAC Firmware (Load once)")
    print("       - /<model_name>/: Model-specific cartridges (Load dynamically)")