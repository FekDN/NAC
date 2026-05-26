# NAC Inspection Tools

Three utilities for examining compiled `.nac` model files.

---

## NAC_info.py — File Inspector

Prints the complete structure of a `.nac` file: header fields, all sections, parameter list, tensor shapes, embedded resources, and tokenizer manifest (if present).

```bash
python NAC_info.py <model.nac>
```

**Output includes:**
- Format version, quantization method, weight storage location
- I/O counts, `d_model`
- Section offsets table (MMAP, OPS, CMAP, CNST, PERM, DATA, PROC, ORCH, RSRC)
- All parameter names and IDs
- Internally stored tensor shapes, dtypes, and sizes
- Embedded resource files (vocab, merges, etc.)
- Disassembled TISA tokenizer manifest (requires `TISA_tokenizer.py`)

**Dependencies:** `NAC_kernels.py` (optional), `TISA_tokenizer.py` (optional, for PROC section)

---

## NAC_reconstructor.py — Decompiler

Reconstructs the computation graph from a `.nac` file into a human-readable pseudo-code listing. Each instruction is shown as a named variable assignment, with full argument resolution and optional memory map annotations.

```bash
python NAC_reconstructor.py <model.nac>
python NAC_reconstructor.py -m <model.nac>     # include MMAP schedule annotations
python NAC_reconstructor.py --mmap <model.nac>
```

**Output format:**
```
v0  = user_input(name='input_ids')
v1  = load_param(name='embeddings.weight', shape=[30522, 768], dtype=float16)
v2  = aten.embedding(v1, v0, 0, False, False)
v3  = aten.layer_norm(v2, [768], v4, v5, 1e-12)
...
return v147
```

With `-m`, each line gains a MMAP annotation:
```
v43 = aten.relu(v42)   FREE -> 26, SAVE_RESULT -> 43, PRELOAD -> 44
```

**Dependencies:** `NAC_kernels.py` (optional)

---

## NAC_weights.py — Weights Manager

A universal utility for extracting and injecting model weights between `.nac` containers and standard `.safetensors` files. It dynamically updates the binary geometry of the `.nac` file in-place and fully preserves quantization metadata.

```bash
# Extract internal weights into a single .safetensors file
python NAC_weights.py extract <model.nac>

# Extract and split weights into multiple shards (e.g., for GitHub upload limits)
python NAC_weights.py extract <model.nac> --shards 3

# Inject external .safetensors (auto-discovered) back into the .nac file
python NAC_weights.py inject <model.nac>
```

**Key Features:**
- **In-place Optimization:** Extracts weights to make the `.nac` file lightweight for inference, recalculating the binary section offsets automatically.
- **Auto-Discovery & Sharding:** The `inject` command automatically finds and merges `model.safetensors` or `model-00001-of-00003.safetensors` shards. External files are safely cleaned up after successful injection.
- **Quantization Awareness:** Fully preserves custom NAC quantization formats (`INT8_TENSOR`, `INT8_CHANNEL`, `BLOCK_FP8`) by storing native `int8` tensors alongside serialized JSON metadata (`nac_quant_meta`) inside the `.safetensors` header.
- **Strict Validation:** Validates tensor shapes during injection against the expected NAC geometry to prevent `reshape` errors and guarantee runtime stability.

**Dependencies:** `torch`, `safetensors` (`pip install torch safetensors`)

---

## NAC_graph.py — Interactive Graph Viewer

Launches a local web server and opens an interactive computation graph visualization in the browser. Nodes are laid out in execution order (left → right). Weight tensors cluster above their consuming op; scalar constants cluster below.

```bash
python NAC_graph.py <model.nac>
python NAC_graph.py <model.nac> --port 8080
python NAC_graph.py <model.nac> --no-browser   # start server without opening browser
```

Default URL: **http://127.0.0.1:5050**

**Node types:**

| Shape | Colour | Meaning |
|-------|--------|---------|
| Pentagon `▶` | green | User input |
| Pentagon `▶` | cyan | Model output |
| Rectangle | orange | Weight tensor (param) |
| Diamond | grey-blue | Scalar constant |
| Circle | purple / pink / teal / … | Operation (colour = op group) |
| Dashed rect | red | Control flow |

**Controls:**

| Control | Action |
|---------|--------|
| Scroll | Zoom in / out |
| Drag background | Pan |
| Drag node | Move node permanently |
| Click node | Highlight node and its direct connections; click again to reset |
| `weights` button | Show / hide weight tensor nodes and their edges |
| `consts` button | Show / hide scalar constant nodes and their edges |
| `fit` button | Zoom to fit all visible nodes |
| `reset` button | Restore all nodes to computed layout positions |
| Search box | Dim all nodes whose name does not match the query |

**Dependencies:** `flask` (`pip install flask`), `NAC_kernels.py` (optional)

---

## Common Notes

All three tools work on any `.nac` file regardless of whether weights are stored internally or in an external `.safetensors` file. `NAC_kernels.py` is optional but recommended — without it, standard NAC operation IDs will not be resolved to their canonical names.
