# NAC: The Universal Genome and ISA for Artificial Intelligence

NAC is a universal Instruction Set Architecture (ISA) and compiler for AI, representing neural networks as a standardized 'genome' for deep analysis, optimization, and hardware synthesis.

---

## The Problem: AI Models are Black Boxes

Modern AI models are complex computational graphs, tightly coupled to their implementation frameworks (PyTorch, TensorFlow) and hardware. Each model is a "black box" where architecture and parameters are intertwined, making it impossible to transfer knowledge between architectures, compare them objectively, and develop universal AI systems.

## The Solution: A New Layer of Abstraction

The NAC project introduces a fundamentally new abstraction layer: a unified binary format and instruction set where any neural network can be represented as standardized, executable code. NAC is a **universal machine language for AI**, analogous to a processor's ISA, but designed for the semantics of neural computation graphs.

This decouples the model's architectural "body" from its parametric "mind," enabling three powerful paradigms:

### 1. NAC as an Instruction Set Architecture (ISA)
*   **Hardware Synthesis:** NAC's formalized binary language can serve as an ISA specification for next-generation AI accelerators (ASICs, FPGAs). Each fundamental operation (`A >= 10`) is a potential hardware instruction.
*   **Adaptive Computing:** This enables a new class of universal AI processors where any model runs as a `ABCD[]` microprogram, reconfiguring hardware blocks on the fly. This combines the speed of ASICs with the flexibility of GPUs.

### 2. NAC as a Neural Genome
*   **Comparative Genomics for AI:** We can create a "Phylogenetic Tree" of AI architectures by treating NAC signatures as genomes and recurring patterns as "genes." This allows us to map the entire AI ecosystem, discovering universal principles and tracking the "evolution" of models (e.g., from ResNet to ConvNeXt).
*   **Discovering Foundational Building Blocks:** The system automatically mines a library of fundamental patterns (e.g., ConvBlock, ResidualBlock, TransformerLayer) from a vast corpus of models, creating a universal vocabulary of AI design.

### 3. NAC for Generative AI & Knowledge Transfer
*   **Generative Architecture Synthesis (NAS):** By learning the statistical "grammar" of AI design from the knowledge base, NAC enables a generative synthesizer that builds new, statistically sound architectures not by random search, but through informed construction.
*   **Knowledge Transplantation:** By defining a canonical mapping between operations, NAC makes it possible to transfer trained weights ("skills") from a block in one model to a compatible block in another, drastically reducing retraining costs.

---

## Instructions for use in the `/test/` directory

---

## Neural Architecture Code (NAC) v1.6 Specification

The NAC format is designed for the compact, unified, and machine-readable
representation of neural network computation graphs. Each graph is a sequence
of nodes, where each node is described by a variable-length command.


## 1. Introduction and Philosophy of the NAC Standard

### 1.1. Purpose of the Standard (Target Platform: FPGA/ASIC)

The NAC standard is designed not as a format for general-purpose processors (CPU/GPU), but as a **specification for implementation on specialized hardware**, such as Field-Programmable Gate Arrays (FPGA) and Application-Specific Integrated Circuits (ASIC). The Python implementation provided within the ecosystem serves as a functional reference interpreter and a development tool, but it is not the target platform for achieving maximum performance. The ultimate objective is to create a hardware core capable of natively executing NAC instructions, achieving maximum efficiency and power savings.

### 1.2. Overall Ecosystem Architecture (Compiler → Format → Runtime)

The NAC ecosystem is divided into three logical components, which allows for the isolation of complexity and ensures stability:

1.  **Compiler (`NAC.py` + `NAC_optimizer.py`):** This is the system's front-end, responsible for converting high-level models from PyTorch into the low-level NAC format. The compilation process involves several stages:
    *   **Export:** Using `torch.export` to obtain a "raw" computation graph. This stage is the most unstable part of the pipeline, as `torch.export` does not provide a graph ready for direct execution or training, and its output can vary between PyTorch versions.
    *   **Cleaning and Optimization (`NAC_optimizer.py`):** The raw graph undergoes a series of transformations: constant folding, common subexpression elimination (CSE), algebraic simplifications, removal of paired inverse operations, and elimination of identity operations (like `dropout` in `eval` mode). This step is critically important for obtaining a clean, efficient, and deterministic graph.
    *   **Canonization:** The optimized graph is translated into the NAC canonical form, where the multitude of PyTorch ATen operations are reduced to a small, strictly defined set of NAC instructions.

2.  **Format (`.nac`):** This is the standardized binary representation of the canonized graph. It is designed as a self-contained artifact that includes everything necessary for model execution: instructions, metadata, weights, and resources (e.g., for a tokenizer).

3.  **Executor/Runtime (`NAC_run.py` / FPGA / ASIC):** This is the system's back-end. Its task is to interpret the sequence of instructions from a `.nac` file. Because all the complexity and variability of the original model were eliminated during compilation, the runtime deals with a simple and stable set of instructions, making it ideal for hardware implementation.

### 1.3. Key Design Principles

The NAC standard is built on three fundamental principles:

*   **Compactness:** The format uses numerical IDs, relative offsets, and signature reuse to minimize redundancy. This makes `.nac` files small and reduces memory requirements during execution, which is critical for embedded systems and hardware implementations.
*   **Canonization:** Instead of supporting hundreds of operations from deep learning frameworks, NAC reduces them to a small, orthogonal, and stable set of canonical instructions. This significantly simplifies the development and verification of the runtime (especially hardware) and makes the standard resilient to changes in upstream libraries.
*   **Hardware Orientation:** The entire design, from the simple, sequential ABCD instruction structure to the static nature of the graph, is aimed at direct and efficient implementation in digital logic. The absence of dynamic elements, such as data-dependent control flow (in the current version), makes the graph easily parallelizable and pipelineable at the hardware level.

### 1.4. Versioning (Current Version: 1.6)

The current version of the standard is **NAC v1.6**. The version is explicitly stated in the header of the `.nac` file, which ensures backward compatibility and allows for future extensions. Any changes to the header structure, instruction format, or the semantics of system operations will require an increment of the standard's version.


## 2. Structure of the Binary File (.nac)

### 2.1. Overall File Layout (Header and Sections)

A `.nac` format file is a binary container with a sequential structure. It begins with a fixed 88-byte header, followed by several sections of variable-length data. The header contains a "table of contents"—absolute offsets from the beginning of the file to the start of each section. This allows the runtime to quickly locate the necessary data without having to scan the entire file sequentially.

### 2.2. File Header (88 bytes)

The header has a fixed length and contains essential metadata about the model and the file's structure. All numerical values are stored in Little Endian format.

#### 2.2.1. Magic Bytes and Version (4 bytes)
*   **Offset:** `0`
*   **Length:** 3 bytes
*   **Value:** `b'NAC'` (`0x4E`, `0x41`, `0x43`). These serve to quickly identify the file as belonging to the NAC format.
*   ---
*   **Offset:** `3`
*   **Length:** 1 byte
*   **Type:** `uint8`
*   **Description:** The version number of the standard. For the current specification, this value is `1`.

#### 2.2.2. Quantization and Storage Byte (1 byte)
*   **Offset:** `4`
*   **Length:** 1 byte
*   **Type:** `uint8` (bitfield)
*   **Description:** This byte encodes two parameters: the quantization method and the weight storage method.
    *   **Most Significant Bit (bit 7):** Weight storage flag.
        *   `1` (`0x80`): The model's weights are stored inside this `.nac` file in the `DATA` section.
        *   `0`: The weights are stored in an external `.safetensors` file, which must have the same name as the `.nac` file and be located in the same directory.
    *   **Least Significant 7 bits (bits 0-6):** Quantization method ID.
        *   `0`: No quantization (weights are in `float32` format).
        *   `1`: `FP16`.
        *   `2`: `INT8_TENSOR` (per-tensor INT8 quantization).
        *   `3`: `INT8_CHANNEL` (per-channel INT8 quantization).
        *   `4`: `BLOCK_FP8` (block-wise FP8 quantization).

#### 2.2.3. Input/Output Counts (5 bytes)
*   **Offset:** `5`
*   **Length:** 2 bytes
*   **Type:** `uint16`
*   **Description:** The number of user inputs that the model expects during execution.
*   ---
*   **Offset:** `7`
*   **Length:** 2 bytes
*   **Type:** `uint16`
*   **Description:** The number of outputs that the model returns.
*   ---
*   **Offset:** `9`
*   **Length:** 1 byte
*   **Description:** Reserved for future use, should be `0`.

#### 2.2.4. Model Dimension `d_model` (2 bytes)
*   **Offset:** `10`
*   **Length:** 2 bytes
*   **Type:** `uint16`
*   **Description:** The key dimension of the model, typically corresponding to the embedding size or hidden state size in transformers. A value of `0` indicates that it is not defined.

#### 2.2.5. Section Offset Table (76 bytes)

*   **Offset:** `12`
*   **Length:** 72 bytes (9 offsets of 8 bytes each)
*   **Type:** `uint64[]`
*   **Description:** An array of nine 64-bit unsigned integers, each representing the absolute offset (in bytes) from the beginning of the file to the start of the corresponding section. If a section is absent, its offset is `0`. The offsets are stored in the following order:

| Index | Field name       | Section tag | Description                                             |
|------:|------------------|-------------|---------------------------------------------------------|
| 0     | `mmap_offset`    | `MMAP`      | Memory Map — coprocessor schedule                       |
| 1     | `ops_offset`     | `OPS `      | Executable instruction stream                           |
| 2     | `cmap_offset`    | `CMAP`      | Canonical Map — operation ID → name                     |
| 3     | `cnst_offset`    | `CNST`      | Constants storage                                       |
| 4     | `perm_offset`    | `PERM`      | Permutations — signature ID → argument layout string    |
| 5     | `data_offset`    | `DATA`      | Parameter metadata and optional internal weight tensors |
| 6     | `proc_offset`    | `PROC`      | Preprocessing — compiled tokenizer manifest             |
| 7     | `orch_offset`    | `ORCH`      | Orchestrator — MEP bytecode and constant pool           |
| 8     | `rsrc_offset`    | `RSRC`      | Resources — embedded auxiliary files                    |

*   ---
*   **Offset:** `84`
*   **Length:** 4 bytes
*   **Description:** Padding. Reserved and should be filled with zeros.

### 2.3. Section Structure

#### 2.3.1. General Section Format (4-byte Tag, Data)
Each section (except for the header) begins with a 4-byte ASCII tag that identifies its type. The section's data immediately follows the tag. This structure allows for easy location and verification of sections when reading the file.

#### 2.3.2. Description of Section Tags
*   `MMAP`: (*Memory Map*) Contains a schedule of memory management commands for a parallel coprocessor.
*   `OPS `: Contains the model's executable code—a sequence of instructions in the ABCD format.
*   `CMAP`: (*Canonical Map*) A table mapping numerical operation IDs to their string-based canonical names.
*   `CNST`: (*Constants*) A storage for constant values (numbers, strings, lists) used in the graph.
*   `PERM`: (*Permutations*) A table mapping signature IDs to strings that describe the types and semantics of operation arguments.
*   `DATA`: Contains metadata about the model's parameters (weights) and inputs. If the internal storage flag is set, the weight tensors themselves are also stored here.
*   `PROC`: (*Processing*) A section for preprocessing data, primarily for a compiled tokenizer manifest.
*   `ORCH`: (*Orchestrator*) Contains the compiled MEP (Model Execution Plan) bytecode and its associated constant pool. See Section 9.
*   `RSRC`: (*Resources*) A storage for auxiliary resources, such as tokenizer vocabularies, merge files, etc.

**A Note on Autonomy:** The executable NAC code (`OPS`) does not require all model data (weights, resources) to be stored within a single `.nac` file. The storage flag in the header allows weights to be placed in an external `.safetensors` file, and tokenizer resources can be loaded from an external directory. This flexibility enables the creation of fully autonomous files for small models, as well as efficient handling of large models where weights can occupy tens of gigabytes.


## 3. The ABCD Instruction Format (OPS Section)

The `OPS` section is the core of a `.nac` file, containing the model's computation graph represented as a linear sequence of instructions. Each instruction has a variable length and is encoded in the ABCD format, which provides a detailed definition of the operation, its arguments, and its dependencies.

### 3.1. General Instruction Structure

An instruction consists of four sequential fields: `A`, `B`, `C`, and `D`. `A` and `B` have a fixed length of 1 byte each. The lengths of fields `C` and `D` are determined dynamically during a strictly sequential read.

```
[A (1 byte)] [B (1 byte)] [C (variable)] [D (variable)]
```

Each instruction in the sequence implicitly has an index, which is used for calculating dependencies. The result of executing the `i`-th instruction is available to subsequent instructions.

### 3.2. Field `A` (1 byte): Operation ID
*   **Type:** `uint8`
*   **Description:** A numerical identifier for the operation to be performed. This value serves as a key to look up the canonical name of the operation in the `CMAP` section. Field `A` also provides the basic branching logic for the parser.

#### 3.2.1. Regular Operations (`A >= 10`)
Values of `A` from 10 upwards are reserved for standard computational operations. For such operations, the subsequent fields `B`, `C`, and `D` are interpreted according to the general rules.

#### 3.2.2. Special (System) Operations (`A < 10`)
Values of `A` from 0 to 9 are reserved for special, "system," instructions (`<INPUT>`, `<OUTPUT>`, etc.) which have a unique structure for their `C` and `D` fields. (Described in detail in Section 7).

### 3.3. Field `B` (1 byte): Signature ID / Variant
*   **Type:** `uint8`
*   **Description:** This field has a dual purpose depending on the value of field `A`.

#### 3.3.1. For Regular Operations: Link to PERM Section
When `A >= 10`, field `B` is a numerical identifier for the operation's **signature**. This value is a key to look up a string in the `PERM` section that defines the number and semantics of the operation's arguments.

#### 3.3.2. For Special Operations: Subtype Discriminator
When `A < 10`, field `B` is used as a **discriminator** that specifies the variant of the system operation.

#### 3.3.3. Reserved Value `B=0`
A value of `B=0` signifies that the operation has no signature, which usually implies it has no arguments.

### 3.4. Field `C` (variable length): List of Constant IDs
*   **Element Type:** `int16`
*   **Description:** This field contains the identifiers of constants that are used as arguments for the operation.

#### 3.4.1. Determining Length and Reading the Field
**Rule:** For regular operations (`A >= 10`), if the signature `PERM[B]` contains at least one character denoting a constant, the parser expects a `C` field.
1.  **Read Length Prefix:** The parser reads the **first 2 bytes** as an `int16`. This value, `num_consts`, indicates how many *more* 16-bit constant IDs follow.
2.  **Read Constant IDs:** The parser reads `num_consts * 2` bytes.
3.  **The total length of field `C` in bytes is `2 + num_consts * 2`.**
4.  If `num_consts` is 0, field `C` consists of only 2 bytes containing a zero. If the signature contains no constants, this field might be absent (depending on the compiler's implementation). The implementation in `NAC_run.py` implies that if the signature expects constants, field `C` will be read.

### 3.5. Field `D` (variable length): Universal List of Pointers to Arguments
*   **Element Type:** `int16`
*   **Description:** This field contains a list of relative offsets pointing to the sources of **all** of the operation's arguments.

#### 3.5.1. Determining Field Length
The length of field `D` (in number of 16-bit elements) is **always strictly equal** to the length of the signature string obtained from the `PERM` section using the ID from field `B`. For example, for the signature `"TSc"` (length 3), field `D` will contain exactly three 16-bit values (6 bytes).

#### 3.5.2. Interpretation of Values
Each element `D[i]` corresponds to the `i`-th character in the signature string and is interpreted as follows:

*   **Non-Zero Value (Relative Offset):**
    If `D[i]` is not `0`, it is interpreted as a **relative offset** from the current instruction to a predecessor instruction. The runtime calculates the absolute index of the predecessor as `current_index + D[i]` and uses the result of that instruction's execution as the `i`-th argument. **This rule applies regardless of the argument type specified in the signature.** This means that even if the `i`-th signature character is a constant type (`'i'`, `'f'`), but `D[i]` is non-zero, the runtime will still fetch the result of another operation as this argument. This allows for dynamically computed values for arguments that are typically constants (e.g., a `dim` parameter).

*   **Zero Value (Marker for Constants from `C`):**
    If `D[i]` is `0`, it serves as a **marker** that instructs the runtime to fetch the next available value from the list of constant IDs previously read from field `C` for this argument. The order of zeros in `D` corresponds exactly to the order of constant IDs in `C` (after the length prefix).


## 4. Memory Map (MMAP Section)

The `MMAP` section contains a schedule of memory management commands designed to be executed by a dedicated parallel coprocessor. Its primary purpose is to decouple memory operations (loading weights, freeing buffers) from the main computational pipeline (`OPS` section), thus hiding memory latency and maximizing the utilization of computational units. The `MMAP` section essentially provides a microprogram for this memory coprocessor.

### 4.1. Purpose and Philosophy

Execution is synchronized by "ticks," where each tick corresponds to the index of an instruction in the `OPS` section. While the main core executes instruction `i` from `OPS`, the memory coprocessor executes all commands scheduled for tick `i` from `MMAP`.

This parallel execution enables critical optimizations:
*   **Asynchronous Preloading:** The coprocessor can initiate the loading of weights from slow memory (e.g., DRAM) into fast local memory (SRAM/cache) several ticks before they are needed by the main core.
*   **Efficient Buffer Management:** The coprocessor can free memory buffers immediately after their last use, preventing memory fragmentation and reducing the overall memory footprint.
*   **Pipelining:** Results from one operation can be directly forwarded to the next without being written to a shared memory space.

### 4.2. Section Format

The section begins with the 4-byte ASCII tag `MMAP`, followed by a 4-byte integer specifying the number of records.

| Field                 | Type      | Size (bytes) | Description                                                                  |
|-----------------------|-----------|:------------:|------------------------------------------------------------------------------|
| **Section Tag**       | `char[4]` | 4            | ASCII characters `'M' 'M' 'A' 'P'`.                                         |
| **Number of Records** | `uint32`  | 4            | The total number of ticks for which at least one memory command is scheduled.|
| **Records...**        | `Record[]`| variable     | A sequence of records, one for each scheduled tick.                          |

### 4.3. Record Format

Each record bundles all commands that must be executed at a specific tick.

| Field                     | Type      | Size (bytes) | Description                                                                    |
|---------------------------|-----------|:------------:|--------------------------------------------------------------------------------|
| **Instruction ID (Tick)** | `uint16`  | 2            | The index of the instruction in `OPS` at which these commands execute.         |
| **Number of Commands**    | `uint8`   | 1            | The number of `Command` structures that follow for this tick.                  |
| **Commands...**           | `Command[]`| variable    | A sequence of `Number of Commands` commands.                                   |

### 4.4. Command Format

Each command is an atomic instruction for the memory coprocessor.

| Field           | Type     | Size (bytes) | Description                                            |
|-----------------|----------|:------------:|--------------------------------------------------------|
| **Action Type** | `uint8`  | 1            | A numeric code identifying the type of memory operation.|
| **Target ID**   | `uint16` | 2            | The ID of an instruction that is the target of the action.|

### 4.5. Semantics of Actions and Targets

| Code | Command Name    | `Target ID` Interpretation                                                                                                                                          |
|:----:|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `10` | **SAVE_RESULT** | The ID of the **current** instruction. Instructs the coprocessor to save the result of the operation at the current tick into shared memory (used when consumed by multiple future operations). |
| `20` | **FREE**        | The ID of a **previously executed** instruction. Frees the memory buffer allocated for the result of that instruction or for a preloaded weight. Scheduled at the tick *after* the last use. |
| `30` | **FORWARD**     | The ID of the **next** instruction that will consume the result. Directly routes the result to the input of the target operation, bypassing shared memory. Used when there is only one consumer. |
| `40` | **PRELOAD**     | The ID of an `<INPUT B=1>` (`load_param`) instruction. Instructs the coprocessor to begin asynchronously loading the specified parameter from slow to fast memory before the main core reaches that instruction. |

#### Example

```
v43 = aten.relu.default(v42)   FREE -> 26,  SAVE_RESULT -> 43,  PRELOAD -> 44
```

At **tick 43**, the memory coprocessor executes three commands:

1.  **Record for Tick 43** — Instruction ID: `43`, Number of Commands: `3`
2.  **Command 1 (FREE):** Action Type: `20`, Target ID: `26`
3.  **Command 2 (SAVE_RESULT):** Action Type: `10`, Target ID: `43`
4.  **Command 3 (PRELOAD):** Action Type: `40`, Target ID: `44`


## 5. Metadata Sections

The metadata sections (`CMAP`, `PERM`, `CNST`) provide the runtime with all the necessary information to decode and interpret the instructions from the `OPS` section. These sections are loaded into memory during initialization and are used as lookup tables for fast access. Each section begins with a 4-byte tag, followed by a 4-byte integer (`uint32`) that specifies the number of entries in the section.

### 5.1. `CMAP` Section (Canonical Map)

#### 5.1.1. Purpose
The `CMAP` section contains the mapping from numerical operation identifiers (`A` from the ABCD instruction) to their canonical string names. This section is **optional** and is only present if the model uses operations that are not part of the standard, hardware-accelerated NAC set. Standard operations (e.g., `nac.add`, `nac.matmul`) have fixed, predefined IDs and are not recorded here.

#### 5.1.2. Record Format
*   **Operation ID** (2 bytes, `uint16`): The unique numerical identifier used in the `A` field.
*   **Name Length** (1 byte, `uint8`): The length of the operation name string in bytes (max 255).
*   **Operation Name** (variable, UTF-8): The canonical name, e.g., `aten.special_op.default`.

**Example:** Custom operation `custom.op` with ID=201: `0xC9 0x00` (ID), `0x0A` (length=10), UTF-8 bytes of `"custom.op"`.

#### 5.1.3. Role of `CANONICAL_OP_MAP` in Formation
The compiler uses an intelligent canonization mechanism:
1.  **Normalization:** Cleans PyTorch operation names (e.g., `aten.add_.Tensor` → `add`).
2.  **Direct Mapping:** Maps the cleaned name to a standard NAC operation (e.g., `add` → `nac.add`).
3.  **Alias Mapping (`CANONICAL_OP_MAP`):** For non-obvious names (`mm` → `nac.matmul`) or those requiring argument reordering (`rsub`).
4.  **Custom Operation:** If unmappable, the full original name is preserved, assigned a high-level ID (≥ 201), and this mapping is recorded in `CMAP`.

### 5.2. `PERM` Section (Permutations / Signatures)

#### 5.2.1. Purpose
The `PERM` section is key to decoding instruction arguments. It maps numerical signature IDs (`B` from the ABCD instruction) to strings that describe the number, order, type, and semantics of operation arguments.

#### 5.2.2. Record Format
*   **Signature ID** (2 bytes, `uint16`): The unique identifier used in the `B` field.
*   **String Length** (1 byte, `uint8`): The length of the signature string in bytes (max 255).
*   **Signature String** (variable, UTF-8): A sequence of characters, one per argument.

**Example:** Signature `TSc` with ID=100: `0x64 0x00` (ID), `0x03` (length=3), `0x54 0x53 0x63` (`"TSc"`).

#### 5.2.3. Specification of Codes in the Signature String

*   **Tensor codes** (corresponding `D[i]` is a non-zero relative offset):
    *   `Q`: Query (attention)
    *   `K`: Key (attention)
    *   `V`: Value (attention)
    *   `M`: Mask
    *   `B`: Bias vector
    *   `W`: Weight matrix
    *   `T`: Generic tensor input
    *   `P`: Parameter / persistent buffer

*   **Constant codes** (corresponding `D[i]` is `0`; value is taken from `C`):
    *   `A`: Axis / Dimension (usually `int`)
    *   `S`: Shape / Size (usually `list[int]`)
    *   `i`: `integer`
    *   `f`: `float`
    *   `b`: `boolean`
    *   `s`: `string`
    *   `c`: Complex / other JSON-serializable constant (e.g., `torch.dtype` as string)

### 5.3. `CNST` Section (Constants)

#### 5.3.1. Purpose
Centralized storage for all constant scalar and list values used in the computation graph. Instructions refer to these values via IDs from the `C` field.

#### 5.3.2. Record Format

*   **Constant ID** (2 bytes, `uint16`): Unique identifier.
*   **Type Code** (1 byte, `uint8`): Data type of the constant.
    *   `0`: `null`
    *   `1`: `bool`
    *   `2`: `int64`
    *   `3`: `float64`
    *   `4`: `string`
    *   `5`: `list[int32]`
    *   `6`: `list[float32]`
*   **Length** (2 bytes, `uint16`): For types `2, 3, 4` — length in **bytes** of data. For types `5, 6` — number of **elements**. For types `0, 1` — ignored.
*   **Value** (variable, binary): For `int64` — 8 bytes LE. For `float64` — 8 bytes LE. For `list[int32]` — sequence of 4-byte LE integers. For `list[float32]` — sequence of 4-byte LE floats.

**Example:** Constant `[1, 2]` with ID=50: `0x32 0x00` (ID), `0x05` (type=list_int), `0x02 0x00` (length=2), `0x01 0x00 0x00 0x00`, `0x02 0x00 0x00 0x00`.


## 6. Data and Resource Sections (DATA, RSRC, PROC)

### 6.1. `DATA` Section

The `DATA` section stores parameter metadata and, optionally, the weight tensors themselves. It begins with the 4-byte tag `DATA`.

The section is read sequentially in three blocks in the following order:

**Block 1: Parameter ID → Name mapping**

Maps numerical parameter IDs (used in `<INPUT B=1>`) to their original string names from the source model.

*   Starts with a 4-byte `uint32` giving the number of records.
*   Each record:
    *   **Parameter ID** (2 bytes, `uint16`)
    *   **Name Length** (2 bytes, `uint16`)
    *   **Name** (variable, UTF-8), e.g., `layers.0.attention.self.query.weight`

**Block 2: Input index → Name mapping**

Maps the absolute index of each `<INPUT B=0>` instruction to the name of the corresponding model input.

*   Starts with a 4-byte `uint32` giving the number of records.
*   Each record:
    *   **Instruction Index** (2 bytes, `uint16`): Absolute index `i` in the `OPS` section.
    *   **Name Length** (2 bytes, `uint16`)
    *   **Name** (variable, UTF-8), e.g., `input_ids`

**Block 3: Internal weight tensors** *(present only when the internal storage flag is set in the header)*

*   Starts with a 4-byte `uint32` giving the number of tensors.
*   Each tensor record:

| Field               | Type     | Size (bytes)       | Description                                        |
|---------------------|----------|--------------------|-----------------------------------------------------|
| **Parameter ID**    | `uint16` | 2                  | Links this tensor to Block 1 name mapping.          |
| **Metadata Length** | `uint32` | 4                  | Total length of the binary metadata block in bytes. |
| **Data Length**     | `uint64` | 8                  | Length of the raw tensor data in bytes.             |
| **Metadata**        | binary   | `Metadata Length`  | Compact binary descriptor — see §6.1.1.             |
| **Data**            | binary   | `Data Length`      | Raw contiguous tensor data, no padding.             |

#### 6.1.1. Binary Tensor Metadata Format

The metadata block is a **compact binary structure**, not JSON. It is parsed sequentially:

| Field         | Type       | Size (bytes)     | Description                                             |
|---------------|------------|------------------|---------------------------------------------------------|
| **DType ID**  | `uint8`    | 1                | Data type enum (see table below).                       |
| **Rank**      | `uint8`    | 1                | Number of dimensions.                                   |
| **Shape**     | `uint32[]` | `Rank × 4`       | Dimensions in order, each as a 4-byte LE unsigned int.  |
| **Quant Code**| `uint8`    | 1                | Quantization type code (same enum as header byte §2.2.2).|

**DType encoding:**

| Code | Type       | Code | Type     |
|:----:|------------|:----:|----------|
| `0`  | `float32`  | `5`  | `int64`  |
| `1`  | `float64`  | `6`  | `int16`  |
| `2`  | `float16`  | `7`  | `int8`   |
| `3`  | `bfloat16` | `8`  | `uint8`  |
| `4`  | `int32`    | `9`  | `bool`   |

**Example:** A 2-D `float16` tensor of shape `[768, 768]` with no quantization would produce the metadata block:
```
02        ← DType = float16 (2)
02        ← Rank = 2
00 03 00 00   ← 768 (LE uint32)
00 03 00 00   ← 768 (LE uint32)
00        ← Quant = none (0)
```
Total metadata length: 11 bytes.

### 6.2. `PROC` Section (Processing)

#### 6.2.1. Purpose
Stores data required for preprocessing steps executed before the main model runs. Currently used exclusively to store a compiled **tokenizer manifest** in the TISA format.

#### 6.2.2. Format
The section begins with the `PROC` tag, followed by:
*   **Manifest Length** (4 bytes, `uint32`): Total length of the manifest binary data in bytes.
*   **Data** (variable): Binary data of the tokenizer manifest.

### 6.3. `RSRC` Section (Resources)

#### 6.3.1. Purpose
Embeds auxiliary resource files into the `.nac` file, primarily tokenizer support files (`vocab.json`, `merges.txt`, `spiece.model`, etc.).

#### 6.3.2. Format
The section begins with the `RSRC` tag, followed by:
*   **Number of Files** (4 bytes, `uint32`).
*   A sequence of file records:
    *   **Name Length** (2 bytes, `uint16`)
    *   **Name** (variable, UTF-8), e.g., `vm_vocab.json`
    *   **Data Length** (4 bytes, `uint32`)
    *   **Data** (variable): Raw binary content.


## 7. Special (System) Operations

Special operations with an ID `< 10` are the fundamental building blocks of the graph, managing data input/output and the flow of execution. Unlike regular computational operations, they have a unique structure for their `C` and `D` fields that does not depend on signatures from the `PERM` section.

### 7.1. `<INPUT>` Operation (ID=2)

Serves as the entry point for all data into the graph. The `B` field specifies the data source. `D` is always empty (no dependencies).

#### 7.1.1. Variant `B=0`: User Data Input
*   **Description:** One of the model's user inputs (e.g., `input_ids`, `attention_mask`). The runtime expects a corresponding tensor for each such instruction at inference time.
*   **C:** `[]` (empty)

#### 7.1.2. Variant `B=1`: Parameter (Weight/Bias)
*   **Description:** Loads a weight or bias tensor from storage.
*   **C:** `[2, param_id]` — length prefix `2` followed by the 16-bit `param_id`, which links this instruction to the `DATA` section.

#### 7.1.3. Variant `B=2`: State
*   **Description:** Loads a state tensor (e.g., KV-cache). The runtime manages persistent storage and updating of these states between calls.
*   **C:** `[2, state_id]` — length prefix `2` followed by a 16-bit `state_id`.

#### 7.1.4. Variant `B=3`: Lifted Constant
*   **Description:** A constant value "lifted" to the input level by `torch.export` because it could not be statically proven to be invariant. Can be either a tensor constant (e.g., `torch.ones(...)`) or a scalar (e.g., `1.0`, `True`). The runtime expects a corresponding value in the input list for each such instruction, just like `B=0`. The `DATA` section may contain a descriptive name for it via the input index mapping.
*   **C:** `[2, const_id]` — length prefix `2` followed by a 16-bit `const_id` referencing the `CNST` section.

### 7.2. `<OUTPUT>` Operation (ID=3)

Defines the output data of the graph. Collects results of previous instructions and returns them.

#### 7.2.1. Variant `B=0`: Final Model Output
*   **Description:** Designates the final result(s) of the entire model. Execution terminates after this instruction.
*   **C:** `[num_outputs + 1, ...]` — the `C` field has length `num_outputs + 1`; the purpose of individual elements beyond the length prefix is reserved.
*   **D:** `[offset_1, ..., offset_n]` — relative offsets to the instructions whose results are the model's outputs. The count equals `num_outputs` from the file header.

#### 7.2.2. Variant `B=1`: Intermediate Output (Subgraph)
*   **Description:** Returns a result from a subgraph branch without terminating the full graph execution.
*   **D:** Relative offsets to the subgraph output instructions.

### 7.3. Reserved Operations

#### 7.3.1. `<CONTROL_FLOW>` (ID=6)
*   **Purpose:** Reserved for conditional branching (if/else).
*   **Intended structure:**
    *   `C`: `[3, true_branch_len, false_branch_len]`
    *   `D`: `[predicate_offset]` — offset to the boolean condition instruction.
*   **Status:** Not implemented in the reference runtime.

#### 7.3.2. `<CONVERGENCE>` (ID=7)
*   **Purpose:** Merges multiple parallel execution branches (MoE experts, model ensembling).
*   **Structure (unique, breaks general ABCD rules):**
    *   `B`: Coherence threshold (0–100) for intelligent merging.
    *   `C`: `[input_offset]`
    *   `D`: `[end_block_offset, num_branches, branch_1_offset, ...]`
*   **Status:** Experimental; the reference implementation covers demonstration use cases only.


## 8. Execution Process (Runtime Logic)

### 8.1. Loading and Initialization

1.  **Read Header:** Extract version, quantization flag, I/O counts, and section offsets.
2.  **Load Metadata:**
    *   Predefined standard NAC operations are populated into `id_to_canonical`.
    *   **`CMAP`:** Custom `(ID → Name)` pairs are merged into `id_to_canonical`.
    *   **`PERM`:** `(ID → Signature String)` pairs are loaded into `permutations`.
    *   **`CNST`:** Records are deserialized into native types and stored in `constants`.
3.  **Load Parameters:**
    *   *External weights:* `DATA` Block 1 provides `(ID → Name)` mapping; tensors are loaded from the `.safetensors` file by name.
    *   *Internal weights:* `DATA` Block 3 is read; for each tensor the binary metadata is parsed (§6.1.1) to recover `dtype`, `shape`, and quantization code, then raw bytes are deserialized into a native tensor.
4.  **Dequantize:** If quantization is specified, weights are converted to `float32` once at load time.
5.  **Initialize States:** Storage for KV-cache and other states is allocated if `<INPUT B=2>` instructions are present.

### 8.2. Instruction Execution Loop

1.  Allocate `results[N]` buffer (N = number of instructions).
2.  For each instruction `i` from `0` to `N-1`:
    1.  Read `A`, `B`, `C`, `D`.
    2.  Gather arguments (§8.3).
    3.  Look up the operation by `A` in `id_to_canonical`, invoke the corresponding kernel.
    4.  Store the result in `results[i]`.
3.  Terminate when `<OUTPUT B=0>` is executed or the instruction stream ends; return the designated outputs.

### 8.3. Argument Gathering Mechanism

For instruction `i` with fields `A`, `B`, `C`, `D`:

1.  Look up `signature = permutations[B]`.
2.  Create iterators `d_iter` over `D` and `c_iter` over the constant IDs in `C`.
3.  For each character `p_code` in `signature`, advance `d_iter` to get `d_val`:
    *   **`d_val ≠ 0`:** Compute `ancestor = i + d_val`; append `results[ancestor]` to arguments.
    *   **`d_val == 0`:** Consume the next ID from `c_iter`; look it up in `constants`; append that value.
4.  The resulting ordered argument list is passed to the operation kernel.


## 9. Orchestrator Section (ORCH)

### 9.1. Purpose

The `ORCH` section stores the compiled output of the **MEP (Model Execution Plan) Orchestrator**, a higher-level scheduler that sits above the raw `OPS` instruction stream. Its primary role is to encode a pre-compiled execution plan—in the form of compact bytecode—that directs an intelligent runtime (e.g., the `NAC_MEP` interpreter or a dedicated hardware coprocessor) on how to dispatch, sequence, and manage resources across the computational graph.

The `ORCH` section is optional. Its absence means no pre-compiled execution plan is available, and the runtime falls back to sequential interpretation of `OPS`.

### 9.2. Section Format

The section begins with the 4-byte tag `ORCH`, immediately followed by two 4-byte fields:

| Field             | Type     | Size (bytes) | Description                                                       |
|-------------------|----------|:------------:|-------------------------------------------------------------------|
| **Section Tag**   | `char[4]`| 4            | ASCII `'O' 'R' 'C' 'H'`.                                         |
| **Bytecode Length** | `uint32` | 4          | Total size of the MEP bytecode stream in bytes.                   |
| **Const Count**   | `uint32` | 4            | Number of constants in the MEP constant pool.                     |
| **Bytecode**      | binary   | `Bytecode Length` | The compiled MEP instruction stream.                         |
| **Const Pool**    | binary   | variable     | The MEP constant pool, referenced by bytecode instructions.       |

Both `Bytecode Length` and `Const Count` are read together as a single 8-byte read (`struct '<II'`).

### 9.3. Relationship to Other Sections

The MEP bytecode operates at a level above `OPS`: it references `OPS` instruction indices as targets, uses the same `DATA` and `CNST` sections for operands, and can encode bulk operations (e.g., "execute block of ops 10–50 in parallel") that are too coarse-grained for the per-instruction `MMAP` schedule. The `ORCH` section complements rather than replaces `MMAP`.

---

### Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

---

## License

The source code of this project is licensed under the **GNU General Public License v3.0**. See the `LICENSE` file for details.

The accompanying documentation, including this README and the project's White Paper, is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.
