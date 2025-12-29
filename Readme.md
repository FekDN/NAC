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

 ## Neural Architecture Code (NAC) v1.5 Specification

 The NAC format is designed for the compact, unified, and machine-readable
 representation of neural network computation graphs. Each graph is a sequence
 of nodes, where each node is described by a variable-length command.

 
## 1. Introduction and Philosophy of the NAC Standard

### 1.1. Purpose of the Standard (Target Platform: FPGA/ASIC)

The NAC standard is designed not as a format for general-purpose processors (CPU/GPU), but as a **specification for implementation on specialized hardware**, such as Field-Programmable Gate Arrays (FPGA) and Application-Specific Integrated Circuits (ASIC). The Python implementation provided within the ecosystem serves as a functional reference interpreter and a development tool, but it is not the target platform for achieving maximum performance. The ultimate objective is to create a hardware core capable of natively executing NAC instructions, achieving maximum efficiency and power savings.

### 1.2. Overall Ecosystem Architecture (Compiler -> Format -> Runtime)

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

### 1.4. Versioning (Current Version: 1)

The current version of the standard is **NAC v1**. The version is explicitly stated in the header of the `.nac` file, which ensures backward compatibility and allows for future extensions. Any changes to the header structure, instruction format, or the semantics of system operations will require an increment of the standard's version.


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

#### 2.2.5. Section Offset Table (72 bytes + 4 bytes padding)
*   **Offset:** `12`
*   **Length:** 72 bytes (9 offsets of 8 bytes each)
*   **Type:** `uint64[]`
*   **Description:** An array of nine 64-bit unsigned integers, each representing the absolute offset (in bytes) from the beginning of the file to the start of the corresponding section. If a section is absent, its offset is `0`.
    1.  `ops_offset`: Offset to the `OPS` section.
    2.  `cmap_offset`: Offset to the `CMAP` section.
    3.  `cnst_offset`: Offset to the `CNST` section.
    4.  `perm_offset`: Offset to the `PERM` section.
    5.  `data_offset`: Offset to the `DATA` section.
    6.  `proc_offset`: Offset to the `PROC` section.
    7.  `meta_offset`: Offset to the `META` section.
    8.  `rsrc_offset`: Offset to the `RSRC` section.
    9.  `reserved2_offset`: Reserved.
*   ---
*   **Offset:** `84`
*   **Length:** 4 bytes
*   **Description:** Padding. Reserved and should be filled with zeros.

### 2.3. Section Structure

#### 2.3.1. General Section Format (4-byte Tag, Data)
Each section (except for the header) begins with a 4-byte ASCII tag that identifies its type. The section's data immediately follows the tag. This structure allows for easy location and verification of sections when reading the file.

#### 2.3.2. Description of Section Tags
*   `OPS `: Contains the model's executable code—a sequence of instructions in the ABCD format.
*   `CMAP`: (`Canonical Map`) A table mapping numerical operation IDs to their string-based canonical names.
*   `CNST`: (`Constants`) A storage for constant values (numbers, strings, lists) used in the graph.
*   `PERM`: (`Permutations`) A table mapping signature IDs to strings that describe the types and semantics of operation arguments.
*   `DATA`: Contains metadata about the model's parameters (weights) and inputs. If the internal storage flag is set, the weight tensors themselves are also stored here.
*   `PROC`: (`Processing`) A section for preprocessing data, primarily for a compiled tokenizer manifest.
*   `RSRC`: (`Resources`) A storage for auxiliary resources, such as tokenizer vocabularies, merge files, etc.
*   `META`: A reserved section for future extended metadata.

**A Note on Autonomy:** The executable NAC code (`OPS`) does not require all model data (weights, resources) to be stored within a single `.nac` file. The storage flag in the header allows weights to be placed in an external `.safetensors` file, and tokenizer resources can be loaded from an external directory. This flexibility enables the creation of fully autonomous files for small models, as well as efficient handling of large models where weights can occupy tens of gigabytes.


## 3. The ABCD Instruction Format (OPS Section)

The `OPS` section is the core of a `.nac` file, containing the model's computation graph represented as a linear sequence of instructions. Each instruction has a variable length and is encoded in the ABCD format, which provides a detailed definition of the operation, its arguments, and its dependencies.

### 3.1. General Instruction Structure

An instruction consists of four sequential fields: `A`, `B`, `C`, and `D`. `A` and `B` have a fixed length of 1 byte each. The lengths of fields `C` and `D` are determined dynamically during a strictly sequential read.

`[A (1 byte)] [B (1 byte)] [C (variable)] [D (variable)]`

Each instruction in the sequence implicitly has an index, which is used for calculating dependencies. The result of executing the `i`-th instruction is available to subsequent instructions.

### 3.2. Field `A` (1 byte): Operation ID
*   **Type:** `uint8`
*   **Description:** A numerical identifier for the operation to be performed. This value serves as a key to look up the canonical name of the operation in the `CMAP` section. Field `A` also provides the basic branching logic for the parser.

#### 3.2.1. Regular Operations (`A >= 10`)
Values of `A` from 10 upwards are reserved for standard computational operations. For such operations, the subsequent fields `B`, `C`, and `D` are interpreted according to the general rules.

#### 3.2.2. Special (System) Operations (`A < 10`)
Values of `A` from 0 to 9 are reserved for special, "system," instructions (`<INPUT>`, `<OUTPUT>`, etc.) which have a unique structure for their `C` and `D` fields. (Described in detail in Section 6).

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


## 4. Metadata Sections

The metadata sections (`CMAP`, `PERM`, `CNST`) provide the runtime with all the necessary information to decode and interpret the instructions from the `OPS` section. These sections are loaded into memory during initialization and are used as lookup tables for fast access. Each section begins with a 4-byte tag, followed by a 4-byte integer (`uint32`) that specifies the number of entries in the section.

### 4.1. `CMAP` Section (Canonical Map)

#### 4.1.1. Purpose
The `CMAP` section contains the mapping from numerical operation identifiers (`A` from the ABCD instruction) to their canonical string names. This allows the binary format to be compact by using 1-byte IDs instead of full string names in every instruction, while still maintaining readability and debuggability.

#### 4.1.2. Record Format
The section consists of a sequence of records, each having the following structure:

*   **Operation ID (2 bytes, `uint16`):** The unique numerical identifier for the operation, used in the `A` field of an instruction.
*   **Name Length (1 byte, `uint8`):** The length of the subsequent operation name string in bytes. The maximum name length is 255 bytes.
*   **Operation Name (variable length, UTF-8):** The canonical name of the operation, e.g., `nac.add` or `aten.layer_norm.default`.

**Example:** The record for `nac.add` with ID=16 would look like this in binary: `0x10 0x00` (ID=16), `0x07` (length=7), `0x6E 0x61 0x63 0x2E 0x61 0x64 0x64` ("nac.add").

#### 4.1.3. Role of `CANONICAL_OP_MAP` in Formation
The `CMAP` section is generated during the compilation stage (`NAC.py`) based on the `CANONICAL_OP_MAP` dictionary. This dictionary performs semantic normalization:
1.  **Aliasing:** Many different operations from PyTorch ATen (e.g., `aten.add.Tensor`, `add`) are reduced to a single canonical name (`nac.add`).
2.  **Generalization:** More complex operations (e.g., `aten.masked_fill`) are converted into their more general counterparts (`nac.where`).

Thanks to this process, the dictionary of canonical operations (and consequently, the `CMAP` section) remains small and stable, which simplifies the runtime and makes the 256-ID limit for operations more than sufficient.

### 4.2. `PERM` Section (Permutations / Signatures)

#### 4.2.1. Purpose
The `PERM` section is key to decoding instruction arguments. It contains the mapping from numerical signature IDs (`B` from the ABCD instruction) to strings that describe the number, order, type, and semantics of the arguments for an operation.

#### 4.2.2. Record Format
The structure of records in `PERM` is similar to `CMAP`:

*   **Signature ID (2 bytes, `uint16`):** The unique numerical identifier used in the `B` field.
*   **String Length (1 byte, `uint8`):** The length of the subsequent signature string in bytes (maximum 255).
*   **Signature String (variable length, UTF-8):** A sequence of characters, where each character describes one argument.

**Example:** The record for the signature `TSc` (tensor, shape, other constant) with ID=100 might look like: `0x64 0x00` (ID=100), `0x03` (length=3), `0x54 0x53 0x63` ("TSc").

#### 4.2.3. Specification of Codes in the Signature String
Each character in the signature string represents one argument and defines its type (tensor or constant), and may also carry additional semantic information.

*   **Codes for Tensors** (indicate that the corresponding element in the `D` field is a relative offset):
    *   `Q`: Query (attention)
    *   `K`: Key (attention)
    *   `V`: Value (attention)
    *   `M`: Mask
    *   `B`: Bias vector
    *   `W`: Weight matrix
    *   `T`: Tensor (a generic tensor input)

*   **Codes for Constants** (indicate that the corresponding element in the `D` field is `0`, and the value is taken from `C`):
    *   `A`: Axis / Dimension (usually an `int`)
    *   `S`: Shape / Size (usually a `list[int]`)
    *   `i`: `integer`
    *   `f`: `float`
    *   `b`: `boolean`
    *   `s`: `string`
    *   `c`: `complex` (any other JSON-serializable constant, e.g., a `torch.dtype` as a string)

### 4.3. `CNST` Section (Constants)

#### 4.3.1. Purpose
The `CNST` section serves as a centralized storage for all constant values used in the computation graph. Instructions refer to these values via IDs from the `C` field.

#### 4.3.2. Record Format
Records in `CNST` have the following structure:

*   **Constant ID (2 bytes, `uint16`):** The unique numerical identifier for the constant.
*   **String Length (1 byte, `uint8`):** The length of the subsequent JSON string in bytes (maximum 255).
*   **JSON String (variable length, UTF-8):** The value of the constant, serialized into JSON format. Using JSON allows for a flexible representation of various data types: numbers (`123`, `1.5e-5`), strings (`"hello"`), booleans (`true`), lists (`[1, 2, 3]`), and `null`.

**Example:** The record for the constant `1.0` with ID=50 might look like: `0x32 0x00` (ID=50), `0x03` (length=3), `0x31 0x2E 0x30` ("1.0").


## 5. Data and Resource Sections (DATA, RSRC, PROC)

These sections are responsible for storing data that is not part of the executable graph but is necessary for its operation: model parameters (weights), information about inputs and outputs, and auxiliary resources such as tokenizer data.

### 5.1. `DATA` Section

This section is the central repository for metadata about parameters and, optionally, for the parameter tensors themselves. It begins with the 4-byte tag `DATA`.

#### 5.1.1. Mapping of Parameter IDs to Names
This part of the section maps the numerical parameter IDs, used in `<INPUT>` instructions of type `B=1`, to their original string names from the source model. This is useful for debugging and for loading external weights from a `.safetensors` file, where tensors are identified by name.
*   **Format:** It starts with a 4-byte integer (`uint32`) specifying the number of records, followed by a sequence of records, each with the structure:
    *   **Parameter ID (2 bytes, `uint16`):** The numerical ID of the parameter.
    *   **Name Length (2 bytes, `uint16`):** The length of the parameter name in bytes.
    *   **Name (variable length, UTF-8):** The parameter's name, e.g., `layers.0.attention.self.query.weight`.

#### 5.1.2. Mapping of Input Indices to Names
Similar to parameters, this part maps the index of an `<INPUT>` instruction of type `B=0` to the name of the corresponding model input.
*   **Format:** A 4-byte integer (`uint32`) with the number of records, followed by the records themselves:
    *   **Instruction Index (2 bytes, `uint16`):** The absolute index `i` of the `<INPUT>` instruction in the `OPS` section.
    *   **Name Length (2 bytes, `uint16`):** The length of the input name.
    *   **Name (variable length, UTF-8):** The input's name, e.g., `input_ids`.

#### 5.1.3. Internal Weight Storage Format
This data block is present in the `DATA` section **only if** the internal weight storage flag is set in the file's header.
*   **Format:** The block begins with a 4-byte integer (`uint32`) indicating the number of tensors stored in the file, followed by a sequence of tensor records.
*   **Tensor Record Structure:**
    *   **Parameter ID (2 bytes, `uint16`):** The ID that links this tensor to its name from section 5.1.1.
    *   **Number of Meta Properties (1 byte, `uint8`):** The number of key-value pairs in the metadata block.
    *   **Metadata Length (4 bytes, `uint32`):** The total length of the metadata block in bytes.
    *   **Data Length (8 bytes, `uint64`):** The length of the tensor's binary data in bytes.
    *   **Metadata (variable length):** A block containing additional information about the tensor.
    *   **Data (variable length):** The raw binary data of the tensor.

*   **Metadata Format (Key-Value):**
    The metadata block consists of a sequence of `(key, value)` records without a general header. Each record has the structure:
    *   **Key Length (1 byte, `uint8`):** The length of the key string.
    *   **Key (UTF-8):** The key string (e.g., `shape`, `dtype`, `quant_type`).
    *   **Value Length (2 bytes, `uint16`):** The length of the JSON value string.
    *   **Value (UTF-8):** The JSON-serialized value string.

*   **Data Type Encoding (DType Enum):**
    In the metadata, the `dtype` key uses a numerical enumeration to encode the tensor's data type:
    *   `0`: `float32`
    *   `1`: `float64`
    *   `2`: `float16`
    *   `3`: `bfloat16`
    *   `4`: `int32`
    *   `5`: `int64`
    *   `6`: `int16`
    *   `7`: `int8`
    *   `8`: `uint8`
    *   `9`: `bool`

### 5.2. `PROC` Section (Processing)

#### 5.2.1. Purpose
The `PROC` section is designed to store data required for preprocessing steps that are executed before the main model runs. In the current version of the standard, it is used exclusively to store a compiled **tokenizer manifest** in the TISA format.

#### 5.2.2. Format
The section's structure is very simple and consists of the `PROC` tag followed by:
*   **Manifest Length (4 bytes, `uint32`):** The total length of the manifest's binary data in bytes.
*   **Data (variable length):** The binary data of the tokenizer manifest.

### 5.3. `RSRC` Section (Resources)

#### 5.3.1. Purpose
The `RSRC` section is used for embedding auxiliary resource files into the `.nac` file, which allows for the creation of fully self-contained artifacts. It is primarily used to store files required for the tokenizer to function, such as vocabularies (`vocab.json`), merge files (`merges.txt`), or SentencePiece models (`spiece.model`).

#### 5.3.2. Format
The section begins with the `RSRC` tag, followed by:
*   **Number of Files (4 bytes, `uint32`):** The total number of files stored in the section.
*   This is followed by a sequence of records for each file.

#### 5.3.3. File Record Format
Each embedded file is described by the following structure:
*   **Name Length (2 bytes, `uint16`):** The length of the file name in bytes.
*   **Name (variable length, UTF-8):** The name of the file, e.g., `vm_vocab.json`.
*   **Data Length (4 bytes, `uint32`):** The length of the file's binary content in bytes.
*   **Data (variable length):** The raw binary content of the file.


## 6. Special (System) Operations

Special operations with an ID < 10 are the fundamental building blocks of the graph, managing data input/output and the flow of execution. Unlike regular computational operations, they have a unique structure for their `C` and `D` fields that does not depend on signatures from the `PERM` section.

### 6.1. `<INPUT>` Operation (ID=2)

This operation serves as the entry point for all data into the graph, whether it's user input, model parameters, or constants. The `B` field specifies the data source.

*   **A = 2**
*   **D = []** (empty, as an input has no dependencies)

#### 6.1.1. Variant `B=0`: Data Input
*   **Description:** Represents one of the model's user inputs (e.g., `input_ids`, `attention_mask`). The runtime expects a corresponding tensor to be provided for each such instruction when the model is run.
*   **B = 0**
*   **C = []** (empty)

#### 6.1.2. Variant `B=1`: Parameter (Weight/Bias)
*   **Description:** Loads a parameter tensor (weight or bias) from storage.
*   **B = 1**
*   **C = `[2, param_id]`** (length 2, `param_id` is a 16-bit parameter ID). The ID links this instruction to a parameter name in the `DATA` section and to the corresponding tensor (either internal or external).

#### 6.1.3. Variant `B=2`: State
*   **Description:** Loads a state tensor. This operation is designed for models that maintain state between calls, such as the KV-cache in generative transformers. The runtime manages the storage and updating of these states.
*   **B = 2**
*   **C = `[2, state_id]`** (length 2, `state_id` is a 16-bit state ID).

#### 6.1.4. Variant `B=3`: Lifted Constant
*   **Description:** Loads a constant that was "lifted" from the graph to the input level during the `torch.export` process. These are typically tensor constants (e.g., created via `torch.ones`) that are not parameters.
*   **B = 3**
*   **C = `[2, const_id]`** (length 2, `const_id` is a 16-bit constant ID from the `CNST` section).

### 6.2. `<OUTPUT>` Operation (ID=3)

This operation defines the output data of a graph or subgraph. It collects the results of previous instructions and returns them.

*   **A = 3**
*   **C:** Defined by the `B` variant.

#### 6.2.1. Variant `B=0`: Final Model Output
*   **Description:** Designates the final result(s) of the entire model. The graph's execution terminates after this instruction.
*   **B = 0**
*   **C = `[num_outputs + 1, ...]`** (the length of the `C` field is the number of outputs + 1). The purpose of the elements in `C` is not defined in the current version.
*   **D = `[offset_1, offset_2, ..., offset_n]`**: A list of relative offsets to the instructions whose results are the model's outputs. The number of offsets is equal to `num_outputs` from the file header.

#### 6.2.2. Variant `B=1`: Intermediate Output (for Subgraphs)
*   **Description:** Used to return a result from a subgraph, for example, within a branch of conditional execution or a `<CONVERGENCE>` operation. It does not terminate the execution of the entire graph.
*   **B = 1**
*   **C:** Not used / structure is undefined.
*   **D:** A list of relative offsets to the output instructions of the subgraph.

### 6.3. Reserved Operations

These operations are defined in the standard, but their full implementation in the reference runtime may be absent or specific. They lay the groundwork for future extensions.

#### 6.3.1. `<CONTROL_FLOW>` (ID=6)
*   **Purpose:** Reserved for implementing conditional branching (if/else).
*   **Structure (intended):**
    *   `B`: Undefined.
    *   `C`: `[3, true_branch_len, false_branch_len]`. `true_branch_len` and `false_branch_len` are the lengths of the respective branches in instructions.
    *   `D`: `[predicate_offset]`. A relative offset to the instruction whose result (a boolean tensor) is used as the condition.
*   **Status:** Not implemented in `NAC_run.py`.

#### 6.3.2. `<CONVERGENCE>` (ID=7)
*   **Purpose:** Reserved for complex logic involving the merging of multiple parallel execution branches. The primary application is for merging experts in MoE models or for model ensembling.
*   **Structure:** Has a unique structure that breaks the general ABCD rules.
    *   `B`: A coherence threshold (0-100) for "intelligent" merging.
    *   `C`: `[input_offset]`. An offset to the input tensor that is passed to each branch.
    *   `D`: `[end_block_offset, num_branches, branch_1_offset, branch_2_offset, ...]`. Describes the structure of the parallel branches.
*   **Status:** The implementation in `NAC_run.py` is a demonstration and may not cover all use cases. It is considered an experimental feature.


## 7. Execution Process (Runtime Logic)

The runtime is the component responsible for loading a `.nac` file, interpreting the ABCD instructions, and performing the computations. The reference implementation, `NAC_run.py`, demonstrates all the key stages of this process.

### 7.1. Loading and Initialization
Before computations begin, the runtime performs a one-time setup:

1.  **Read Header:** The 88-byte header is read to extract the version, quantization information, input/output counts, and, most importantly, the offsets to all sections.
2.  **Load Metadata into Memory:**
    *   **`CMAP` Section:** The `(ID, Name)` records are loaded into a dictionary (hash table) named `id_to_canonical` for fast mapping of operation IDs to their names.
    *   **`PERM` Section:** The `(ID, Signature String)` records are loaded into the `permutations` dictionary.
    *   **`CNST` Section:** The `(ID, JSON String)` records are loaded into the `constants` dictionary, where the JSON string is deserialized into the appropriate Python data type (number, list, etc.).
3.  **Load Parameters:**
    *   **External Weights:** If the weight storage flag in the header is cleared, the runtime looks for a corresponding `.safetensors` file. It reads the `(ID -> Name)` mapping from the `DATA` section and loads the tensors from the `.safetensors` file into the `parameters` dictionary by their names.
    *   **Internal Weights:** If the flag is set, the runtime reads the tensor block from the `DATA` section. For each tensor, it reads and deserializes the metadata (including `shape` and `dtype`), then reads the binary data and constructs a tensor (e.g., a `numpy.ndarray`) from it.
4.  **Dequantize Parameters (If Necessary):** Immediately after loading, if the tensor's metadata or the file header specifies a quantization method (e.g., `INT8_CHANNEL`), the runtime applies the reverse operation (dequantization), converting the weights to the standard `float32` format. This is done once at load time to simplify subsequent computations.
5.  **Initialize States:** The runtime initializes storage for states (e.g., KV-cache) if the graph contains any `<INPUT>` instructions with type `B=2`.

### 7.2. Instruction Execution Loop (from OPS Section)

After initialization, the runtime is ready to perform computations. The process is a loop over the array of instructions, pre-loaded from the `OPS` section.

1.  **Initialize Results Buffer:** An array named `results` is created with a size equal to the number of instructions in the graph. `results[i]` will store the result of executing the `i`-th instruction.
2.  **Iteration:** The runtime iterates through the instructions from `i = 0` to `N-1`.
3.  **Parse Instruction:** At each iteration `i`, the runtime reads the `A`, `B`, `C`, and `D` fields of the current instruction.
4.  **Gather Arguments:** The argument gathering mechanism is invoked (see 7.3), which prepares a list of arguments to be passed to the operation's kernel.
5.  **Invoke Operation Kernel:**
    *   Using the operation ID `A`, the runtime finds the corresponding name in `id_to_canonical`.
    *   Based on the operation name, the corresponding computational function (kernel), e.g., `op_nac_add`, is found and called.
    *   The prepared list of arguments is passed to the kernel.
6.  **Store Result:** The result returned by the kernel is stored in `results[i]`.
7.  **Termination:** The loop continues until an `<OUTPUT>` instruction (with `B=0`) is executed or the end of the instruction list is reached. The result(s) specified in the `<OUTPUT>` instruction are returned to the user.

### 7.3. Argument Gathering Mechanism

This is a key part of the runtime's logic at each iteration. For the current instruction `i` with fields `A`, `B`, `C`, `D`:

1.  **Get Signature:** The signature string (e.g., `"TSc"`) is retrieved from the `permutations` dictionary using `ID=B`.
2.  **Prepare Iterators:** Iterators are created for the list `D` (`d_iter`) and for the list of constants from `C` (`c_iter`).
3.  **Iterate Over Signature:** The runtime iterates over each character `p_code` in the signature string, simultaneously advancing the `d_iter` iterator.
    1.  The next value `d_val` is fetched from `d_iter`.
    2.  **If `d_val` is not `0` (Relative Offset):**
        *   The ancestor index is calculated: `ancestor_idx = i + d_val`.
        *   The value `results[ancestor_idx]` is fetched from the results buffer.
        *   This result is added to the argument list for the current operation.
    3.  **If `d_val` is `0` (Constant Marker):**
        *   The next constant ID is fetched from the `c_iter` iterator.
        *   The actual constant value is retrieved from the `constants` dictionary using this ID.
        *   This value is added to the argument list.
4.  **Result:** Upon completion of the iteration, a complete, ordered list of arguments is formed, ready to be passed to the operation's kernel.

---

## Copyright (c) 2025 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

---

## License

The source code of this project is licensed under the **GNU General Public License v3.0**. See the `LICENSE` file for details.


The accompanying documentation, including this README and the project's White Paper, is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.

