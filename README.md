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

## How It Works: The NAC Compilation Pipeline

The framework implements a full compilation and analysis pipeline:

`AI Model (PyTorch) -> [Decomposer] -> Flat NAC Graph -> [Pattern Miner] -> [Compressor] -> Compressed NAC Signature & Rules`

1.  **Decomposition ("Sequencing"):** A model is traced and converted into a "flat" graph of fundamental NAC operations.
2.  **Pattern Mining ("Gene Finding"):** The analyzer processes a large collection of flat graphs to discover recurring, statistically significant patterns ("genes").
3.  **Compression ("Genomic Assembly"):** The original flat graph is re-encoded using the discovered patterns, creating a compact, high-level, hierarchical signature.
4.  **Rule Extraction ("Comparative Genomics"):** A Rule Miner analyzes the compressed signatures to find high-level design principles, transition probabilities, and structural relationships.

## Roadmap

This project lays the groundwork for a new field of "neuro-compilation" and "computational genomics for AI."

#### Short-Term Goals
-   [ ] Develop robust decomposers for a wide range of architectures (CNNs, RNNs, GNNs).
-   [ ] Enhance the Pattern Miner to support graph-based and non-contiguous patterns (e.g., skip-connections).
-   [ ] Extend the NAC ISA to include control flow operations (`if`, `for`) for RNNs and dynamic models.
-   [ ] Execute the large-scale "Comparative Genomics" study on 90+ representative models to build the foundational Knowledge Base.

#### Long-Term Vision
-   [ ] Establish **NAC-ISA as a new open standard** for AI architecture representation.
-   [ ] Develop a prototype **NAC-native AI processor** (FPGA/ASIC).
-   [ ] Build a **generative AI system** that designs novel, high-performance architectures based on the NAC Knowledge Base.
-   [ ] Publish groundbreaking research on the "Universal Code of Neural Networks."

---


 ## Neural Architecture Code (NAC) v1.1 Specification

 The NAC format is designed for the compact, unified, and machine-readable
 representation of neural network computation graphs. Each graph is a sequence
 of nodes, where each node is described by a variable-length command.

 **1. Node Structure**
 Every node in the binary stream consists of a fixed-length header (5 bytes)
 and an optional variable-length data block. The structure is determined by the
 value of the first byte (`A`).

 There are three command types:
 1. Fundamental Operation (standard `ABCD[]` command)
 2. Pattern Invocation (command with `<PATTERN_PREFIX>`)
 3. Copy Command (command with `<COPY_PREFIX>`)

 **2. Byte `A`: Prefix and Operation ID (1 byte, Unsigned)**
 Byte `A` defines the command type and/or serves as an index for a fundamental operation.

   - `0`: `<NONE>` - Reserved. Should not appear in valid graphs.
   - `1`: `<PAD>` - Reserved for data alignment if needed.
   - `2`: `<DATA_INPUT>` - A node representing an entry point for user-provided data (e.g., `input_ids`).
   - `3`: `<PARAM_INPUT>` - A node representing an entry point for model parameters (weights, biases). Mostly legacy; `<CONST_REF>` is preferred.
   - `4`: `<OUTPUT>` - A node whose inputs are the outputs of the entire graph.
   - `5`: `<CONST_REF>` - A node representing a reference to a parameter, identified by its name ID (e.g., `get_attr` in torch.fx).
   - `6`: `<PATTERN_PREFIX>` - Command Prefix. Indicates this node is a high-level pattern (macro) invocation.
   - `7`: `<COPY_PREFIX>` - Command Prefix. Indicates this node is a compression command for repeating nodes.
   - `8-9`: `<RESERVED>` - Reserved for future command types (e.g., control flow).
   - `10-255`: Fundamental Operation ID - A direct index into the `registry.json["canonical_to_index"]` dictionary.

 **3. Command Types and Formats**

 **3.1. Command: Fundamental Operation (`A >= 10`)**
 The standard command for executing a single mathematical operation.

   - Format: `A B C D[]`
   - Length: 5 bytes + `2 * num_inputs`
   - Fields:
     - `A` (1 byte, `10-255`): Operation ID from `registry.json["canonical_to_index"]`.
     - `B` (1 byte, `1-255`): Call variation ID from `registry.json["variation_to_index"]`. Describes the canonical permutation of tensor inputs.
     - `C` (2 bytes, `signed`): Index for a constant or constant group.
       - `C = 0`: No constants are used.
       - `C > 0`: A direct ID for a literal constant from `registry.json["constants"]`.
       - `C < 0`: An ID for a group of constants. The `group_id` is `-(C + 1)`. The group definition is in `registry.json["index_to_constant_group"]`.
     - `D[]` (variable): List of input dependencies.
       - `num_inputs` (1 byte): The number of inputs.
       - `input_id` (2 bytes, `unsigned`) x `num_inputs`: Absolute IDs of source nodes within the current graph.

 **3.2. Command: Pattern Invocation (`A = 6`)**
 This command executes an entire subgraph (a pattern) defined in the registry.

   - Format: `A E₀ E₁ unused D[]`
   - Length: 5 bytes + `2 * num_inputs`
   - Fields:
     - `A` (1 byte): Always `6` (`<PATTERN_PREFIX>`).
     - `E₀` (1 byte, occupies the `B` field): The lower byte of the 2-byte pattern ID.
     - `E₁` (1 byte, 1st byte of the `C` field): The upper byte of the 2-byte pattern ID.
       - The full `pattern_id` is reassembled as `(E₁ << 8) | E₀`, allowing for 65,536 unique pattern IDs (0 to 65535).
     - `unused` (1 byte, 2nd byte of the `C` field): Reserved, should be `0`.
     - `D[]` (variable): List of arguments (inputs) passed to the pattern. The structure is identical to `D[]` for a fundamental operation.

 **3.3. Command: Copy (`A = 7`)**
 This command is used to compress sequences of identical, input-less nodes
 (Run-Length Encoding), such as repeated `get_attr` calls.

   - Format: `A B C D[]`
   - Length: 5 bytes + `2 * num_inputs` (which is `5 + 2*2 = 9` bytes total)
   - Fields:
     - `A` (1 byte): Always `7` (`<COPY_PREFIX>`).
     - `B` (1 byte): The Operation ID (`A` component) of the node to be copied.
     - `C` (2 bytes, `signed`): The number of times the node should be repeated.
     - `D[]` (4 bytes): Repurposed to store the `B` and `C` components of the node being copied.
       - `num_inputs` (1 byte): Always `2`.
       - `input_1` (2 bytes, `signed`): The `B` component (Variation ID) of the template node.
       - `input_2` (2 bytes, `signed`): The `C` component (Constant ID) of the template node.

 **4. Relation to `registry.json`**
 The NAC format is not self-contained. It requires a corresponding `registry.json` file
 for interpretation. The registry acts as a dictionary, mapping the integer IDs used
 in the binary format to their full string definitions, constant values, parameter
 names, and pattern definitions.
 
---

*   feklindn@gmail.com 

---

## License

The source code of this project is licensed under the **GNU General Public License v3.0**. See the `LICENSE` file for details.

The accompanying documentation, including this README and the project's White Paper, is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.
