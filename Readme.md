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
 -   [X] ~~Extend the NAC ISA to include control flow operations (`if`, `while`) for RNNs and dynamic models.~~
 -   [ ] Execute the large-scale "Comparative Genomics" study on 90+ representative models to build the foundational Knowledge Base.

 #### Long-Term Vision
 -   [ ] Establish **NAC-ISA as a new open standard** for AI architecture representation.
 -   [ ] Develop a prototype **NAC-native AI processor** (FPGA/ASIC).
 -   [ ] Build a **generative AI system** that designs novel, high-performance architectures based on the NAC Knowledge Base.
 -   [ ] Publish groundbreaking research on the "Universal Code of Neural Networks."

---


 ## Neural Architecture Code (NAC) v1.2 Specification

 The NAC format is designed for the compact, unified, and machine-readable
 representation of neural network computation graphs. Each graph is a sequence
 of nodes, where each node is described by a variable-length command.

 **1. Node Structure**
 Every node in the binary stream consists of a fixed-length header (5 bytes)
 and an optional variable-length data block. The structure is determined by the
 value of the first byte (`A`).

 There are four command types:
 1. Fundamental Operation (standard `ABCD[]` command)
 2. Pattern Invocation (command with `<PATTERN_PREFIX>`)
 3. Copy Command (command with `<COPY_PREFIX>`)
 4. Control Flow Command (command with `<CONTROL_FLOW_PREFIX>`)

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
   - `8`: `<CONTROL_FLOW_PREFIX>` - Command Prefix. Defines a structured control flow block (e.g., if/while).
   - `9`: `<RESERVED>` - Reserved for future command types.
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
       - `input_id` (2 bytes, `signed`) x `num_inputs`: Source node identifiers. Can be an absolute ID or a relative offset (see Section 5).

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
     - `D[]` (variable): List of absolute IDs for arguments (inputs) passed to the pattern. The structure is identical to `D[]` for a fundamental operation.

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

 **3.4. Command: Control Flow (`A = 8`)**
 Defines a structured control flow construct like `IF-ELSE` or `WHILE`. The logical
 blocks (`then`, `else`, `loop_body`) must be laid out linearly in the NAC stream
 immediately following this command node.

   - Format: `A B C D[]`
   - Length: 5 bytes + `2 * num_inputs`
   - Fields:
     - `A` (1 byte): Always `8` (`<CONTROL_FLOW_PREFIX>`).
     - `B` (1 byte): Sub-opcode defining the control flow type.
       - `1`: `IF-ELSE`.
       - `2`: `WHILE`.
     - `C` (2 bytes, `signed`): Unused, should be `0`.
     - `D[]` (variable): Contains metadata describing the layout of the following blocks and their external dependencies. All IDs are absolute.
       - For `IF-ELSE (B=1)`:
         - `D[0]`: `len_then_block` (number of nodes in the `then` block).
         - `D[1]`: `len_else_block` (number of nodes in the `else` block).
         - `D[2]`: `condition_node_id` (absolute ID of the node whose boolean result is the condition).
         - `D[3...]`: Absolute IDs of external nodes required by the `then`/`else` blocks.
       - For `WHILE (B=2)`:
         - `D[0]`: `len_condition_graph` (number of nodes in the condition-evaluation subgraph).
         - `D[1]`: `len_loop_body_graph` (number of nodes in the loop body subgraph).
         - `D[2...]`: Absolute IDs of nodes providing the initial state for the loop variables.

 **4. Relation to `registry.json`**
 The NAC format is not self-contained. It requires a corresponding `registry.json` file
 for interpretation. The registry acts as a dictionary, mapping the integer IDs used
 in the binary format to their full string definitions, constant values, parameter
 names, and pattern definitions:
 ```json
   {
      "canonical_to_index": {},
      "index_to_canonical": {},
      "variation_to_index": {},
      "constants": {},
      "constant_group_to_index": {},
      "index_to_constant_group": {},
      "param_name_to_index": {},
      "index_to_param_name": {},
      "patterns": {},
      "rules": {
         "transition_rules": {},
         "composition_rules": {},
         "structural_rules": {}
      }
   }
 ```

 *   `canonical_to_index`: Maps a canonical operation signature string (e.g., `"aten.add.Tensor:node_args(2):kwargs()"`) to its unique integer ID (the `A` component).
 *   `index_to_canonical`: The reverse mapping of the above, from an integer ID back to the signature string.
 *   `variation_to_index`: Maps an input permutation signature (e.g., `"in0,in1"`) to its unique integer ID (the `B` component).
 *   `constants`: A dictionary mapping a constant's unique integer ID to its serialized value (e.g., `{"type": "int", "value": 1}`). This is the primary store for literal constants (the `C` component).
 *   `constant_group_to_index`: Maps a sorted string of constant assignments (e.g., `"arg1=10;arg2=True"`) to a unique group ID, used when a node has multiple constant inputs.
 *   `index_to_constant_group`: The reverse mapping of the above.
 *   `param_name_to_index`: Maps a parameter's string name (e.g., `"layer.0.weight"`) to its unique integer ID. Used by the `<CONST_REF>` operation.
 *   `index_to_param_name`: The reverse mapping of the above.
 *   `patterns`: Maps a pattern's unique integer ID to its Base64-encoded NAC binary representation. This is the library of discovered "genes."
 *   `rules`: Contains high-level design principles discovered by the `RuleMiner`.
     *   `transition_rules`: Probabilistic rules describing which node is likely to follow another (e.g., `"After 'LayerNorm' -> 'MultiHeadAttention' with 95% probability"`).
     *   `composition_rules`: Frequently occurring combinations of (Operation, Variation, Constants), representing common "semantic idioms."
     *   `structural_rules`: Multi-input relationships describing how a node is formed from a specific combination of parent node types.

 **5. Addressing Model: Absolute vs. Relative**
 NAC uses a hybrid addressing model for node dependencies specified in the `D[]` component.
 The type of addressing depends on the context (scope) of the link.

   - **5.1. Global Scope (Absolute Addressing):**
     - **When:** Links between nodes in the main graph, or links that cross the boundary
       into a logical block (e.g., inputs to a pattern or a control flow structure).
     - **Format:** A non-negative signed 2-byte integer (`0` to `32767`) representing the
       absolute index of the source node from the beginning of the graph.

   - **5.2. Block Scope (Relative Addressing):**
     - **When:** Links between nodes that are *inside* a self-contained logical block.
       This applies to nodes within a pattern definition or within the body of a
       control flow construct (`then`, `else`, `loop_body`, `condition_body`).
     - **Format:** A negative signed 2-byte integer (`-1`, `-2`, etc.) representing
       the offset from the *current* node's position.
     - **Example:** `D=[-1]` refers to the immediately preceding node within the same block.
     - **Benefit:** This makes logical blocks (patterns, loops) modular and relocatable. They can
       be moved within the graph without needing to recompute all internal dependency links.

---

## Why NAC? The Power of Canonical Representation

 The NAC standard is built on atomic primitives that cover over 95% of all computations in modern AI models. In effect, we are encoding the PyTorch ATen library into a binary instruction format.

 One might ask: why use the canonical `A, B, C` approach instead of simply assigning a unique 2-byte ID to each of the ~3,000 operators in `native_functions.yaml` (the source of all ATen operators)?

 The answer lies in the difference between **syntax** and **semantics**. The `native_functions.yaml` file doesn't contain 3,000 unique semantic operations; it contains a vast number of overloads for the same core ideas.

 For example, the `add` operation appears in `native_functions.yaml` as:
 - `add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1)`
 - `add.Scalar(Tensor self, Scalar other, Scalar alpha=1)`
 - `add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out)`
 - `add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1)` (in-place version)
 - `add_.Scalar(Tensor(a!) self, Scalar other, *, Scalar alpha=1)` (in-place version)
 - ...and several other variations.

 A naive encoding would treat these as completely distinct operations, creating syntactic noise that hides their fundamental relationship. The `A,B,C,D[]` approach solves this through **semantic unification**:

 *   **`A` (Operation ID):** Encodes the **semantic core** of an operation. For instance, `aten.add` with a specific number of tensor inputs and keyword argument names.
     *   `"aten.add.Tensor:node_args(2):kwargs()"` -> `A = 150`
     *   `"aten.add.Tensor:node_args(1):kwargs(alpha)"` -> `A = 151`
 *   **`B` (Variation ID):** Encodes **syntactic variations**—how tensors and scalars are passed as inputs (e.g., `in0,in1` vs. `in1,in0`).
 *   **`C` (Constant ID):** Encodes **non-tensor data** (scalars, strings, booleans, such as `alpha=1.5`).
 *   **`D[]` (Dependencies):** Encodes the **graph structure** by passing data dependencies, much like function arguments.

 In NAC, `add.Tensor` and `add.Scalar` can share the same `A` (semantic core) but differ in `C` (since `other` is a constant in the second case). This preserves semantic commonality. As a result, the **PatternMiner can discover a pattern involving "addition" regardless of whether a tensor is added to another tensor or a scalar.** Under a naive ATen encoding, these would be two entirely different, unrelated patterns.

 ### The ATen-based Registry: A Breakthrough Approach

 Basing the NAC registry on ATen provides several key advantages:

 *   **Stability and Standardization:** The ATen C++ API is far more stable across PyTorch versions than the Python API. Its operations have clear, strongly-typed signatures. NAC captures this stability in `registry.json`, creating a durable, long-term standard.
 *   **Rich Metadata:** ATen operations are rich with metadata (`dtype`, `layout`, `device`, `autograd` status). NAC encodes this information into the `B` (variation) and `C` (constant) components, making the ISA **context-aware**.
 *   **Hardware Mapping:** These canonical, fundamental operations can be directly mapped to an ASIC, creating a universal computational environment for any model.
 By analyzing a model's NAC signature, we can do more than just identify patterns; we can quantify them. A simple counter on the A component (Operation ID) across a graph or an entire corpus of models provides a precise, data-driven profile of computational demand. This tells us exactly how many times each fundamental operation (e.g., conv2d, matmul, layer_norm) is called. This statistical profile is a direct specification for ASIC design, allowing architects to determine the optimal number of identical computational units (e.g., matrix multiplication cores, normalization engines) needed for efficient, and potentially single-pass, model execution.

 ### From Data-Driven Discovery to Innovation

 Searching for patterns in a flat `ABCD[]` graph is **data-driven discovery, not hand-crafted engineering.** We are empirically discovering the "genetic alphabet" of AI from the collective experience of thousands of models, rather than imposing human assumptions. NAC uncovers hidden design principles by analyzing raw data from real-world architectures.

 This leads to the discovery of "unexpected" patterns as a source of innovation:

 #### a) Convergent Evolution
 The flat graph analysis will reveal universal semantics hidden by implementation details.
 *   **Different Architectures, Same Core Pattern:**
     *   **ResNet:** `Conv2D → BatchNorm → ReLU`
     *   **EfficientNet:** `DepthwiseConv → Swish → SEBlock`
     *   **NAC View:** Both may reduce to the same fundamental `ABCD[]` sequence for `"Conv → Norm → Activation"`. NAC proves that this semantic idiom is universal, regardless of the specific activation function or normalization layer used.

 #### b) Mutations and Alleles
 We can identify variations of a core "gene," creating a quantitative basis for a phylogenetic tree.
 *   **One Gene, Different Alleles:**
     *   **Base Pattern:** `[Conv_Op, Norm_Op, Act_Op]`
     *   **ResNet Allele:** `[Conv2D, BatchNorm, ReLU]`
     *   **MobileNet Allele:** `[DepthwiseConv, BatchNorm, ReLU6]`
     *   **ConvNeXt Allele:** `[DepthwiseConv, LayerNorm, GELU]`

 #### c) Non-Obvious Patterns
 NAC can discover novel, cross-domain patterns in hybrid architectures that are not immediately apparent.
 *   **What might NAC find in hybrid models?**
     *   `Pattern_Hybrid_001: [Attention, Conv2D, Upsample]` (Common in Diffusion models?)
     *   `Pattern_Unknown_002: [Linear, Dropout, Linear, SkipToLayerN]` (An MLP-Mixer motif?)

 ### Statistical Significance vs. Hand-Crafted Blocks

 The problem with hand-crafted approaches is bias. Researchers favor "classic" blocks (`Conv+BN+ReLU`), publication bias favors "elegant" architectures, and domain specificity limits generalization.

 **The NAC Advantage:**

 *   **Data-Driven:** Patterns are extracted from the real-world evidence of thousands of models.
 *   **Unbiased:** The most frequent patterns are, by definition, the most successful and effective architectural choices.
 *   **Cross-Domain:** Universal patterns emerge that are effective across vision, NLP, audio, and graphs.

 Finally, NAC opens the door to discovering **"non-contiguous" or "gapped" patterns**. Many architectures contain structurally disconnected blocks that are functionally linked. These are akin to "genetic regulatory networks" in biology—genes located on different chromosomes that work in concert to perform a single function. NAC provides the framework to discover these deep, functional relationships.

---

*   feklindn@gmail.com 

---

## License

The source code of this project is licensed under the **GNU General Public License v3.0**. See the `LICENSE` file for details.


The accompanying documentation, including this README and the project's White Paper, is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.
