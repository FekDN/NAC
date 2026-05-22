## Guide to Running and Testing the NAC Ecosystem

This guide describes the full workflow for working with NAC: from setting up the environment and compiling models to running them and analyzing the results with supplementary utilities.

### 1. Project Structure and File Layout

For the system to function correctly, all scripts must be located in the same directory. Ensure your working directory contains the following files:

*   `NAC.py` (The main compiler)
*   `NAC_run.py` (The runtime environment)
*   `NAC_kernels.py` (Kernels for computational operations used by the runtime)
*   `NAC_optimizer.py` (Graph optimizer for the compiler)
*   `TISA_tokenizer.py` (Compiler and runtime for the autonomous tokenizer)
*   `CompileTest.py` (Script for batch-compiling models)
*   `NACmodels_test.py` (Script for testing the compiled models)
*   `NAC_info.py` (Utility for inspecting `.nac` files)
*   `NAC_reconstructor.py` (Utility for reconstructing pseudo-code from `.nac` files)

### 2. Workflow

The process is divided into two distinct stages: **Compilation** and **Execution**.

---

### **Stage 1: Compiling Models (using `CompileTest.py`)**

This is an offline process performed once for each model. Its purpose is to convert standard models from `transformers` or `torchvision` into the `.nac` format.

#### 2.1. Environment Setup

1.  **Install Libraries:** You will need specific versions of the libraries due to the instability of `torch.export`. It is highly recommended to create a separate virtual environment.

    ```bash
    pip install torch==2.5.1 torchvision
    pip install transformers==4.57.3
    pip install diffusers accelerate safetensors huggingface_hub numpy Pillow
    ```
    **Attention:** Using different versions of `torch` and `transformers` may require significant changes to the code in `CompileTest.py` (within the model wrappers and `dummy_args`) and `NACmodels_test.py` (in the input preparation logic).

2.  **Download Model Weights:** On the first run, the `CompileTest.py` script will automatically download all required models from the Hugging Face Hub. This may take a long time and require substantial disk space. Subsequent runs will use the local cache.

3.  **The `registry.json` File:** The script will create and update a file named `registry.json` in the current directory. This file stores a global mapping of all operations, constants, and signatures encountered across all compiled models. It is important to run the compilation in the same directory to ensure the registry is comprehensive.

#### 2.2. Running the Compilation Script

Execute the following command in your terminal:

```bash
python CompileTest.py
```

The script will sequentially process all models, performing the following steps for each:
1.  **Load** the pre-trained model from `transformers` or `torchvision`.
2.  **Create a Wrapper:** The model is often enclosed in a special wrapper class to adapt its interface and fix incompatibilities with `torch.export`.
3.  **Export to Graph (`torch.export`):** The wrapped model is traced using dummy input data (`dummy_args`).
4.  **Optimize the Graph (`NAC_optimizer.py`):** The "raw" graph is cleaned and optimized.
5.  **Generate Artifacts (`NAC.py`):** The final assembly of the `.nac` file, `.safetensors` (if needed), and the update of `registry.json` takes place.

Upon completion, the compiled `.nac` files for all models will be available in your directory.

---

### **Stage 2: Executing, Testing, and Fine-Tuning Models (using `NACmodels_test.py`)**

The `NACmodels_test.py` script serves as the universal orchestrator for running any `.nac` container. Since execution logic (Inference, Training, and Pre/Post-processing) is encoded natively inside the `.nac` file's MEP section, this single script can handle vision, text, and generative models without requiring model-specific Python code.

#### 2.4. General Usage

The script relies on interactive prompts built into the `.nac` file, but you can also provide pre-answers via the command line interface (CLI) to automate the flow.

```bash
python NACmodels_test.py <model.nac> [CLI arguments (pre-answers)] [--flags]
```

**Basic Inference Examples:**

```bash
# Vision (ResNet / MobileNetV3)
python NACmodels_test.py resnet18.nac cat.jpg
python NACmodels_test.py mobilenet_v3_small.nac dog.png

# Fill-Mask (RoBERTa)
python NACmodels_test.py roberta-base-fill-mask.nac "The capital of France is <mask>."

# Sentiment Analysis (DistilBERT)
python NACmodels_test.py distilbert-sst2-sentiment.nac "This is absolutely terrible."

# Translation (T5)
# Notice: T5 MEP orchestrator relies on both encoder and decoder parts 
# combined via Python's execution logic natively written in CompileTest
python NACmodels_test.py t5-decoder.nac "translate English to German: Hello world"

# Text Generation (GPT-2)
python NACmodels_test.py gpt2-text-generation.nac "In a shocking turn of events, "

# Image Generation (Stable Diffusion)
python NACmodels_test.py sd-vae-decoder.nac "A cyberpunk city at night, neon lights"
```

#### 2.5. Execution Modes (`--mode`)

The `--mode` flag allows you to selectively execute branches within the MEP orchestrator.

*   `--mode infer_train` (Default): Performs a standard forward pass (inference), shows results, and then prompts the user for a target label to perform fine-tuning.
*   `--mode infer`: Executes only the inference branch and exits.
*   `--mode train`: Bypasses the initial inference result printout and immediately begins fine-tuning based on the provided inputs.

```bash
# Only perform inference (no training prompts)
python NACmodels_test.py mobilenet_v3_small.nac dog.png --mode infer
```

#### 2.6. Advanced Training Strategies (`--train-mode`)

The runtime supports two fundamentally different mathematical strategies for On-Device Training.

*   `--train-mode head_only` (Default): Highly efficient Transfer Learning. The runtime performs an analytical gradient calculation for the output head and dynamically auto-expands the classification layer (neurons/classes) if the target ID is higher than the current capacity. Only the final `Linear` layer is updated. Recommended for edge devices.
*   `--train-mode trng`: Executes the complete, compiled backward graph (`TRNG` section). Propagates gradients through all registered un-frozen layers in the network. PyTorch's AOTAutograd graph is utilized. *Note: Memory intensive.*

```bash
# Provide both the input (image) and the training target (42) automatically
# Perform full backpropagation via the TRNG graph
python NACmodels_test.py resnet18.nac bird.jpg 42 --mode train --train-mode trng
```

#### 2.7. In-Memory Patching and Binary Rewriting (`--patch` / `--rewrite`)

A revolutionary feature of the NAC format is the ability to modify hyperparameters (like epochs, learning rates, thresholds) natively mapped into the binary, without needing to re-compile the graph.

*   `--patch KEY=VALUE`: Temporarily modifies a constant in memory for the current run only.
*   `--rewrite KEY=VALUE`: Modifies the constant and surgically overwrites the actual `.nac` file on disk.

```bash
# Run fine-tuning, but temporarily change the learning rate from 0.005 to 0.0001
# Provide inputs: [1] image path, [2] target class
python NACmodels_test.py mobilenet_v3_small.nac tree.jpg 1000 --patch 0.005=0.0001

# Permanently rewrite the "num_epochs" constant from 3 to 10 in the .nac file
python NACmodels_test.py distilbert-sst2-sentiment.nac "Awesome" 1 --rewrite 3=10
```

---

### 3. Auxiliary Utilities

The NAC ecosystem includes two useful utilities for analysis and debugging.

#### 3.1. `NAC_info.py` — File Inspector

This utility allows you to "look inside" a `.nac` file and view its structure and contents in a human-readable format. It displays information from the header, all metadata tables (CMAP, PERM, CNST), the parameter list, and also disassembles the tokenizer manifest.

**Usage:**
```bash
python NAC_info.py gpt2-text-generation.nac
```

#### 3.2. `NAC_reconstructor.py` — Pseudo-code Reconstructor

This utility performs a reverse transformation: it reads the binary instruction graph from the `OPS` section and reconstructs pseudo-code that resembles Python. This is extremely useful for debugging, analyzing the model's logic, and verifying the correctness of the compiler and optimizer.

**Usage:**
```bash
python NAC_reconstructor.py gpt2-text-generation.nac
```

---

### 4. Principles of the `generate_artifacts` call (for Developers)

The `generate_artifacts` function in `NAC.py` is the central point of the compilation process. Understanding its arguments is crucial for adding new models.

*   `model_name: str`: The base name for the output files.
*   `model: torch.nn.Module`: The model instance (often a wrapper).
*   `dummy_args: Tuple`: A tuple of dummy tensors for tracing.
*   `quantization_method: str`: The quantization method (`'none'`, `'INT8_TENSOR'`, etc.).
*   `dynamic_shapes`: A dictionary for describing dynamic axes.
*   `store_weights_internally: bool`: A flag indicating whether to store weights inside the `.nac` file (`True`) or in a separate `.safetensors` file (`False`).
*   `tokenizer_repo: str`: The Hugging Face repository for fetching tokenizer data.
*   `optimize: bool`: Enables or disables the graph optimization stage.
*   `tokenizer_input: str`: A representative string for calibrating the autonomous TISA tokenizer.
