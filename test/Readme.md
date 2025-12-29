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

### **Stage 2: Executing and Testing Models (using `NACmodels_test.py`)**

This is the online process that uses the generated `.nac` files. It demonstrates the main advantage of NAC—running models in a lightweight environment without heavy dependencies.

#### 2.3. Requirements

*   The compiled `.nac` files (and `.safetensors` if weights are stored externally).
*   Installed libraries: `numpy` and `Pillow` (for image processing models).
*   The `imagenet_classes.json` file (for ResNet), which must be downloaded separately and placed in the same directory.

#### 2.4. Running the Execution Script

The script is run from the command line, specifying the `.nac` file and the input data.

```bash
# Image Classification
python NACmodels_test.py resnet18.nac path/to/your/image.jpg

# Fill-Mask (uses the built-in tokenizer)
python NACmodels_test.py roberta-base-fill-mask.nac "The capital of France is <mask>."

# Text Generation (uses the built-in tokenizer)
python NACmodels_test.py gpt2-text-generation.nac "Hello, I'm a language model,"

# Sentiment Analysis (uses the built-in tokenizer)
python NACmodels_test.py distilbert-sst2-sentiment.nac "This movie is absolutely fantastic!"

# Image Generation (Stable Diffusion)
python NACmodels_test.py sd-unet-256.nac "A photo of an astronaut riding a horse on mars"

# Translation (T5)
python NACmodels_test.py t5-encoder.nac t5-decoder.nac "My name is Wolfgang and I live in Berlin"
```

The script will automatically detect the task based on the filename, initialize `NacRuntime`, preprocess the input data (including tokenization via the built-in TISA VM), run the model, and print the result.

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
