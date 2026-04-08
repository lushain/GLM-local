# GLM-OCR Remote A100 Setup Guide

This guide will walk you through setting up and running the GLM-OCR model on your remote Nvidia A100 machine. It is assumed you are running on a standard Linux environment via an online SSH terminal.

## Prerequisites
- **Git** (to clone your repository)
- **Python 3.9+**
- **Nvidia Drivers / CUDA** (Ensure `nvidia-smi` works and you have CUDA 12.1 or 12.4+ supported drivers).

## Step-by-Step Setup

### 1. Clone the Code
Once you have pushed your local code to GitHub, open your remote A100 terminal and clone the repository:
```bash
git clone <your-github-repo-url>
cd <your-repository-directory> # Replace with the name of your repo
```

### 2. Set Up a Virtual Environment 
Creating an isolated environment ensures you don't break the system's Python packages.

Using `venv`:
```bash
python3 -m venv venv
source venv/bin/activate
```
*(Alternatively, if you use `conda`, you can create an environment via `conda create -n glmocr python=3.10 -y && conda activate glmocr`)*

### 3. Install GPU-Accelerated PyTorch 
Since you have an A100, we want to maximize performance by installing PyTorch with the latest CUDA bindings:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install the GLM-OCR SDK
Install the official package with self-hosted layout features enabled. This pulls all required dependencies for parsing complex documents (like PDFs, layouts, formulas).
```bash
pip install "glmocr[selfhosted]"
```

### 5. Run the Inference Script
We've set up the script to automatically default to reading `testDocs/dixon51.pdf` and outputting the Markdown results.

Execute the script:
```bash
python run_glm_ocr.py
```
*Note: Due to the size of the 0.9B parameter model, the very first time you run this command on your A100, it may take 1-2 minutes to download the model weights from Hugging Face. Subsequent runs will be much faster.*

---

## Overriding the Default Document
If you wish to test a different file later, simply pass the path using the `--document_path` flag:
```bash
python run_glm_ocr.py --document_path some_other_document.png --output ./my_output_directory
```
