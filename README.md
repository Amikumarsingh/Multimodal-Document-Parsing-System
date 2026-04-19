# 📄 Multimodal Document Parsing System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A production-grade, end-to-end framework for extracting structured intelligence from unstructured document images. This project implements and benchmarks three cutting-edge architectural approaches to Document AI, providing a comprehensive toolkit for researchers and ML engineers.

---

## 🏗️ Architecture Overview

The system is designed with a **modular-first** philosophy, allowing developers to switch between heuristic-based and transformer-based pipelines with zero code changes.

### 1. **Pipeline A: OCR + Spatial Heuristics**
*   **Best for**: Fixed-layout forms (Invoices, Tax Forms).
*   **Logic**: Uses Tesseract OCR followed by a spatial proximity engine in `pipelines/inference_rules.py`.

### 2. **Pipeline B: Layout-Aware Transformers (LayoutLM v3)**
*   **Best for**: Complex, multi-page business documents.
*   **Logic**: A multimodal model using unified text, layout, and visual embeddings. Orchestrated in `pipelines/inference_llm.py`.

### 3. **Pipeline C: OCR-Free Generative Parsing (Donut)**
*   **Best for**: Mobile-captured, noisy, or handwritten documents.
*   **Logic**: An end-to-end generative transformer that reads images as pixels and outputs JSON directly. Orchestrated in `pipelines/inference_donut.py`.

---

## 📂 Project Structure

```text
multimodal-doc-parser/
├── configs/            # Centralized YAML configurations (base_config.yaml)
├── data/               # Data loaders and multimodal preprocessors
├── models/             # Core architecture definitions (LayoutLM, Donut, OCR)
├── pipelines/          # High-level orchestration for each strategy
├── scripts/            # CLI tools for prediction, API deployment, and eval
├── trainer/            # Fine-tuning scripts for HF Transformers
├── utils/              # Shared logic (BBox math, Image processing)
└── evaluation/         # Benchmarking suite (Precision/Recall/F1)
```

---

## 🚀 Quick Start

### 1. Prerequisites
*   **Python**: 3.9 or higher
*   **System Dependencies**: 
    *   [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
    *   [Poppler](https://poppler.freedesktop.org/) (for PDF conversion)

### 2. Installation
```bash
git clone https://github.com/Amikumarsingh/Multimodal-Document-Parsing-System.git
cd Multimodal-Document-Parsing-System
pip install -r requirements.txt
```

### 3. Real-Time Inference
Analyze any document using your preferred pipeline from the root directory:

```bash
# Pipeline A: Rule-Based OCR
python scripts/predict_rules.py --image path/to/invoice.png

# Pipeline B: LayoutLM v3
python scripts/predict_llm.py --image path/to/receipt.jpg

# Pipeline C: Donut (OCR-Free)
python scripts/predict_donut.py --image path/to/photo.jpg

# Full System Evaluation (Benchmark all 3)
python scripts/evaluate.py --image path/to/test.png
```

### 4. Deploying the API
Launch the containerized FastAPI server:
```bash
docker build -t doc-parser-api .
docker run -p 8000:8000 doc-parser-api
```

---

## 📊 Benchmarking & Performance
The system includes a dedicated evaluation framework to compare models on accuracy and latency.

| Pipeline | F1 Score | Latency (GPU) | Robustness |
| :--- | :--- | :--- | :--- |
| **OCR + Rules** | 0.65 | ~200ms | Low |
| **LayoutLM v3** | **0.88** | **150ms** | High |
| **Donut** | 0.83 | 400ms | **Very High** |

---

## 📑 Research & Documentation
A formal research-style report detailing the methodology, experimental setup, and deeper analysis of results is available in [research_report.md](./research_report.md).

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
