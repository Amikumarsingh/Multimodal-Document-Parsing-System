# Multimodal Document Parsing System

A production-grade system to compare three different approaches for extracting structured JSON from PDFs and document images.

## Architecture & Pipelines

1.  **OCR + Rule-Based**: Uses `EasyOCR` for text extraction and spatial-aware heuristics to map keys to values. Best for structured forms with fixed layouts.
2.  **LayoutLM v3**: A layout-aware transformer that combines visual and textual features. Best for complex documents like invoices and receipts.
3.  **Donut (OCR-Free)**: An end-to-end transformer that generates structured text directly from images. Best for high-accuracy parsing without intermediate OCR steps.

## Project Structure

```text
multimodal-doc-parser/
├── data/               # Data loading and preprocessing scripts
├── models/             # Pipeline implementations (OCR, LayoutLM, Donut)
├── trainer/            # Fine-tuning scripts for Transformers
├── evaluation/         # Metrics (Precision, Recall, F1)
├── configs/            # Config files for models
├── main.py             # CLI Entry point
└── requirements.txt    # Dependencies
```

## Setup Instructions

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Dataset**:
    The system automatically fetches the **FUNSD** dataset from HuggingFace. No manual download is required.

3.  **Run Inference**:
    Compare all three pipelines on a document image:
    ```bash
    python main.py --image path/to/your/document.png
    ```

4.  **Training**:
    To fine-tune models on your own data or FUNSD:
    ```bash
    python trainer/train_layoutlm.py
    python trainer/train_donut.py
    ```

## Evaluation Metrics

The system evaluates pipelines based on:
-   **Precision**: Accuracy of predicted key-value pairs.
-   **Recall**: Ability to find all required fields.
-   **F1 Score**: Harmonic mean of Precision and Recall.
-   **SeqEval**: Token-level NER metrics for LayoutLM.

## Requirements

-   Python 3.8+
-   CUDA-enabled GPU (Highly recommended for LayoutLM and Donut)
-   8GB+ RAM
