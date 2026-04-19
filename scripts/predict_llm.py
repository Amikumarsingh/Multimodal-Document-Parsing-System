import argparse
import json
from pipelines.inference_llm import LayoutLMPipeline

def main():
    parser = argparse.ArgumentParser(description="Pipeline B: LayoutLM v3 Project Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to document image")
    parser.add_argument("--model", type=str, default="microsoft/layoutlmv3-base", help="Model checkpoint or path")
    parser.add_argument("--tesseract_path", type=str, help="Tesseract binary path (Windows)")
    
    args = parser.parse_args()

    pipeline = LayoutLMPipeline(model_path=args.model, tesseract_path=args.tesseract_path)
    result = pipeline.process(args.image)
    
    print("\n--- Extracted JSON (LayoutLM v3) ---")
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
