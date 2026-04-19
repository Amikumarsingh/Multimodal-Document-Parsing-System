import os
import sys
import argparse
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.inference_donut import DonutPipeline

def main():
    parser = argparse.ArgumentParser(description="Pipeline C: Donut Project Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to document image")
    parser.add_argument("--model", type=str, default="naver-clova-ix/donut-base", help="Model checkpoint or path")
    
    args = parser.parse_args()

    pipeline = DonutPipeline(model_path=args.model)
    result = pipeline.process(args.image)
    
    print("\n--- Extracted JSON (Donut) ---")
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
