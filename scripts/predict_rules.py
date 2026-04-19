import argparse
import json
import os
from pipelines.inference_rules import RuleBasedPipeline

def main():
    parser = argparse.ArgumentParser(description="Pipeline A: OCR + Rule-Based Parser")
    parser.add_argument("--image", type=str, required=True, help="Path to the document image")
    parser.add_argument("--tesseract_path", type=str, help="Full path to tesseract.exe (Windows)")
    
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image path '{args.image}' does not exist.")
        return

    # Initialize Pipeline
    pipeline = RuleBasedPipeline(tesseract_path=args.tesseract_path)
    
    # Process
    print(f"Ingesting: {args.image}...")
    try:
        results = pipeline.process(args.image)
        
        # Output results
        print("\n--- Extracted Structured JSON ---")
        print(json.dumps(results, indent=4))
        
    except Exception as e:
        print(f"Execution Error: {str(e)}")
        print("\nNote: Ensure Tesseract OCR is installed on your system.")

if __name__ == "__main__":
    main()
