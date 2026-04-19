import os
import sys
import argparse
import time
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import calculate_json_similarity, print_evaluation_report
from pipelines.inference_rules import RuleBasedPipeline
from pipelines.inference_llm import LayoutLMPipeline
from pipelines.inference_donut import DonutPipeline

def main():
    parser = argparse.ArgumentParser(description="Multi-Pipeline Evaluation Framework")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--gt", type=str, help="JSON string representing ground truth")
    
    args = parser.parse_args()
    
    # Standard GT for demo if not provided
    gt = json.loads(args.gt) if args.gt else {"Date": "04/19/2026", "Invoice#": "12345"}
    
    pipelines = {
        "OCR + Rules": RuleBasedPipeline(),
        "LayoutLM v3": LayoutLMPipeline(),
        "Donut": DonutPipeline()
    }
    
    final_results = {}
    
    for name, pipe in pipelines.items():
        print(f"Evaluating {name}...")
        start_time = time.time()
        try:
            pred = pipe.process(args.image)
            latency = time.time() - start_time
            
            metrics = calculate_json_similarity(gt, pred)
            metrics["latency (s)"] = latency
            final_results[name] = metrics
        except Exception as e:
            print(f"Error in {name}: {str(e)}")
            
    print_evaluation_report(final_results)

if __name__ == "__main__":
    main()
