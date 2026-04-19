import argparse
from PIL import Image
from models.ocr_rules import OCRRuleBasedParser
from models.layoutlm_v3 import LayoutLMV3Wrapper
from models.donut import DonutWrapper
from evaluation.metrics import calculate_json_similarity, print_evaluation_report

def main():
    parser = argparse.ArgumentParser(description="Multimodal Document Parsing System")
    parser.add_argument("--image", type=str, required=True, help="Path to document image")
    parser.add_argument("--pipeline", type=str, choices=["all", "ocr_rules", "layoutlm", "donut"], default="all")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    
    results = {}

    # Pipeline 1: OCR + Rule-Based
    if args.pipeline in ["all", "ocr_rules"]:
        print("\n--- Running OCR + Rule-Based Pipeline ---")
        ocr_parser = OCRRuleBasedParser()
        res_ocr = ocr_parser.parse(image)
        results["OCR + Rules"] = res_ocr
        print(f"Extracted: {res_ocr}")

    # Pipeline 2: LayoutLM (Pre-trained/Mock)
    if args.pipeline in ["all", "layoutlm"]:
        print("\n--- Running LayoutLM v3 Pipeline ---")
        # In a real scenario, this would use the LayoutLM processor to get tokens
        # For simplicity in this demo, we use placeholder logic
        llm_wrapper = LayoutLMV3Wrapper()
        # Mocking values for demonstration
        res_llm = {"Date": "04/19/2026", "Invoice#": "12345"} 
        results["LayoutLM v3"] = res_llm
        print(f"Extracted: {res_llm}")

    # Pipeline 3: Donut (End-to-End)
    if args.pipeline in ["all", "donut"]:
        print("\n--- Running Donut Pipeline ---")
        donut_wrapper = DonutWrapper()
        res_donut = donut_wrapper.predict(image)
        results["Donut (OCR-free)"] = res_donut
        print(f"Extracted: {res_donut}")

    # Comparison / Final Report
    print("\n" + "="*40)
    print("FINAL COMPARISON REPORT")
    print("="*40)
    
    # Mock Ground Truth for comparison demonstration
    gt = {"Date": "04/19/2026", "Invoice#": "12345", "Total": "$100.00"}
    
    eval_results = {}
    for name, pred in results.items():
        eval_results[name] = calculate_json_similarity(gt, pred)
        
    print_evaluation_report(eval_results)

if __name__ == "__main__":
    main()
