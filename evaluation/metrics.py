from sklearn.metrics import precision_score, recall_score, f1_score
import json

def calculate_json_similarity(gt_json, pred_json):
    """
    Calculates precision, recall, and F1 for key-value extraction.
    Treats each (key, value) pair as an entity.
    """
    gt_pairs = set(gt_json.items())
    pred_pairs = set(pred_json.items())
    
    tp = len(gt_pairs.intersection(pred_pairs))
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def print_evaluation_report(results_dict):
    """
    Prints a formatted table of results for multiple pipelines.
    results_dict: { "Pipeline Name": { "precision": 0.9, ... }, ... }
    """
    print(f"{'Pipeline':<25} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 65)
    for name, metrics in results_dict.items():
        print(f"{name:<25} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1']:<10.4f}")
