from datasets import load_dataset

def load_funsd(split="train"):
    """
    Loads the FUNSD dataset from HuggingFace.
    """
    dataset = load_dataset("nielsr/funsd", split=split)
    return dataset

def get_labels():
    """
    Returns the list of NER labels used in FUNSD.
    """
    return ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]

if __name__ == "__main__":
    train_ds = load_funsd("train")
    print(f"Loaded {len(train_ds)} samples from training set.")
    print(f"Sample features: {train_ds.features.keys()}")
    print(f"Labels: {get_labels()}")
