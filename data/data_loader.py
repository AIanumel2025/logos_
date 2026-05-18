from datasets import load_dataset

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

def load_tweet_eval():
    """
    Loads the TweetEval sentiment dataset from Hugging Face.
    """
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    return dataset


def get_splits():
    """
    Returns dataset splits (train/validation/test).
    """
    dataset = load_tweet_eval()
    return dataset.keys()


def get_sample(split="train", index=0):
    """
    Returns a sample tweet from a given split.
    """
    dataset = load_tweet_eval()
    return dataset[split][index]


def get_dataset_shape():
    """
    Returns dataset structure information.
    """
    dataset = load_tweet_eval()
    return {k: len(dataset[k]) for k in dataset.keys()}
