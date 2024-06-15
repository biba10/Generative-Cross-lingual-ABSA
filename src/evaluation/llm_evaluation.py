def evaluate_llm(y_true: list[list[tuple | str]], y_pred: list[list[tuple | str]]) -> tuple[float, float, float]:
    """Evaluate the LLMs."""
    tp = 0
    fp = 0
    fn = 0
    # convert each list in the list of lists to a set
    y_true = [set(labels) for labels in y_true]
    y_pred = [set(labels) for labels in y_pred]

    for predictions, labels in zip(y_pred, y_true):
        for prediction in predictions:
            if prediction in labels:
                tp += 1
            else:
                fp += 1

        for label in labels:
            if label not in predictions:
                fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0
    return f1, precision, recall
