from src.evaluation.f1_score import F1ScoreGeneral


class F1ScoreSeq2Seq(F1ScoreGeneral):
    """F1 score for evaluation of Seq2Seq models."""
    full_state_update = False

    def __init__(self) -> None:
        """Initialize the F1 score metric."""
        super().__init__()

    def update(
            self,
            predictions: set[str] | set[tuple[str, str]] | set[tuple[str, str, str]],
            labels: set[str] | set[tuple[str, str]] | set[tuple[str, str, str]],
    ) -> None:
        """
        Update the metric with given predictions and labels.

        :param predictions: predictions
        :param labels: labels
        :return: None
        """
        for prediction in predictions:
            if prediction in labels:
                self.tp += 1
            else:
                self.fp += 1

        for label in labels:
            if label not in predictions:
                self.fn += 1
