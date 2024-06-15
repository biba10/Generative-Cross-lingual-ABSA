from src.evaluation.f1_score import F1ScoreGeneral
from src.evaluation.f1_score_seq2seq import F1ScoreSeq2Seq


class F1ScoreTotal(F1ScoreGeneral):
    """F1 score for merging more F1 scores."""
    full_state_update = False

    def __init__(self) -> None:
        """Initialize the F1 score metric."""
        super().__init__()

    def update(self, f1_scores: list[F1ScoreSeq2Seq], ) -> None:
        """
        Update the metric with list of F1 scores. First, reset the metric. Then, update the metric with given F1 scores.

        :param f1_scores: list of F1 scores
        :return: None
        """
        self.tp = 0
        self.fp = 0
        self.fn = 0
        for f1_score in f1_scores:
            self.tp += f1_score.tp
            self.fp += f1_score.fp
            self.fn += f1_score.fn
