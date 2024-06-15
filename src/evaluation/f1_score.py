from abc import ABC

import torch
from torchmetrics import Metric


class F1ScoreGeneral(Metric, ABC):
    """F1 score abstract class for evaluation of models. F1 score is calculated as 2 * (precision * recall) / (precision + recall)."""

    def __init__(self) -> None:
        """Initialize the F1 score metric."""
        super().__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def calculate_precision(self) -> torch.Tensor:
        """
        Calculate precision. Precision is calculated as tp / (tp + fp).

        :return: precision
        """
        if self.tp + self.fp == 0:
            return torch.tensor(0.0)
        precision = self.tp / (self.tp + self.fp)
        return precision

    def calculate_recall(self) -> torch.Tensor:
        """
        Calculate recall. Recall is calculated as tp / (tp + fn).

        :return: recall
        """
        if self.tp + self.fn == 0:
            return torch.tensor(0.0)
        recall = self.tp / (self.tp + self.fn)
        return recall

    def compute(self) -> torch.Tensor:
        """
        Calculate F1 score. F1 score is calculated as 2 * (precision * recall) / (precision + recall).

        :return: F1 score
        """
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        if precision + recall == 0:
            return torch.tensor(0.0)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
