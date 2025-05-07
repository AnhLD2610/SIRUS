from __future__ import annotations
from typing import Any, Iterable
import torch
from torch import Tensor, nn
import numpy as np 
from enum import Enum
from torch.nn import functional as F


def _convert_to_tensor(a: list | np.ndarray | Tensor):
    if not isinstance(a, Tensor):
        a = torch.tensor(a)
    return a

def _convert_to_batch(a: Tensor):
    if a.dim() == 1:
        a = a.unsqueeze(0)
    return a


def _convert_to_batch_tensor(a: list | np.ndarray | Tensor):
    a = _convert_to_tensor(a)
    a = _convert_to_batch(a)
    return a

def normalize_embeddings(embeddings: Tensor):
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)

def cos_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor):
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

class TripletDistanceMetric(Enum):
    """The metric for the triplet loss"""

    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)


class TripletLoss(nn.Module):
    def __init__(
        # self, distance_metric=TripletDistanceMetric.COSINE, triplet_margin: float = 1.0
        self, distance_metric=TripletDistanceMetric.COSINE, triplet_margin: float = 1.0

    ) -> None:
        """
        This class implements triplet loss. Given a triplet of (anchor, positive, negative),
        the loss minimizes the distance between anchor and positive while it maximizes the distance
        between anchor and negative. It compute the following loss function:

        ``loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0)``.

        Margin is an important hyperparameter and needs to be tuned respectively.

        Args:
            model: SentenceTransformerModel
            distance_metric: Function to compute distance between two
                embeddings. The class TripletDistanceMetric contains
                common distance metrices that can be used.
            triplet_margin: The negative should be at least this much
                further away from the anchor than the positive.

        """
        super().__init__()
        # self.model = model
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin

    def forward(self, rep_anchor, rep_pos, rep_neg) -> Tensor:

        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)
    
        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
       
        # losses = torch.where(losses == self.triplet_margin, torch.tensor(0.0, device=losses.device), losses)

        return losses.mean()
