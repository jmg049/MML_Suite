from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveTripletLoss(nn.Module):
    def __init__(
        self,
        margin: float = 1.0,
        use_adaptive_margin: bool = True,
        distance_function: str = "euclidean",
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        """
        Contrastive triplet loss with adaptive margin for class-aware embedding learning.
        
        Inspired by SCOREQ paper's adaptive margin approach, adapted for discrete class labels
        instead of continuous MOS scores.
        
        Args:
            margin: Base margin for triplet loss
            use_adaptive_margin: Whether to use adaptive margin based on class differences
            distance_function: Distance metric ("euclidean" or "cosine")
            reduction: Reduction method ("mean", "sum", or "none")
            eps: Small value for numerical stability
        """
        super().__init__()
        self.margin = margin
        self.use_adaptive_margin = use_adaptive_margin
        self.distance_function = distance_function
        self.reduction = reduction
        self.eps = eps
        
    def compute_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between embeddings."""
        if self.distance_function == "euclidean":
            return torch.cdist(x, y, p=2)
        elif self.distance_function == "cosine":
            x_norm = F.normalize(x, p=2, dim=1)
            y_norm = F.normalize(y, p=2, dim=1)
            return 1 - torch.mm(x_norm, y_norm.t())
        else:
            raise ValueError(f"Unsupported distance function: {self.distance_function}")
    
    def create_triplet_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Create mask for valid triplets based on class labels.
        
        A valid triplet (i, j, k) satisfies:
        - labels[i] == labels[j] (positive pair)
        - labels[i] != labels[k] (negative pair)
        - i != j != k (distinct indices)
        
        Args:
            labels: Class labels [batch_size]
            
        Returns:
            mask: Boolean tensor [batch_size, batch_size, batch_size]
        """
        batch_size = labels.size(0)
        device = labels.device
        
        # Create pairwise label comparison matrices
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # [batch_size, batch_size]
        labels_not_equal = ~labels_equal
        
        # Create index inequality masks
        indices = torch.arange(batch_size, device=device)
        indices_i = indices.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
        indices_j = indices.unsqueeze(0).unsqueeze(2)  # [1, batch_size, 1]  
        indices_k = indices.unsqueeze(0).unsqueeze(1)  # [1, 1, batch_size]
        
        i_not_equal_j = indices_i != indices_j  # [batch_size, batch_size, 1]
        i_not_equal_k = indices_i != indices_k  # [batch_size, 1, batch_size]
        j_not_equal_k = indices_j != indices_k  # [1, batch_size, batch_size]
        
        # Expand for triplet comparisons
        # For triplet (i, j, k): labels[i] == labels[j] and labels[i] != labels[k]
        positive_mask = labels_equal.unsqueeze(2)  # [batch_size, batch_size, 1]
        negative_mask = labels_not_equal.unsqueeze(1)  # [batch_size, 1, batch_size]
        
        # Combine all constraints
        triplet_mask = (
            positive_mask &  # labels[i] == labels[j]
            negative_mask &  # labels[i] != labels[k]
            i_not_equal_j &  # i != j
            i_not_equal_k &  # i != k
            j_not_equal_k    # j != k
        )
        
        return triplet_mask
    
    def compute_adaptive_margin(
        self, 
        labels: torch.Tensor, 
        triplet_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive margin based on class label differences.
        
        Adapts SCOREQ's adaptive margin from continuous MOS to discrete classes:
        adaptive_margin = (class_diff_ij - class_diff_ik) / (num_classes - 1)
        
        For discrete classes, we use class distance (0 for same class, 1 for different classes).
        
        Args:
            labels: Class labels [batch_size]
            triplet_mask: Valid triplet mask [batch_size, batch_size, batch_size]
            
        Returns:
            adaptive_margins: Adaptive margins for each triplet [batch_size, batch_size, batch_size]
        """
        num_classes = labels.max().item() + 1
        
        # Compute class differences
        # For same class: distance = 0, for different class: distance = 1
        labels_expanded_i = labels.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
        labels_expanded_j = labels.unsqueeze(0).unsqueeze(2)  # [1, batch_size, 1]
        labels_expanded_k = labels.unsqueeze(0).unsqueeze(1)  # [1, 1, batch_size]
        
        class_diff_ij = (labels_expanded_i != labels_expanded_j).float()  # Should be 0 for valid triplets
        class_diff_ik = (labels_expanded_i != labels_expanded_k).float()  # Should be 1 for valid triplets
        
        # Adaptive margin: normalized difference in class distances
        adaptive_margin = (class_diff_ij - class_diff_ik) / max(num_classes - 1, 1)
        
        # Only compute for valid triplets
        adaptive_margin = adaptive_margin * triplet_mask.float()
        
        return adaptive_margin
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive triplet loss.
        
        Args:
            embeddings: Embedding vectors [batch_size, embedding_dim]
            labels: Class labels [batch_size]
            
        Returns:
            Dictionary with loss components
        """
        batch_size = embeddings.size(0)
        
        if batch_size < 3:
            # Need at least 3 samples for triplet loss
            return {
                "contrastive_loss": torch.tensor(0.0, device=embeddings.device, requires_grad=True),
                "num_triplets": torch.tensor(0),
                "num_valid_triplets": torch.tensor(0),
            }
        
        # Create triplet mask for valid triplets
        triplet_mask = self.create_triplet_mask(labels)
        num_valid_triplets = triplet_mask.sum()
        
        if num_valid_triplets == 0:
            # No valid triplets in this batch
            return {
                "contrastive_loss": torch.tensor(0.0, device=embeddings.device, requires_grad=True),
                "num_triplets": torch.tensor(batch_size ** 3),
                "num_valid_triplets": torch.tensor(0),
            }
        
        # Compute pairwise distances
        distances = self.compute_distance(embeddings, embeddings)
        
        # Extract distances for triplets
        # distances[i, j] = distance between embedding i and embedding j
        positive_distances = distances.unsqueeze(2)  # [batch_size, batch_size, 1]
        negative_distances = distances.unsqueeze(1)  # [batch_size, 1, batch_size]
        
        # Compute base triplet loss
        if self.use_adaptive_margin:
            adaptive_margins = self.compute_adaptive_margin(labels, triplet_mask)
            margin_term = self.margin + adaptive_margins
        else:
            margin_term = self.margin
        
        # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
        triplet_loss = F.relu(positive_distances - negative_distances + margin_term)
        
        # Apply mask to only consider valid triplets
        triplet_loss = triplet_loss * triplet_mask.float()
        
        # Reduce loss
        if self.reduction == "mean":
            if num_valid_triplets > 0:
                loss = triplet_loss.sum() / num_valid_triplets
            else:
                loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        elif self.reduction == "sum":
            loss = triplet_loss.sum()
        else:  # no reduction
            loss = triplet_loss
        
        return {
            "contrastive_loss": loss,
            "num_triplets": torch.tensor(batch_size ** 3),
            "num_valid_triplets": num_valid_triplets,
            "mean_positive_distance": (positive_distances * triplet_mask.float()).sum() / max(num_valid_triplets, 1),
            "mean_negative_distance": (negative_distances * triplet_mask.float()).sum() / max(num_valid_triplets, 1),
        }


class CombinedMSEContrastiveLoss(nn.Module):
    def __init__(
        self,
        mse_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        margin: float = 1.0,
        use_adaptive_margin: bool = True,
        distance_function: str = "euclidean",
        reduction: str = "mean",
    ):
        """
        Combined MSE and Contrastive Triplet Loss for C-MAM training.
        
        Combines structural reconstruction loss (MSE) with class discrimination loss (contrastive).
        
        Args:
            mse_weight: Weight for MSE reconstruction loss
            contrastive_weight: Weight for contrastive triplet loss
            margin: Base margin for triplet loss
            use_adaptive_margin: Whether to use adaptive margin
            distance_function: Distance metric for contrastive loss
            reduction: Reduction method
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.contrastive_weight = contrastive_weight
        
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.contrastive_loss = ContrastiveTripletLoss(
            margin=margin,
            use_adaptive_margin=use_adaptive_margin,
            distance_function=distance_function,
            reduction=reduction,
        )
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Predicted embeddings [batch_size, embedding_dim]
            targets: Target embeddings [batch_size, embedding_dim]
            labels: Class labels [batch_size]
            
        Returns:
            Dictionary with loss components
        """
        # MSE reconstruction loss
        mse = self.mse_loss(predictions, targets)
        
        # Contrastive triplet loss
        contrastive_dict = self.contrastive_loss(predictions, labels)
        contrastive = contrastive_dict["contrastive_loss"]
        
        # Combined loss
        total_loss = self.mse_weight * mse + self.contrastive_weight * contrastive
        
        return {
            "total_loss": total_loss,
            "mse_loss": mse,
            "contrastive_loss": contrastive,
            "mse_weighted": self.mse_weight * mse,
            "contrastive_weighted": self.contrastive_weight * contrastive,
            "num_triplets": contrastive_dict["num_triplets"],
            "num_valid_triplets": contrastive_dict["num_valid_triplets"],
            "mean_positive_distance": contrastive_dict.get("mean_positive_distance", torch.tensor(0.0)),
            "mean_negative_distance": contrastive_dict.get("mean_negative_distance", torch.tensor(0.0)),
        }