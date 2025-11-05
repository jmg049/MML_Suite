from typing import Callable, Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MIEstimator(nn.Module):
    def __init__(self, input_dims: List[int], z_dim: int):
        super().__init__()
        total_input_dim = sum(input_dims) + z_dim
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, inputs: List[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat(inputs + [z], dim=1))


class CMAMLoss(nn.Module):
    def __init__(
        self,
        x_dims: int | list[int] = 0,
        z_dim: int = 0,
        cosine_weight: float = 0.0,
        mae_weight: float = 0.0,
        mse_weight: float = 1.0,
        rec_weight: float = 1.0,
        cls_weight: float = 0.005,
        mmd_weight: float = 0.0,
        moment_weight: float = 0.0,
        cyclic_weight: float = 0.0,
        mi_weight: float = 0.0,
        num_moments: int = 2,
        mmd_sigma: float = 1.0,
        maximize_cosine: bool = True,
        epsilon: float = 1e-8,
        huber_weight: float = 0.0,
        cls_loss_type: Literal["ce", "bce", "mse"] = "ce",
        num_classes: Optional[int] = None,
    ):
        """
        Initialize the CMAMLoss module.

        Args:
            x_dim (int): Dimension of input features.
            z_dim (int): Dimension of output features.
            cosine_weight (float): Weight for cosine similarity loss.
            mae_weight (float): Weight for mean absolute error loss.
            mse_weight (float): Weight for mean squared error loss.
            rec_weight (float): Weight for reconstruction loss.
            cls_weight (float): Weight for classification loss.
            mmd_weight (float): Weight for maximum mean discrepancy loss.
            moment_weight (float): Weight for moment matching loss.
            cyclic_weight (float): Weight for cyclic consistency loss.
            mi_weight (float): Weight for mutual information loss.
            num_moments (int): Number of moments to match in moment matching loss.
            mmd_sigma (float): Sigma parameter for Gaussian kernel in MMD loss.
            maximize_cosine (bool): Whether to maximize or minimize cosine similarity.
            epsilon (float): Small value to avoid division by zero.
            cls_loss_type (str): Type of classification loss ('ce', 'bce', or 'mse').
            num_classes (Optional[int]): Number of classes for classification task.
        """
        super(CMAMLoss, self).__init__()


        self.cosine_weight = cosine_weight
        self.mae_weight = mae_weight
        self.mse_weight = mse_weight
        self.rec_weight = rec_weight
        self.mmd_weight = mmd_weight
        self.moment_weight = moment_weight
        self.cyclic_weight = cyclic_weight
        self.mi_weight = mi_weight
        self.cls_weight = cls_weight
        self.num_moments = num_moments
        self.mmd_sigma = mmd_sigma
        self.maximize_cosine = maximize_cosine
        self.epsilon = epsilon
        self.cls_loss_type = cls_loss_type
        self.huber_weight = huber_weight
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=epsilon)
        self.mae_loss = nn.L1Loss(reduction="mean")
        self.mse_loss = nn.MSELoss(reduction="mean")

        if mi_weight > 0:
            self.mi_estimator = MIEstimator(x_dims, z_dim)

        cls_loss_type = cls_loss_type.lower()
        if cls_weight > 0:
            if cls_loss_type == "ce":
                self.cls_loss = nn.CrossEntropyLoss()
            elif cls_loss_type == "bce":
                self.cls_loss = nn.BCEWithLogitsLoss()
            elif cls_loss_type == "mse":
                self.cls_loss = nn.MSELoss()
            else:
                raise ValueError(f"Unsupported classification loss type: {cls_loss_type}")
        print(f"{self}")

    def __str__(self) -> str:
        return (
            f"CMAMLoss(\n"
            f"  cosine_weight={self.cosine_weight:.3f},\n"
            f"  mae_weight={self.mae_weight:.3f},\n"
            f"  mse_weight={self.mse_weight:.3f},\n"
            f"  rec_weight={self.rec_weight:.3f},\n"
            f"  cls_weight={self.cls_weight:.3f},\n"
            f"  mmd_weight={self.mmd_weight:.3f},\n"
            f"  moment_weight={self.moment_weight:.3f},\n"
            f"  cyclic_weight={self.cyclic_weight:.3f},\n"
            f"  mi_weight={self.mi_weight:.3f},\n"
            f"  cls_loss_type='{self.cls_loss_type}'\n"
            f")"
        )

    def gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        Compute the Gaussian kernel between two tensors.

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.
            sigma (float): Sigma parameter for Gaussian kernel.

        Returns:
            torch.Tensor: Gaussian kernel matrix.
        """
        dist = torch.cdist(x, y, p=2) ** 2

        return torch.exp(-dist / (2 * sigma**2))

    def mmd_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Maximum Mean Discrepancy (MMD) loss between two tensors.

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: MMD loss value.
        """
        xx = self.gaussian_kernel(x, x, self.mmd_sigma)
        yy = self.gaussian_kernel(y, y, self.mmd_sigma)
        xy = self.gaussian_kernel(x, y, self.mmd_sigma)
        return xx.mean() + yy.mean() - 2 * xy.mean()

    def moment_matching_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the moment matching loss between two tensors.

        Args:
            x (torch.Tensor): First input tensor.
            y (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Moment matching loss value.
        """
        loss = 0
        for i in range(1, self.num_moments + 1):
            x_moment = torch.mean(torch.pow(x, i), dim=0)
            y_moment = torch.mean(torch.pow(y, i), dim=0)
            loss += torch.mean((x_moment - y_moment) ** 2)
        return loss

    def cyclic_consistency_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        forward_func: Callable,
    ) -> torch.Tensor:
        """
        Compute the cyclic consistency loss between original and reconstructed tensors.

        This loss ensures that when we apply the forward function to the reconstructed tensor,
        we should get back something close to the original tensor.

        Args:
            original (torch.Tensor): Original input tensor.
            reconstructed (torch.Tensor): Reconstructed tensor.
            forward_func (Callable): The forward function of the model that maps from the reconstructed space back to the original space.

        Returns:
            torch.Tensor: Cyclic consistency loss value.
        """
        # Apply the forward function to the reconstructed tensor
        cyclic_reconstruction = forward_func(reconstructed)

        # Compute the loss between the original and the cyclic reconstruction
        loss = F.mse_loss(cyclic_reconstruction, original)

        return loss

    def mutual_information_loss(self, inputs: List[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        """
        Compute the mutual information loss between input modalities and latent representations.

        Args:
            inputs (List[torch.Tensor]): List of input tensors for each modality.
            z (torch.Tensor): Latent representation tensor.

        Returns:
            torch.Tensor: Mutual information loss value.
        """
        pos_samples = self.mi_estimator(inputs, z)
        neg_samples = self.mi_estimator(inputs, z[torch.randperm(z.shape[0])])
        return -torch.mean(pos_samples) + torch.log(torch.mean(torch.exp(neg_samples)))

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        originals: Optional[torch.Tensor] = None,
        reconstructed: Optional[torch.Tensor] = None,
        forward_func: Optional[Callable] = None,
        cls_logits: Optional[list[torch.Tensor] | torch.Tensor] = None,
        cls_labels: Optional[list[torch.Tensor] | torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss and individual loss components.

        Args:
            predictions (torch.Tensor): Predicted tensor.
            targets (torch.Tensor): Target tensor.
            original (Optional[torch.Tensor]): Original input tensor for cyclic consistency.
            reconstructed (Optional[torch.Tensor]): Reconstructed tensor for cyclic consistency.
            forward_func (Optional[Callable]): Forward function for cyclic consistency.
            cls_logits (Optional[torch.Tensor]): Classification logits.
            cls_labels (Optional[torch.Tensor]): Classification labels.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing total loss and individual loss components.
        """


        cosine_sim = self.cosine_similarity(predictions, targets).mean()
        # cosine_loss = -cosine_sim if self.maximize_cosine else cosine_sim
        cosine_loss = (1 - cosine_sim) * self.cosine_weight
        mae = self.mae_loss(predictions, targets) * self.mae_weight
        mse = self.mse_loss(predictions, targets) * self.mse_weight

        total_loss = cosine_loss + mae + mse

        loss_dict = {
            "cosine": cosine_loss,
            "mae": mae,
            "mse": mse,
        }

        if self.mmd_weight > 0:

            mmd = self.mmd_loss(predictions, targets)
            total_loss += self.mmd_weight * mmd
            loss_dict["mmd"] = mmd

        if self.moment_weight > 0:
            moment_loss = self.moment_matching_loss(predictions, targets)
            total_loss += self.moment_weight * moment_loss
            loss_dict["moment_loss"] = moment_loss

        if self.cyclic_weight > 0 and originals is not None and reconstructed is not None and forward_func is not None:
            cyclic_loss = self.cyclic_consistency_loss(originals, reconstructed, forward_func)
            total_loss += self.cyclic_weight * cyclic_loss
            loss_dict["cyclic_loss"] = cyclic_loss

        if self.mi_weight > 0 and originals is not None:
            mi_loss = self.mutual_information_loss(originals, predictions)
            total_loss += self.mi_weight * mi_loss
            loss_dict["mi_loss"] = mi_loss

        if self.cls_weight > 0 and cls_logits is not None and cls_labels is not None:
            cls_loss = self.cls_loss(cls_logits, cls_labels)
            total_loss += self.cls_weight * cls_loss
            loss_dict["cls_loss"] = cls_loss

        if self.huber_weight > 0:
            huber_loss = self.huber_loss(predictions=predictions, targets=targets)
            total_loss += self.huber_weight * huber_loss
            loss_dict["huber"] = huber_loss

        loss_dict["total_loss"] = total_loss
        return loss_dict
    
    def __str__(self) -> str:
        """
        Return a string representation of the CMAMLoss configuration.
        
        Returns:
            str: A formatted string showing active loss components and their weights.
        """
        active_components = []
        
        # Check each weight and add to active components if non-zero
        if self.mse_weight > 0:
            active_components.append(f"MSE(w={self.mse_weight:.3f})")
        
        if self.mae_weight > 0:
            active_components.append(f"MAE(w={self.mae_weight:.3f})")
        
        if self.cosine_weight > 0:
            direction = "max" if self.maximize_cosine else "min"
            active_components.append(f"Cosine(w={self.cosine_weight:.3f}, {direction})")
        
        if self.mmd_weight > 0:
            active_components.append(f"MMD(w={self.mmd_weight:.3f}, σ={self.mmd_sigma:.1f})")
        
        if self.moment_weight > 0:
            active_components.append(f"Moment(w={self.moment_weight:.3f}, n={self.num_moments})")
        
        if self.cyclic_weight > 0:
            active_components.append(f"Cyclic(w={self.cyclic_weight:.3f})")
        
        if self.mi_weight > 0:
            active_components.append(f"MI(w={self.mi_weight:.3f})")
        
        if self.cls_weight > 0:
            active_components.append(f"Class(w={self.cls_weight:.3f}, {self.cls_loss_type})")
        
        # Join all active components with ' + '
        loss_str = " + ".join(active_components) if active_components else "No active loss components"
        
        return f"CMAMLoss({loss_str})"
    

    def to_latex(self) -> str:
        """
        Generate a LaTeX representation of the loss function with symbolic coefficients and specific inputs.

        Returns:
            str: LaTeX string representing the loss function.
        """
        terms = []

        # Define the mapping of weight attributes to LaTeX subscript names
        loss_terms = [
            ("cosine_weight", "Cos"),
            ("mae_weight", "MAE"),
            ("mse_weight", "MSE"),
            ("mmd_weight", "MMD"),
            ("moment_weight", "moment"),
            ("cyclic_weight", "cyclic"),
            ("mi_weight", "MI"),
            ("cls_weight", "Cls"),
            ("huber_weight", "Huber")
        ]

        # Define the mapping of loss term names to their specific inputs
        inputs_mapping = {
            "Cos": (r"\hat{f}", "f"),
            "MAE": (r"\hat{f}", "f"),
            "MSE": (r"\hat{f}", "f"),
            "MMD": (r"\hat{f}", "f"),
            "moment": (r"\hat{f}", "f"),
            "cyclic": (r"\hat{f}", "f"),
            "MI": (r"\hat{f}", "f"),
            "Cls": (r"\hat{y}", "y"),
            "Huber": (r"\hat{f}", "f")
        }

        for weight_attr, term_name in loss_terms:
            # Retrieve the weight value; default to 0 if attribute doesn't exist
            weight = getattr(self, weight_attr, 0)
            if weight > 0:
                # Get inputs for the term
                inputs = inputs_mapping.get(term_name, (r"x", "x"))  # Default inputs if not specified

                # Construct the LaTeX term with \lambda_{term} * \mathcal{L}_{term}(inputs)
                # Using raw f-strings to handle backslashes correctly
                term = rf"\lambda_{{\text{{{term_name}}}}} \mathcal{{L}}_{{\text{{{term_name}}}}}({inputs[0]}, {inputs[1]})"
                terms.append(term)

        # Combine all active terms into the total loss equation
        latex = r"\mathcal{L}_{\text{total}} = " + " + ".join(terms)

        return latex

    def huber_loss(self, predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """
        Compute Huber loss between predictions and targets.
        
        Args:
            predictions (torch.Tensor): Predicted tensor.
            targets (torch.Tensor): Target tensor.
            delta (float): Threshold parameter for Huber loss.
            
        Returns:
            torch.Tensor: Huber loss value.
        """
        # Calculate element-wise absolute difference
        abs_diff = torch.abs(predictions - targets)
        
        # Apply quadratic part for small differences (< delta)
        quadratic_part = torch.min(abs_diff, torch.tensor(delta)) ** 2 / 2
        
        # Apply linear part for large differences (>= delta)
        linear_part = abs_diff - torch.tensor(delta) / 2
        linear_part = torch.max(linear_part, torch.tensor(0.0))
        
        # Combine parts
        loss = quadratic_part + linear_part
        
        return loss.mean()