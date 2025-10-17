"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()
        self.pos_dim = pos_dim
        self.view_dir_dim = view_dir_sim

        self.positional_dims = 60
        self.linear_layers = nn.ModuleList(
                [nn.Linear(self.positional_dims, feat_dims)]+
                [nn.Linear(feat_dims+self.positional_dims if i==4 else feat_dim, feat_dim) for i in range(8)])
        self.density_prediction = nn.Linear(feat_dim, 1)
        self.bottle_neck = nn.Linear(feat_dim + view_dir_dims, 128)
        self.rgb = nn.Linear(128, 3)


    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """
        x = pos

        for i in range(9): 
            x = self.linear_layers(x)
            x = F.ReLU(x)

            if i==4:
                x = torch.cat([x, pos], -1) 

        sigma = self.density_prediction(x)
        x = torch.cat([x, view_dir], -1)
        x = F.ReLU(self.bottle_neck(x))
        rgb = self.rgb(x) 
        return sigma, rgb
        
            

