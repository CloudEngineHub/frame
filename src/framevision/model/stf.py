from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from framevision import geometry as geo
from framevision.model.components import SpatioTemporalTransformer


class STF(nn.Module):
    def __init__(
        self,
        num_keypoints: int,
        time_steps: int,
        num_views: int = 2,
        undersampling_factor: int = 1,
        transform_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.time_steps = time_steps
        self.num_views = num_views
        self.undersampling_factor = undersampling_factor
        self.transform_kwargs = transform_kwargs if transform_kwargs is not None else {}
        self.transformer = SpatioTemporalTransformer(num_keypoints, num_views, time_steps, **kwargs)

    def forward(
        self,
        joints_3D_cc: Float[Tensor, "B T V J 3"],
        left2middle: Float[Tensor, "B 4 4"],
        right2middle: Float[Tensor, "B 4 4"],
        middle2world: Float[Tensor, "B T 4 4"],
        **kwargs,
    ):
        """
        Forward pass to predict 3D keypoints from a history of 3D keypoints in camera coordinates.

        Args:
            joints_3D_cc: Predictions of 3D keypoints in camera coordinates for the previous T frames.
            middle2world: VR pose tracking over the past T frames in world coordinates. This is the M frame of reference in the paper.
            left2middle: Transformation matrix from left camera to the middle/VR frame of reference. This is coming from calibration.
            right2middle: Transformation matrix from right camera to the middle/VR frame of reference. This is coming from calibration.
        """
        B, T, V, J, _ = joints_3D_cc.shape

        cams2floor, floor2world = self.compute_transformations(left2middle, right2middle, middle2world)
        joints_3D = geo.rototranslate(joints_3D_cc, cams2floor)

        joints_3D_fl = self.transformer(joints_3D)
        joints_3D_wr = geo.rototranslate(joints_3D_fl, floor2world)

        last_pred_last_step = joints_3D_wr[:, -1:]  # Shape: (B, 1, J, 3)
        return dict(joints_3D=last_pred_last_step, all_joints_3D=joints_3D_wr)

    @torch.autocast("cuda", enabled=False)
    def compute_transformations(self, left2middle, right2middle, middle2world):
        """
        Args:
            left2middle: Transformation matrix from left to middle camera frame. Shape: (B, 4, 4).
            right2middle: Transformation matrix from right to middle camera frame. Shape: (B, 4, 4).
            middle2world: Transformation matrix from middle to world frame. Shape: (B, T, 4, 4).

        Returns:
            cams2floor_last: Transformation matrix from cameras to the last floor frame. Shape: (B, T, 2, 4, 4).
            floor_last2world: Transformation matrix from the last floor frame to the world frame. Shape: (B, T, 4, 4).
        """

        # Computing the transformations from the cameras to the middle frame
        cams2middle = torch.stack([left2middle, right2middle], dim=1)  # Shape: (B, 2, 4, 4)

        # Compute the transformation from world coordinate to the last floor frame
        middle2world_last = middle2world[:, -1].unsqueeze(1)  # Shape: (B, 1, 4, 4)
        middle2floor_last = geo.compute_relpose_to_floor(middle2world_last, **self.transform_kwargs)  # Shape: (B, 1, 4, 4)
        world2floor_last = middle2floor_last @ geo.invert_SE3(middle2world_last)  # Shape: (B, 1, 4, 4)
        floor_last2world = geo.invert_SE3(world2floor_last)  # Shape: (B, 1, 4, 4)

        # Unsqueeze approriate dimension to make sure they match
        cams2middle = cams2middle.unsqueeze(1)  # Shape: (B, 1, 2, 4, 4)
        middle2world = middle2world.unsqueeze(2)  # Shape: (B, T, 1, 4, 4)

        # Compute the transformation from the cameras to world coordinates
        cams2world = middle2world @ cams2middle  # Shape: (B, T, 2, 4, 4)

        # Compute the transformation from the cameras to the last floor frame
        world2floor_last = world2floor_last.unsqueeze(2)  # Shape: (B, 1, 1, 4, 4)
        cams2floor_last = world2floor_last @ cams2world  # Shape: (B, T, 2, 4, 4)

        return cams2floor_last, floor_last2world
