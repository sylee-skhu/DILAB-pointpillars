# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Region Similarity Calculators for BoxLists.

Region Similarity Calculators compare a pairwise measure of similarity
between the boxes in two BoxLists.
"""
from abc import ABCMeta
from abc import abstractmethod

import numpy as np

from second.core import box_np_ops

class RegionSimilarityCalculator(object):
  """Abstract base class for 2d region similarity calculator."""
  __metaclass__ = ABCMeta

  def compare(self, boxes1, boxes2):
    """Computes matrix of pairwise similarity between BoxLists.

    This op (to be overriden) computes a measure of pairwise similarity between
    the boxes in the given BoxLists. Higher values indicate more similarity.

    Note that this method simply measures similarity and does not explicitly
    perform a matching.

    Args:
      boxes1: [N, 5] [x,y,w,l,r] tensor.
      boxes2: [M, 5] [x,y,w,l,r] tensor.

    Returns:
      a (float32) tensor of shape [N, M] with pairwise similarity score.
    """
    return self._compare(boxes1, boxes2)

  @abstractmethod
  def _compare(self, boxes1, boxes2):
    pass


class RotateIouSimilarity(RegionSimilarityCalculator):
  """Class to compute similarity based on Intersection over Union (IOU) metric.

  This class computes pairwise similarity between two BoxLists based on IOU.
  """

  def _compare(self, boxes1, boxes2):
    """Compute pairwise IOU similarity between the two BoxLists.

    Args:
      boxlist1: BoxList holding N boxes.
      boxlist2: BoxList holding M boxes.

    Returns:
      A tensor with shape [N, M] representing pairwise iou scores.
    """

    return box_np_ops.riou_cc(boxes1, boxes2)


class NearestIouSimilarity(RegionSimilarityCalculator):
  """Class to compute similarity based on the squared distance metric.

  This class computes pairwise similarity between two BoxLists based on the
  negative squared distance metric.
  """

  def _compare(self, boxes1, boxes2):
    """Compute matrix of (negated) sq distances.

    Args:
      boxlist1: BoxList holding N boxes.
      boxlist2: BoxList holding M boxes.

    Returns:
      A tensor with shape [N, M] representing negated pairwise squared distance.
    """
    boxes1_bv = box_np_ops.rbbox2d_to_near_bbox(boxes1)
    boxes2_bv = box_np_ops.rbbox2d_to_near_bbox(boxes2)
    ret = box_np_ops.iou_jit(boxes1_bv, boxes2_bv, eps=0.0)
    return ret


class PlaneIouSimilarity(RegionSimilarityCalculator):
  """Axis-aligned IOU similarity for a single-plane (no-rotation) box
  representation, e.g. the XZ/YZ pillar-axis-ablation branch. There's no
  rotation to snap to a nearest axis (unlike `NearestIouSimilarity`) -- a
  box projected onto XZ/YZ is already axis-aligned by construction (see
  `box_np_ops.project_box3d_to_xz`), so this converts (u, v, proj_w, h)
  straight to a min/max box and calls the same axis-aligned `iou_jit` used
  elsewhere.
  """

  def _compare(self, boxes1, boxes2):
    """Args:
      boxes1, boxes2: [N, 4] / [M, 4] (u, v, proj_w, h) tensors.

    Returns:
      A tensor with shape [N, M] representing pairwise iou scores.
    """
    boxes1_minmax = _plane_box_to_minmax(boxes1)
    boxes2_minmax = _plane_box_to_minmax(boxes2)
    return box_np_ops.iou_jit(boxes1_minmax, boxes2_minmax, eps=0.0)


def _plane_box_to_minmax(boxes):
  u, v, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
  return np.stack(
      [u - w / 2, v - h / 2, u + w / 2, v + h / 2], axis=-1)


class DistanceSimilarity(RegionSimilarityCalculator):
  """Class to compute similarity based on Intersection over Area (IOA) metric.

  This class computes pairwise similarity between two BoxLists based on their
  pairwise intersections divided by the areas of second BoxLists.
  """
  def __init__(self, distance_norm, with_rotation=False, rotation_alpha=0.5):
      self._distance_norm = distance_norm
      self._with_rotation = with_rotation
      self._rotation_alpha = rotation_alpha

  def _compare(self, boxes1, boxes2):
    """Compute matrix of (negated) sq distances.

    Args:
      boxlist1: BoxList holding N boxes.
      boxlist2: BoxList holding M boxes.

    Returns:
      A tensor with shape [N, M] representing negated pairwise squared distance.
    """
    return box_np_ops.distance_similarity(
        boxes1[..., [0, 1, -1]], 
        boxes2[..., [0, 1, -1]], 
        dist_norm=self._distance_norm,
        with_rotation=self._with_rotation,
        rot_alpha=self._rotation_alpha)
