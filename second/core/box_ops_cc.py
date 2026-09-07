"""Pure-Python replacement for the compiled pybind11 `box_ops_cc` extension.

Same rationale as core/non_max_suppression/nms.py: the original module is
built from second/core/cc/box_ops.cc via nvcc + a host C++ compiler, which
isn't available in this environment. shapely (already a dependency of this
repo) gives an equivalent rotated-polygon intersection/union.
"""
import numpy as np
from shapely.geometry import Polygon


def rbbox_iou(box_corners, qbox_corners, standup_iou, standup_thresh=0.0):
    box_corners = np.asarray(box_corners)
    qbox_corners = np.asarray(qbox_corners)
    standup_iou = np.asarray(standup_iou)
    n = box_corners.shape[0]
    k = qbox_corners.shape[0]
    overlaps = np.zeros((n, k), dtype=box_corners.dtype)
    if n == 0 or k == 0:
        return overlaps
    polys = [None] * n
    qpolys = [None] * k
    for i in range(n):
        for j in range(k):
            if standup_iou[i, j] <= standup_thresh:
                continue
            if polys[i] is None:
                p = Polygon(box_corners[i])
                polys[i] = p if p.is_valid else p.buffer(0)
            if qpolys[j] is None:
                q = Polygon(qbox_corners[j])
                qpolys[j] = q if q.is_valid else q.buffer(0)
            inter_area = polys[i].intersection(qpolys[j]).area
            if inter_area <= 0:
                continue
            union_area = polys[i].union(qpolys[j]).area
            if union_area <= 0:
                continue
            overlaps[i, j] = inter_area / union_area
    return overlaps
