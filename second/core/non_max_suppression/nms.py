"""Pure-Python/numba replacement for the compiled pybind11 `nms` extension.

The original repo builds this module from second/core/cc/nms/*.cc via nvcc +
a host C++ compiler (see second/utils/buildtools). That toolchain assumes a
Linux build environment (g++, boost) and isn't available here, so this module
reimplements the same three functions in Python with numba-jitted CPU loops
instead of compiling the C++/CUDA sources.
"""
import numba
import numpy as np


@numba.njit(cache=True)
def _nms_cpu_kernel(boxes, order, thresh, eps):
    ndets = boxes.shape[0]
    suppressed = np.zeros(ndets, dtype=np.uint8)
    area = np.empty(ndets, dtype=boxes.dtype)
    for i in range(ndets):
        area[i] = (boxes[i, 2] - boxes[i, 0] + eps) * (boxes[i, 3] - boxes[i, 1] + eps)
    keep = np.empty(ndets, dtype=np.int64)
    num_keep = 0
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep[num_keep] = i
        num_keep += 1
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(boxes[i, 0], boxes[j, 0])
            xx2 = min(boxes[i, 2], boxes[j, 2])
            w = xx2 - xx1 + eps
            if w > 0:
                yy1 = max(boxes[i, 1], boxes[j, 1])
                yy2 = min(boxes[i, 3], boxes[j, 3])
                h = yy2 - yy1 + eps
                if h > 0:
                    inter = w * h
                    ovr = inter / (area[i] + area[j] - inter)
                    if ovr >= thresh:
                        suppressed[j] = 1
    return keep[:num_keep]


def non_max_suppression_cpu(boxes, order, nms_overlap_thresh, eps=0.0):
    boxes = np.asarray(boxes)
    order = np.asarray(order, dtype=np.int64)
    keep = _nms_cpu_kernel(boxes, order, nms_overlap_thresh, eps)
    return list(keep)


def non_max_suppression(boxes, keep_out, nms_overlap_thresh, device_id=0):
    """Axis-aligned NMS on an already score-sorted `boxes` array.

    Mirrors the pybind11 signature: writes kept indices (into `boxes`, which
    the caller has already sorted by score) into `keep_out` and returns the
    number written.
    """
    order = np.arange(boxes.shape[0], dtype=np.int64)
    keep = _nms_cpu_kernel(np.asarray(boxes), order, nms_overlap_thresh, 0.0)
    num_out = len(keep)
    keep_out[:num_out] = keep
    return num_out


def rotate_non_max_suppression_cpu(box_corners, order, standup_iou, thresh):
    from shapely.geometry import Polygon

    box_corners = np.asarray(box_corners)
    order = np.asarray(order)
    standup_iou = np.asarray(standup_iou)
    ndets = box_corners.shape[0]
    suppressed = np.zeros(ndets, dtype=np.uint8)
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(int(i))
        poly_i = None
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            if standup_iou[i, j] <= 0.0:
                continue
            if poly_i is None:
                poly_i = Polygon(box_corners[i])
                if not poly_i.is_valid:
                    poly_i = poly_i.buffer(0)
            poly_j = Polygon(box_corners[j])
            if not poly_j.is_valid:
                poly_j = poly_j.buffer(0)
            inter_area = poly_i.intersection(poly_j).area
            if inter_area <= 0:
                continue
            union_area = poly_i.union(poly_j).area
            if union_area <= 0:
                continue
            overlap = inter_area / union_area
            if overlap >= thresh:
                suppressed[j] = 1
    return keep
