import numpy as np
from second.core import box_np_ops


class AnchorGeneratorStride:
    def __init__(self,
                 sizes=[1.6, 3.9, 1.56],
                 anchor_strides=[0.4, 0.4, 1.0],
                 anchor_offsets=[0.2, -39.8, -1.78],
                 rotations=[0, np.pi / 2],
                 class_id=None,
                 match_threshold=-1,
                 unmatch_threshold=-1,
                 dtype=np.float32):
        self._sizes = sizes
        self._anchor_strides = anchor_strides
        self._anchor_offsets = anchor_offsets
        self._rotations = rotations
        self._dtype = dtype
        self._class_id = class_id
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold

    @property
    def class_id(self):
        return self._class_id

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    def generate(self, feature_map_size):
        return box_np_ops.create_anchors_3d_stride(
            feature_map_size, self._sizes, self._anchor_strides,
            self._anchor_offsets, self._rotations, self._dtype)

class AnchorGeneratorRange:
    def __init__(self,
                 anchor_ranges,
                 sizes=[1.6, 3.9, 1.56],
                 rotations=[0, np.pi / 2],
                 class_id=None,
                 match_threshold=-1,
                 unmatch_threshold=-1,
                 dtype=np.float32):
        self._sizes = sizes
        self._anchor_ranges = anchor_ranges
        self._rotations = rotations
        self._dtype = dtype
        self._class_id = class_id
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold

    @property
    def class_id(self):
        return self._class_id

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    def generate(self, feature_map_size):
        return box_np_ops.create_anchors_3d_range(
            feature_map_size, self._anchor_ranges, self._sizes,
            self._rotations, self._dtype)


class AnchorGeneratorPlaneRange:
    """Native-resolution anchors for a single-plane (no-rotation) branch,
    e.g. the XZ/YZ pillar-axis-ablation experiment. Anchors are 4-dim
    (u, v, proj_w, h) -- see `box_np_ops.create_plane_anchors` -- not the
    usual 7-dim (x,y,z,w,l,h,r): there's no third axis to fix and no
    rotation to enumerate (see `box_np_ops.project_box3d_to_xz`), so unlike
    `AnchorGeneratorRange` there's only ever one implicit "rotation".
    """
    def __init__(self,
                 u_range,
                 v_range,
                 sizes=[[3.9, 1.56]],
                 class_id=None,
                 match_threshold=-1,
                 unmatch_threshold=-1,
                 dtype=np.float32):
        self._u_range = u_range
        self._v_range = v_range
        self._sizes = sizes
        self._dtype = dtype
        self._class_id = class_id
        self._match_threshold = match_threshold
        self._unmatch_threshold = unmatch_threshold

    @property
    def class_id(self):
        return self._class_id

    @property
    def match_threshold(self):
        return self._match_threshold

    @property
    def unmatch_threshold(self):
        return self._unmatch_threshold

    @property
    def num_anchors_per_localization(self):
        return np.array(self._sizes).reshape([-1, 2]).shape[0]

    def generate(self, feature_map_size):
        # feature_map_size here is [v_size, u_size] (native plane grid,
        # NOT the canonical XY grid AnchorGeneratorRange assumes).
        v_size, u_size = feature_map_size[0], feature_map_size[1]
        return box_np_ops.create_plane_anchors(
            self._u_range, self._v_range, u_size, v_size, self._sizes,
            self._dtype)
