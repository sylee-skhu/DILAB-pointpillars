import pathlib
import pickle
import time
from functools import partial

import numpy as np

from second.core import box_np_ops
from second.core import preprocess as prep
from second.data import kitti_common as kitti
from second.data.preprocess import _read_and_prep_v9


class Dataset(object):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError



class KittiDataset(Dataset):
    def __init__(self, info_path, root_path, num_point_features,
                 target_assigner, feature_map_size, prep_func,
                 anchor_box_ndim=7):
        """`anchor_box_ndim` exists for branches whose anchors aren't the
        usual 7-dim (x,y,z,w,l,h,r) -- e.g. the XZ/YZ pillar-axis-ablation
        branch's native 4-dim (u,v,proj_w,h) anchors (see
        second/core/anchor_generator.py's `AnchorGeneratorPlaneRange`). At
        its default (7), behavior is unchanged.
        """
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        #self._kitti_infos = kitti.filter_infos_by_used_classes(infos, class_names)
        self._root_path = root_path
        self._kitti_infos = infos
        self._num_point_features = num_point_features
        print("remain number of infos:", len(self._kitti_infos))
        # generate anchors cache
        # [352, 400]
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, anchor_box_ndim])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        if anchor_box_ndim == 7:
            anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
                anchors[:, [0, 1, 3, 4, 6]])
        else:
            # See the matching note in second/data/preprocess.py -- this
            # rotated-BEV-box occupancy optimization doesn't apply to a
            # native-resolution plane-only anchor set.
            anchors_bv = None
        anchor_cache = {
            "anchors": anchors,
            "anchors_bv": anchors_bv,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
        }
        self._prep_func = partial(prep_func, anchor_cache=anchor_cache)

    def __len__(self):
        return len(self._kitti_infos)

    @property
    def kitti_infos(self):
        return self._kitti_infos

    def __getitem__(self, idx):
        return _read_and_prep_v9(
            info=self._kitti_infos[idx],
            root_path=self._root_path,
            num_point_features=self._num_point_features,
            prep_func=self._prep_func)
