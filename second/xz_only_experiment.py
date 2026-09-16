"""XZ/YZ-only pillar-axis-ablation: minimal validation experiment.

Standard PointPillars pillarizes on the XY (ground) plane, collapsing Z.
The existing PP-XZ/PP-YZ ablation (second/core/view_transform.py) instead
forms pillars on XZ/YZ, but then bilinear-interpolates that native pseudo
image back into the canonical XY grid so it can reuse XY's anchors/RPN --
without reprojecting to true XY spatial positions (see
second/pytorch/models/voxelnet.py:688-699). That's the leading suspected
cause of PP-XZ/PP-YZ's near-zero KITTI results.

This script tests that directly: it builds an XZ (or YZ) branch that keeps
pillars at native resolution and uses dedicated (no-rotation, axis-aligned)
anchors/target-assignment/box-coding for that plane, instead of resizing
into the XY grid. It deliberately bypasses the .proto config system (per
plan) and constructs the pieces that system would normally build directly
in Python -- everything else (PillarFeatureNet, PointPillarsScatter, RPN,
VoxelNet, the KITTI data pipeline) is reused unmodified. Nothing about the
existing XY path is touched by running this.

Scope (see /home/bohyeon/.claude/plans/tidy-purring-sutton.md): a full
296,960-step XZ run (with ground-truth database sampling, matching
xyres_16.proto's database_sampler) confirmed the fix works dramatically --
at score_threshold~0.2-0.3, Cyclist/Pedestrian recall reached ~50-56%/~42-52%,
vs. the old XZ baseline's ~1.4%/~9.1% KITTI 3D AP. This run applies the
same (now-generalized) pipeline to YZ.

Usage:
    python -m second.xz_only_experiment train --model_dir=<dir> --view=xz [--max_steps=296960]
    python -m second.xz_only_experiment evaluate --model_dir=<dir> --view=xz [--score_threshold=0.3]
"""
import pathlib
import pickle
import time
from functools import partial

import fire
import numpy as np
import torch

from second.core import box_np_ops
from second.core import preprocess as prep
from second.core.sample_ops import DataBaseSamplerV2
from second.pytorch.core import box_torch_ops
from second.core.anchor_generator import AnchorGeneratorPlaneRange
from second.core.region_similarity import PlaneIouSimilarity
from second.core.target_assigner import TargetAssigner
from second.core.voxel_generator import VoxelGenerator
from second.data.dataset import KittiDataset
from second.data.preprocess import prep_pointcloud, merge_second_batch
from second.pytorch.core.box_coders import XZBoxCoderTorch
from second.pytorch.core.losses import (SigmoidFocalClassificationLoss,
                                         WeightedSmoothL1LocalizationLoss)
from second.pytorch.models.voxelnet import VoxelNet, LossNormType
from second.pytorch.train import example_convert_to_torch
import torchplus

# ---------------------------------------------------------------------------
# Fixed experiment config (mirrors second/configs/pointpillars/ped_cycle/
# xyres_16.proto's values wherever the same concept applies -- see that file
# for the source of each number below).
# ---------------------------------------------------------------------------
KITTI_ROOT = "/home/bohyeon/projects/DILAB-pointpillars/second/data/kitti"
TRAIN_INFO_PATH = f"{KITTI_ROOT}/kitti_infos_train.pkl"
VAL_INFO_PATH = f"{KITTI_ROOT}/kitti_infos_val.pkl"
CLASS_NAMES = ["Cyclist", "Pedestrian"]

# Canonical (XY) point cloud range/voxel size -- unchanged from the shared
# config. Only used here for GT BEV-range filtering (x,y bounds); z plays
# no role in that filter, so it doesn't need to match the padded z range
# used below for both XZ and YZ.
CANONICAL_PC_RANGE = [0, -19.84, -2.5, 47.36, 19.84, 0.5]
CANONICAL_VOXEL_SIZE = [0.16, 0.16, 3]

# Z is widened from the canonical [-2.5, 0.5] (span 3.0 -> 19 cells at
# 0.16m) to [-2.6, 0.6] (span 3.2 -> 20 cells): the RPN's conv/deconv stride
# pattern here (layer_strides=[1,2,2], upsample_strides=[1,2,4]) only
# round-trips to a matching shape when the input height is a multiple of 4
# (verified by hand: 19 -> mismatched 19 vs 20 vs 20 after the three deconv
# branches; 20 -> 20/20/20, consistent). The extra 0.1m margin on each side
# is empty space near the sensor's own height range, not a meaningful range
# change. X (296 cells) and Y (248 cells) are already multiples of 4 at
# 0.16m, so they need no such padding.
Z_MIN, Z_MAX = -2.6, 0.6
X_MIN, X_MAX = CANONICAL_PC_RANGE[0], CANONICAL_PC_RANGE[3]
Y_MIN, Y_MAX = CANONICAL_PC_RANGE[1], CANONICAL_PC_RANGE[4]

# (proj_w, h) anchor sizes, taken directly from xyres_16.proto's per-class
# anchor sizes ([w, l, h] there) -- at the anchor's own (implicit) r=0,
# projected width along X (for xz) or Y (for yz) equals l.
ANCHOR_SIZES = [
    [1.76, 1.73],  # Cyclist: l=1.76, h=1.73
    [0.8, 1.73],   # Pedestrian: l=0.8, h=1.73
]
MATCHED_THRESHOLD = 0.5
UNMATCHED_THRESHOLD = 0.35

BATCH_SIZE = 2
NUM_WORKERS = 2
MAX_VOXELS = 12000
BASE_LR = 2e-4
WEIGHT_DECAY = 1e-4
DECAY_STEPS = 27840
DECAY_FACTOR = 0.8


def _view_geometry(view):
    """Everything that differs between the XZ and YZ branches.

    xz: pillars on (x, z), Y collapsed -- native canvas 20 (z) x 296 (x).
    yz: pillars on (y, z), X collapsed -- native canvas 20 (z) x 248 (y).
    Both dimensions in both cases are already multiples of 4 (see Z_MIN/
    Z_MAX comment above), so canonical_hw=None (no resize) works for either
    without further padding.
    """
    assert view in ('xz', 'yz'), f"unknown view {view!r}"
    if view == 'xz':
        u_range = (X_MIN, X_MAX)
        u_size = 296
        spatial_axes = (0, 2)  # x, z
        pillar_voxel_size = [0.16, (Y_MAX - Y_MIN) + 1.0, 0.16]  # y collapsed
        projection_fn = box_np_ops.project_box3d_to_xz
    else:
        u_range = (Y_MIN, Y_MAX)
        u_size = 248
        spatial_axes = (1, 2)  # y, z
        pillar_voxel_size = [(X_MAX - X_MIN) + 1.0, 0.16, 0.16]  # x collapsed
        projection_fn = box_np_ops.project_box3d_to_yz
    v_range = (Z_MIN, Z_MAX)
    v_size = 20
    pillar_pc_range = [X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX]
    return dict(
        spatial_axes=spatial_axes,
        u_range=u_range, v_range=v_range,
        u_size=u_size, v_size=v_size,
        feature_map_size=[v_size, u_size],
        pillar_voxel_size=pillar_voxel_size,
        pillar_pc_range=pillar_pc_range,
        projection_fn=projection_fn,
    )


def _build_target_assigner(view):
    geo = _view_geometry(view)
    anchor_gen = AnchorGeneratorPlaneRange(
        u_range=geo["u_range"],
        v_range=geo["v_range"],
        sizes=ANCHOR_SIZES,
        class_id=None,
        match_threshold=MATCHED_THRESHOLD,
        unmatch_threshold=UNMATCHED_THRESHOLD,
    )
    box_coder = XZBoxCoderTorch()
    return TargetAssigner(
        box_coder=box_coder,
        anchor_generators=[anchor_gen],
        region_similarity_calculator=PlaneIouSimilarity(),
        positive_fraction=None,
        sample_size=512,
        similarity_columns=[0, 1, 2, 3],
        box_ndim=4,
    )


def _build_net(view, target_assigner):
    geo = _view_geometry(view)
    # dense_shape = [batch, z_dim, y_dim, x_dim, C]; whichever of y/x is the
    # collapsed axis gets dim 1, matching second_builder.py's convention.
    if view == 'xz':
        dense_shape = [1, geo["v_size"], 1, geo["u_size"], 64]
    else:
        dense_shape = [1, geo["v_size"], geo["u_size"], 1, 64]
    net = VoxelNet(
        output_shape=dense_shape,
        num_class=len(CLASS_NAMES),
        num_input_features=4,
        vfe_class_name="PillarFeatureNet",
        vfe_num_filters=[64],
        middle_class_name="PointPillarsScatter",
        rpn_layer_nums=[3, 5, 5],
        rpn_layer_strides=[1, 2, 2],
        rpn_num_filters=[64, 128, 256],
        rpn_upsample_strides=[1, 2, 4],
        rpn_num_upsample_filters=[128, 128, 128],
        use_direction_classifier=False,  # no rotation in this plane's boxes
        use_sigmoid_score=True,
        encode_background_as_zeros=True,
        use_rotate_nms=False,  # axis-aligned boxes -> axis-aligned NMS
        multiclass_nms=False,
        nms_score_threshold=0.05,
        nms_pre_max_size=1000,
        nms_post_max_size=300,
        nms_iou_threshold=0.5,
        target_assigner=target_assigner,
        cls_loss_weight=1.0,
        loc_loss_weight=2.0,
        pos_cls_weight=1.0,
        neg_cls_weight=1.0,
        loss_norm_type=LossNormType.NormByNumPositives,
        encode_rad_error_by_sin=False,  # no rotation to sin-encode
        loc_loss_ftor=WeightedSmoothL1LocalizationLoss(sigma=3.0),
        cls_loss_ftor=SigmoidFocalClassificationLoss(gamma=2.0, alpha=0.25),
        voxel_size=geo["pillar_voxel_size"],
        pc_range=geo["pillar_pc_range"],
        pillar_spatial_axes=geo["spatial_axes"],
        canonical_hw=None,  # <-- the actual fix: skip the resize entirely
    )
    return net


def _build_db_sampler():
    """Ground-truth database sampling augmentation (oversamples Cyclist),
    matching second/configs/pointpillars/ped_cycle/xyres_16.proto's
    `database_sampler` block exactly. Same for XZ and YZ.
    """
    db_prepor = prep.DataBasePreprocessor([
        prep.DBFilterByMinNumPoint({"Cyclist": 5}),
        prep.DBFilterByDifficulty([-1]),
    ])
    with open(f"{KITTI_ROOT}/kitti_dbinfos_train.pkl", 'rb') as f:
        db_infos = pickle.load(f)
    return DataBaseSamplerV2(
        db_infos, groups=[{"Cyclist": 8}], db_prepor=db_prepor,
        rate=1.0, global_rot_range=[0, 0])


def _build_dataset(view, target_assigner, training=True):
    geo = _view_geometry(view)
    info_path = TRAIN_INFO_PATH if training else VAL_INFO_PATH
    canonical_voxel_generator = VoxelGenerator(
        voxel_size=CANONICAL_VOXEL_SIZE,
        point_cloud_range=CANONICAL_PC_RANGE,
        max_num_points=100,
    )
    pillar_voxel_generator = VoxelGenerator(
        voxel_size=geo["pillar_voxel_size"],
        point_cloud_range=geo["pillar_pc_range"],
        max_num_points=100,
        max_voxels=20000,
    )
    prep_func = partial(
        prep_pointcloud,
        root_path=KITTI_ROOT,
        class_names=CLASS_NAMES,
        voxel_generator=canonical_voxel_generator,
        pillar_voxel_generator=pillar_voxel_generator,
        target_assigner=target_assigner,
        training=training,
        max_voxels=MAX_VOXELS,
        remove_outside_points=False,
        remove_unknown=False,
        create_targets=training,
        shuffle_points=training,
        gt_rotation_noise=[-0.15707963267, 0.15707963267],
        gt_loc_noise_std=[0.25, 0.25, 0.25],
        global_rotation_noise=[-0.78539816, 0.78539816],
        global_scaling_noise=[0.95, 1.05],
        global_loc_noise_std=(0.2, 0.2, 0.2),
        global_random_rot_range=[0, 0],
        db_sampler=_build_db_sampler() if training else None,
        generate_bev=False,
        without_reflectivity=False,
        num_point_features=4,
        anchor_area_threshold=-1,  # anchors_mask pruning stays disabled (see plan, stage 3)
        gt_points_drop=0.0,
        gt_drop_max_keep=15,
        remove_points_after_sample=False,
        remove_environment=False,
        use_group_id=False,
        out_size_factor=1,  # matches this config's layer_strides[0]//upsample_strides[0]
        anchor_feature_map_size=geo["feature_map_size"],
        anchor_box_ndim=4,
        gt_box_projection_fn=geo["projection_fn"],
    )
    return KittiDataset(
        info_path=info_path,
        root_path=KITTI_ROOT,
        num_point_features=4,
        target_assigner=target_assigner,
        feature_map_size=geo["feature_map_size"],
        prep_func=prep_func,
        anchor_box_ndim=4,
    )


def smoke_test(view='xz'):
    """~20-step sanity check: shapes line up, loss is finite, no crash."""
    geo = _view_geometry(view)
    target_assigner = _build_target_assigner(view)
    print(f"{view} anchors per location: {target_assigner.num_anchors_per_location} "
          f"(expected {len(ANCHOR_SIZES)}, no rotation)")
    total_anchors = geo["u_size"] * geo["v_size"] * target_assigner.num_anchors_per_location
    print(f"{view} total anchors: {geo['u_size']}x{geo['v_size']}x"
          f"{target_assigner.num_anchors_per_location} = {total_anchors}")

    net = _build_net(view, target_assigner).cuda()
    dataset = _build_dataset(view, target_assigner, training=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=merge_second_batch)
    optimizer = torch.optim.Adam(net.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    net.train()
    it = iter(loader)
    for step in range(20):
        example = next(it)
        example_torch = example_convert_to_torch(example)
        ret = net(example_torch)
        loss = ret["loss"].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        net.update_global_step()
        print(f"step={step} loss={float(loss):.4f} "
              f"cls_loss={float(ret['cls_loss_reduced']):.4f} "
              f"loc_loss={float(ret['loc_loss_reduced']):.4f} "
              f"num_pos={int((example['labels'] > 0).sum())}")
    print("smoke test OK")


def train(model_dir, view='xz', max_steps=296960, display_step=50, save_every=5000):
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    target_assigner = _build_target_assigner(view)
    net = _build_net(view, target_assigner).cuda()
    torchplus.train.try_restore_latest_checkpoints(str(model_dir), [net])

    dataset = _build_dataset(view, target_assigner, training=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=merge_second_batch, drop_last=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    optimizer.name = "adam_optimizer"
    torchplus.train.try_restore_latest_checkpoints(str(model_dir), [optimizer])
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: DECAY_FACTOR ** (step // DECAY_STEPS))

    net.train()
    step = net.get_global_step()
    t0 = time.time()
    data_iter = iter(loader)
    while step < max_steps:
        try:
            example = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            example = next(data_iter)
        example_torch = example_convert_to_torch(example)
        ret = net(example_torch)
        loss = ret["loss"].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        net.update_global_step()
        step = net.get_global_step()

        if step % display_step == 0:
            dt = time.time() - t0
            t0 = time.time()
            num_pos = int((example["labels"] > 0).sum())
            num_anchors = example["labels"].size
            print(f"step={step} steptime={dt / display_step:.3f} "
                  f"loss={float(loss):.4f} cls_loss={float(ret['cls_loss_reduced']):.4f} "
                  f"loc_loss={float(ret['loc_loss_reduced']):.4f} "
                  f"num_pos={num_pos} num_anchors={num_anchors} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")
        if step % save_every == 0:
            torchplus.train.save_models(str(model_dir), [net, optimizer], step)

    torchplus.train.save_models(str(model_dir), [net, optimizer], step)
    print(f"done at step {step}")


def _plane_to_minmax(boxes):
    u, v, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([u - w / 2, v - h / 2, u + w / 2, v + h / 2], axis=-1)


def _all_point_ap(scores, tp_flags, num_gt):
    """Standard all-point-interpolated AP (area under the precision-recall
    curve, PASCAL VOC 2010+ / COCO style) -- NOT KITTI's official 11-/40-
    point recall interpolation, since this isn't KITTI's official metric to
    begin with (see `evaluate`'s docstring). `scores`/`tp_flags` are the
    concatenation, across the whole val set, of every post-NMS detection
    for one class; `num_gt` is that class's total GT count across the set.
    """
    if num_gt == 0 or len(scores) == 0:
        return 0.0, np.array([0.0]), np.array([0.0])
    order = np.argsort(-np.asarray(scores))
    tp_flags = np.asarray(tp_flags, dtype=np.float64)[order]
    fp_flags = 1.0 - tp_flags
    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(fp_flags)
    recall = tp_cum / num_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # Make precision non-increasing from the right (standard envelope), then
    # integrate: AP = sum_i (recall_i - recall_{i-1}) * precision_envelope_i.
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_padded = np.concatenate([[0.0], recall])
    ap = float(np.sum((recall_padded[1:] - recall_padded[:-1]) * precision_envelope))
    return ap, recall, precision


def evaluate(model_dir, view='xz', score_threshold=0.3, iou_threshold=0.5, max_samples=None):
    """Plane-only 2D detection AP + recall/precision at one threshold.

    NOT full KITTI 3D AP -- this branch alone only predicts (u, z,
    projected_width, h) (u = x for xz, y for yz), not a full 3D box (see
    module docstring and second/core/box_coders.py's `XZBoxCoder`), and the
    AP computed here uses all-point PR-curve interpolation (see
    `_all_point_ap`), not KITTI's official 11-/40-point recall
    interpolation. Matches detections against GT, both projected into the
    same 4-dim representation, via plain axis-aligned IoU. This is the
    metric that answers this experiment's actual question: does removing
    the interpolation/anchor-mismatch bug make this plane detect real
    objects at all, vs. the old near-0% KITTI 3D AP baseline.
    """
    geo = _view_geometry(view)
    target_assigner = _build_target_assigner(view)
    net = _build_net(view, target_assigner).cuda()
    torchplus.train.try_restore_latest_checkpoints(str(model_dir), [net])
    net.eval()
    print(f"restored checkpoint at global_step={net.get_global_step()}")

    dataset = _build_dataset(view, target_assigner, training=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=merge_second_batch)

    num_class = len(CLASS_NAMES)
    # Fixed-threshold table (still useful for a concrete recall/precision
    # reading), plus per-class (score, is_tp) lists for AP.
    tp = np.zeros(num_class, dtype=np.int64)
    num_gt = np.zeros(num_class, dtype=np.int64)
    num_det = np.zeros(num_class, dtype=np.int64)
    ap_scores = [[] for _ in range(num_class)]
    ap_tp_flags = [[] for _ in range(num_class)]

    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    t0 = time.time()
    with torch.no_grad():
        for i, example in enumerate(loader):
            if i >= n_samples:
                break
            info = dataset.kitti_infos[i]
            example_torch = example_convert_to_torch(example)

            voxel_features = net.voxel_feature_extractor(
                example_torch["voxels"], example_torch["num_points"],
                example_torch["coordinates"])
            spatial_features = net.middle_feature_extractor(
                voxel_features, example_torch["coordinates"], 1)
            preds_dict = net.rpn(spatial_features)
            box_preds = preds_dict["box_preds"].view(1, -1, 4)
            cls_preds = preds_dict["cls_preds"].view(1, -1, num_class)
            anchors = example_torch["anchors"].view(1, -1, 4)
            box_preds = net._box_coder.decode_torch(box_preds, anchors)[0]
            scores = torch.sigmoid(cls_preds)[0]

            box_preds_np = box_preds.cpu().numpy()
            scores_np = scores.cpu().numpy()

            annos = info["annos"]
            keep = annos["name"] != "DontCare"
            names = annos["name"][keep]
            gt_boxes_cam = np.concatenate(
                [annos["location"][keep], annos["dimensions"][keep],
                 annos["rotation_y"][keep][..., np.newaxis]],
                axis=1).astype(np.float32)
            rect = info["calib/R0_rect"].astype(np.float32)
            Trv2c = info["calib/Tr_velo_to_cam"].astype(np.float32)
            gt_boxes_lidar = box_np_ops.box_camera_to_lidar(gt_boxes_cam, rect, Trv2c)
            gt_plane = geo["projection_fn"](gt_boxes_lidar)

            for cls_idx, cls_name in enumerate(CLASS_NAMES):
                cls_gt = gt_plane[names == cls_name]
                num_gt[cls_idx] += len(cls_gt)

                cls_scores = scores_np[:, cls_idx]
                # Low, fixed cutoff (not `score_threshold`) so the AP curve
                # covers the full recall range; NMS still runs at 0.5 IoU.
                sel = cls_scores > 0.01
                cls_dets = box_preds_np[sel]
                cls_det_scores = cls_scores[sel]
                if len(cls_dets) > 0:
                    dets_t = torch.as_tensor(cls_dets, device="cuda")
                    scores_t = torch.as_tensor(cls_det_scores, device="cuda")
                    minmax_t = torch.stack([
                        dets_t[:, 0] - dets_t[:, 2] / 2,
                        dets_t[:, 1] - dets_t[:, 3] / 2,
                        dets_t[:, 0] + dets_t[:, 2] / 2,
                        dets_t[:, 1] + dets_t[:, 3] / 2], dim=-1)
                    keep_idx = box_torch_ops.nms(
                        minmax_t, scores_t, pre_max_size=1000,
                        post_max_size=300, iou_threshold=0.5)
                    if keep_idx is not None:
                        keep_idx_np = keep_idx.cpu().numpy()
                        cls_dets = cls_dets[keep_idx_np]
                        cls_det_scores = cls_det_scores[keep_idx_np]
                    else:
                        cls_dets = cls_dets[:0]
                        cls_det_scores = cls_det_scores[:0]
                num_det[cls_idx] += int((cls_det_scores > score_threshold).sum())

                if len(cls_dets) == 0:
                    continue
                det_tp_flags = np.zeros(len(cls_dets), dtype=np.bool_)
                if len(cls_gt) > 0:
                    gt_minmax = _plane_to_minmax(cls_gt)
                    det_minmax = _plane_to_minmax(cls_dets)
                    ious = box_np_ops.iou_jit(det_minmax, gt_minmax, eps=0.0)
                    # Greedy match in score order (highest-scoring detection
                    # claims its best-IoU GT first) -- standard AP matching.
                    score_order = np.argsort(-cls_det_scores)
                    matched_gt = set()
                    for d_idx in score_order:
                        g_idx = int(np.argmax(ious[d_idx]))
                        if ious[d_idx, g_idx] >= iou_threshold and g_idx not in matched_gt:
                            matched_gt.add(g_idx)
                            det_tp_flags[d_idx] = True
                    tp[cls_idx] += int((det_tp_flags & (cls_det_scores > score_threshold)).sum())
                ap_scores[cls_idx].extend(cls_det_scores.tolist())
                ap_tp_flags[cls_idx].extend(det_tp_flags.tolist())

    print(f"evaluated {n_samples} frames in {time.time() - t0:.1f}s "
          f"(view={view}, score_threshold={score_threshold}, iou_threshold={iou_threshold})")
    print(f"{'class':<12}{'GT':>8}{'Det':>8}{'TP':>8}{'Recall':>10}{'Precision':>10}{'AP':>10}")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        recall = tp[cls_idx] / max(1, num_gt[cls_idx])
        precision = tp[cls_idx] / max(1, num_det[cls_idx])
        ap, _, _ = _all_point_ap(ap_scores[cls_idx], ap_tp_flags[cls_idx], num_gt[cls_idx])
        print(f"{cls_name:<12}{num_gt[cls_idx]:>8}{num_det[cls_idx]:>8}{tp[cls_idx]:>8}"
              f"{recall:>10.3f}{precision:>10.3f}{ap:>10.3f}")


if __name__ == '__main__':
    fire.Fire()
