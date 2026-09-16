"""XY+XZ(+YZ) pillar fusion: minimal validation experiment.

Builds a `MultiViewVoxelNet` (second/pytorch/models/voxelnet.py) with an XY
branch (standard PointPillars) plus one or two auxiliary branches (XZ/YZ),
each kept at native 2D resolution through its own backbone, fused via
channel-fold (see plan at /home/bohyeon/.claude/plans/tidy-purring-sutton.md)
right before the shared detection head. Detection stays fully standard XY:
same GroundBox3dCoder/anchors/target_assigner as plain PointPillars -- XZ/YZ
only add feature channels, no box predictions of their own.

Like second/xz_only_experiment.py, this bypasses the .proto config system
and constructs pieces directly in Python. Nothing here touches the existing
single-branch VoxelNet/RPN classes or the standard xy-only training path
(second/pytorch/train.py) -- verified by `git diff --stat` showing 0
deletions in voxelnet.py.

Usage:
    python -m second.multiview_experiment smoke_test --branches=xz
    python -m second.multiview_experiment train --model_dir=<dir> --branches=xz
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
from second.core import view_transform
from second.core.anchor_generator import AnchorGeneratorStride
from second.core.box_coders import GroundBox3dCoder
from second.core.region_similarity import NearestIouSimilarity
from second.core.sample_ops import DataBaseSamplerV2
from second.core.target_assigner import TargetAssigner
from second.core.voxel_generator import VoxelGenerator
from second.data.dataset import KittiDataset
from second.data.preprocess import prep_pointcloud, merge_second_batch
from second.pytorch.core.box_coders import GroundBox3dCoderTorch
from second.pytorch.core.losses import (SigmoidFocalClassificationLoss,
                                         WeightedSmoothL1LocalizationLoss)
from second.pytorch.models.voxelnet import MultiViewVoxelNet, LossNormType
from second.pytorch.train import example_convert_to_torch, predict_kitti_to_anno
from second.utils.eval import get_coco_eval_result, get_official_eval_result
import torchplus

# ---------------------------------------------------------------------------
# Per-task config. Each entry mirrors its second/configs/pointpillars/*/
# xyres_16.proto exactly (same classes/anchors/loss/optimizer/rpn strides) --
# see those files. Z_MIN/Z_MAX for the xz/yz auxiliary branches are padded
# outward (symmetrically) from the task's own point_cloud_range Z extent so
# that the resulting pillar count along Z is an exact multiple of the task's
# RPN downsample factor (product of rpn_layer_strides): each branch's own
# RPNBackbone needs its input H (Z, for xz/yz) to divide evenly through the
# block/deconv stride pattern for the multi-stage upsample-and-concat to
# round-trip to matching shapes (see second/xz_only_experiment.py). ped_cycle
# uses layer_strides=[1,2,2] (downsample factor 4); car uses [2,2,2]
# (downsample factor 8) -- so car's Z padding is computed to a multiple of 8,
# not 4.
# ---------------------------------------------------------------------------
KITTI_ROOT = "/home/bohyeon/projects/DILAB-pointpillars/second/data/kitti"
TRAIN_INFO_PATH = f"{KITTI_ROOT}/kitti_infos_train.pkl"
VAL_INFO_PATH = f"{KITTI_ROOT}/kitti_infos_val.pkl"

RPN_LAYER_NUMS = [3, 5, 5]
RPN_NUM_FILTERS = [64, 128, 256]
RPN_UPSAMPLE_STRIDES = [1, 2, 4]
RPN_NUM_UPSAMPLE_FILTERS = [128, 128, 128]

BATCH_SIZE = 2
NUM_WORKERS = 2
MAX_VOXELS = 12000
BASE_LR = 2e-4
WEIGHT_DECAY = 1e-4
DECAY_STEPS = 27840
DECAY_FACTOR = 0.8

TASKS = {
    'ped_cycle': dict(
        class_names=["Cyclist", "Pedestrian"],
        pc_range=[0, -19.84, -2.5, 47.36, 19.84, 0.5],
        voxel_size=[0.16, 0.16, 3],
        rpn_layer_strides=[1, 2, 2],  # xyres_16.proto layer_strides[0]=1 -> out_size_factor=1
        z_min=-2.6, z_max=0.6,  # padded to 20 cells (mult. of 4) @ 0.16 res
        anchor_gens=[
            dict(sizes=[0.6, 1.76, 1.73], anchor_strides=[0.16, 0.16, 0.0],
                 anchor_offsets=[0.08, -19.76, -1.465], rotations=[0, 1.57],
                 match_threshold=0.5, unmatch_threshold=0.35),
            dict(sizes=[0.6, 0.8, 1.73], anchor_strides=[0.16, 0.16, 0.0],
                 anchor_offsets=[0.08, -19.76, -1.465], rotations=[0, 1.57],
                 match_threshold=0.5, unmatch_threshold=0.35),
        ],
        db_min_points={"Cyclist": 5},
        db_sample_groups=[{"Cyclist": 8}],
    ),
    'car': dict(
        class_names=["Car"],
        pc_range=[0, -39.68, -3, 69.12, 39.68, 1],
        voxel_size=[0.16, 0.16, 4],
        rpn_layer_strides=[2, 2, 2],  # xyres_16.proto layer_strides[0]=2 -> out_size_factor=2
        z_min=-3.56, z_max=1.56,  # padded to 32 cells (mult. of 8) @ 0.16 res
        anchor_gens=[
            dict(sizes=[1.6, 3.9, 1.56], anchor_strides=[0.32, 0.32, 0.0],
                 anchor_offsets=[0.16, -39.52, -1.78], rotations=[0, 1.57],
                 match_threshold=0.6, unmatch_threshold=0.45),
        ],
        db_min_points={"Car": 5},
        db_sample_groups=[{"Car": 15}],
    ),
}


def _out_size_factor(cfg):
    return cfg['rpn_layer_strides'][0] // RPN_UPSAMPLE_STRIDES[0]


def _xy_branch_config(cfg):
    canonical = VoxelGenerator(cfg['voxel_size'], cfg['pc_range'], max_num_points=100)
    grid = canonical.grid_size  # [nx, ny, nz]
    dense_shape = [1] + grid[::-1].tolist() + [64]  # [1, z=1, y, x, C]
    return {
        'voxel_size': cfg['voxel_size'],
        'pc_range': cfg['pc_range'],
        'spatial_axes': (0, 1),
        'output_shape': dense_shape,
        'rpn_layer_nums': RPN_LAYER_NUMS,
        'rpn_layer_strides': cfg['rpn_layer_strides'],
        'rpn_num_filters': RPN_NUM_FILTERS,
        'rpn_upsample_strides': RPN_UPSAMPLE_STRIDES,
        'rpn_num_upsample_filters': RPN_NUM_UPSAMPLE_FILTERS,
    }, canonical


def _aux_branch_config(cfg, view):
    assert view in ('xz', 'yz')
    pc_range = cfg['pc_range']
    z_min, z_max = cfg['z_min'], cfg['z_max']
    full_pc_range = [pc_range[0], pc_range[1], z_min, pc_range[3], pc_range[4], z_max]
    if view == 'xz':
        voxel_size = [0.16, (pc_range[4] - pc_range[1]) + 1.0, 0.16]
        spatial_axes = (0, 2)
    else:
        voxel_size = [(pc_range[3] - pc_range[0]) + 1.0, 0.16, 0.16]
        spatial_axes = (1, 2)
    gen = VoxelGenerator(voxel_size, full_pc_range, max_num_points=100, max_voxels=20000)
    grid = gen.grid_size  # [nx, ny, nz]
    dense_shape = [1] + grid[::-1].tolist() + [64]
    branch_cfg = {
        'voxel_size': voxel_size,
        'pc_range': full_pc_range,
        'spatial_axes': spatial_axes,
        'output_shape': dense_shape,
        # Same stride config as XY -- the shared axis (X for xz, Y for yz)
        # then round-trips through its own backbone to the SAME length as
        # XY's, with no interpolation, simply because it's the same
        # deterministic stride arithmetic on the same input length.
        'rpn_layer_nums': RPN_LAYER_NUMS,
        'rpn_layer_strides': cfg['rpn_layer_strides'],
        'rpn_num_filters': RPN_NUM_FILTERS,
        'rpn_upsample_strides': RPN_UPSAMPLE_STRIDES,
        'rpn_num_upsample_filters': RPN_NUM_UPSAMPLE_FILTERS,
    }
    return branch_cfg, gen


def _build_target_assigner(cfg):
    """Exactly the task's xyres_16.proto target_assigner: its
    AnchorGeneratorStride(s), NearestIouSimilarity, GroundBox3dCoder.
    """
    box_coder = GroundBox3dCoderTorch(linear_dim=False, vec_encode=False)
    anchor_gens = [AnchorGeneratorStride(**a) for a in cfg['anchor_gens']]
    return TargetAssigner(
        box_coder=box_coder,
        anchor_generators=anchor_gens,
        region_similarity_calculator=NearestIouSimilarity(),
        positive_fraction=None,
        sample_size=512,
    )


def _build_net(cfg, branches, target_assigner, fusion_debug=False):
    branch_configs = {}
    xy_cfg, _ = _xy_branch_config(cfg)
    branch_configs['xy'] = xy_cfg
    for view in branches:
        branch_cfg, _ = _aux_branch_config(cfg, view)
        branch_configs[view] = branch_cfg
    return MultiViewVoxelNet(
        branch_configs=branch_configs,
        fusion_debug=fusion_debug,
        num_class=len(cfg['class_names']),
        vfe_num_filters=[64],
        use_direction_classifier=True,
        use_sigmoid_score=True,
        encode_background_as_zeros=True,
        use_rotate_nms=False,
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
        direction_loss_weight=0.2,
        loss_norm_type=LossNormType.NormByNumPositives,
        encode_rad_error_by_sin=True,
        loc_loss_ftor=WeightedSmoothL1LocalizationLoss(sigma=3.0),
        cls_loss_ftor=SigmoidFocalClassificationLoss(gamma=2.0, alpha=0.25),
        fused_channels=128,
    )


def _build_db_sampler(cfg):
    db_prepor = prep.DataBasePreprocessor([
        prep.DBFilterByMinNumPoint(cfg['db_min_points']),
        prep.DBFilterByDifficulty([-1]),
    ])
    with open(f"{KITTI_ROOT}/kitti_dbinfos_train.pkl", 'rb') as f:
        db_infos = pickle.load(f)
    return DataBaseSamplerV2(
        db_infos, groups=cfg['db_sample_groups'], db_prepor=db_prepor,
        rate=1.0, global_rot_range=[0, 0])


def _build_dataset(cfg, branches, target_assigner, training=True):
    info_path = TRAIN_INFO_PATH if training else VAL_INFO_PATH
    _, canonical_gen = _xy_branch_config(cfg)
    aux_gens = {}
    for view in branches:
        _, gen = _aux_branch_config(cfg, view)
        aux_gens[view] = gen

    out_size_factor = _out_size_factor(cfg)
    grid_size = canonical_gen.grid_size
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]

    prep_func = partial(
        prep_pointcloud,
        root_path=KITTI_ROOT,
        class_names=cfg['class_names'],
        voxel_generator=canonical_gen,
        pillar_voxel_generator=None,  # None -> use canonical (xy) directly
        aux_pillar_voxel_generators=aux_gens,
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
        db_sampler=_build_db_sampler(cfg) if training else None,
        generate_bev=False,
        without_reflectivity=False,
        num_point_features=4,
        anchor_area_threshold=1,  # standard xy anchors_mask pruning, unaffected by aux branches
        gt_points_drop=0.0,
        gt_drop_max_keep=15,
        remove_points_after_sample=False,
        remove_environment=False,
        use_group_id=False,
        out_size_factor=out_size_factor,
    )
    return KittiDataset(
        info_path=info_path,
        root_path=KITTI_ROOT,
        num_point_features=4,
        target_assigner=target_assigner,
        feature_map_size=feature_map_size,
        prep_func=prep_func,
    )


def smoke_test(branches='xz', steps=20, task='ped_cycle'):
    """Step 6 of the plan: run a few steps and log every shape/count the
    plan asks for (F_xy/F_xz/F_yz shape, fused shape, anchor counts,
    pos/neg counts, cls/loc/dir loss, GPU memory).
    """
    cfg = TASKS[task]
    branch_list = list(branches) if isinstance(branches, (tuple, list)) else [
        b for b in branches.split(',') if b]
    target_assigner = _build_target_assigner(cfg)
    net = _build_net(cfg, branch_list, target_assigner, fusion_debug=True).cuda()
    print(f"task={task} branches={['xy'] + branch_list}")
    print(f"num_anchors_per_location={target_assigner.num_anchors_per_location}")

    dataset = _build_dataset(cfg, branch_list, target_assigner, training=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=merge_second_batch)
    optimizer = torch.optim.Adam(net.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    net.train()
    it = iter(loader)
    for step in range(steps):
        example = next(it)
        example_torch = example_convert_to_torch(example)
        ret = net(example_torch)
        loss = ret["loss"].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        net.update_global_step()
        num_pos = int((example["labels"] > 0).sum())
        num_neg = int((example["labels"] == 0).sum())
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"step={step} loss={float(loss):.4f} "
              f"cls_loss={float(ret['cls_loss_reduced']):.4f} "
              f"loc_loss={float(ret['loc_loss_reduced']):.4f} "
              f"dir_loss={float(ret['dir_loss_reduced']):.4f} "
              f"num_pos={num_pos} num_neg={num_neg} "
              f"gpu_mem_gb={mem:.2f}")
    print("smoke test OK")


def train(model_dir, branches='xz', max_steps=296960, display_step=50, save_every=5000,
          task='ped_cycle'):
    cfg = TASKS[task]
    branch_list = list(branches) if isinstance(branches, (tuple, list)) else [
        b for b in branches.split(',') if b]
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    target_assigner = _build_target_assigner(cfg)
    net = _build_net(cfg, branch_list, target_assigner).cuda()
    torchplus.train.try_restore_latest_checkpoints(str(model_dir), [net])

    dataset = _build_dataset(cfg, branch_list, target_assigner, training=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=merge_second_batch, drop_last=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    optimizer.name = "adam_optimizer"
    torchplus.train.try_restore_latest_checkpoints(str(model_dir), [optimizer])

    def _lr_for_step(step):
        # Deliberately NOT a torch LambdaLR/scheduler object: those track
        # their own internal call-count state, which does not get restored
        # across process resume (try_restore_latest_checkpoints only
        # restores net/optimizer). A schedule computed directly from the
        # persisted global `step` is correct on any resume -- this fixed a
        # real bug where a resumed multi-view run trained its last 10k/60k
        # steps at the un-decayed base LR (2e-4 instead of 1.28e-4).
        return BASE_LR * (DECAY_FACTOR ** (step // DECAY_STEPS))

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
        for pg in optimizer.param_groups:
            pg['lr'] = _lr_for_step(step)
        example_torch = example_convert_to_torch(example)
        ret = net(example_torch)
        loss = ret["loss"].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        net.update_global_step()
        step = net.get_global_step()

        if step % display_step == 0:
            dt = time.time() - t0
            t0 = time.time()
            num_pos = int((example["labels"] > 0).sum())
            print(f"step={step} steptime={dt / display_step:.3f} "
                  f"loss={float(loss):.4f} cls_loss={float(ret['cls_loss_reduced']):.4f} "
                  f"loc_loss={float(ret['loc_loss_reduced']):.4f} "
                  f"dir_loss={float(ret['dir_loss_reduced']):.4f} "
                  f"num_pos={num_pos} lr={optimizer.param_groups[0]['lr']:.2e}")
        if step % save_every == 0:
            torchplus.train.save_models(str(model_dir), [net, optimizer], step)

    torchplus.train.save_models(str(model_dir), [net, optimizer], step)
    print(f"done at step {step}")


def evaluate(model_dir, branches='xz', max_samples=None, task='ped_cycle'):
    """Official KITTI 3D/BEV AP (same protocol/metric as every earlier xy
    baseline in this repo -- see second/pytorch/train.py's `evaluate`,
    reused here via `predict_kitti_to_anno` + `get_official_eval_result`),
    directly comparable across XY-only / XY+XZ / XY+YZ / XY+XZ+YZ. Unlike
    second/xz_only_experiment.py's custom axis-aligned-plane AP, this is a
    full 7-dim (x,y,z,w,l,h,r) box evaluation -- MultiViewVoxelNet always
    predicts full 3D boxes, XZ/YZ are feature-only.
    """
    cfg = TASKS[task]
    branch_list = list(branches) if isinstance(branches, (tuple, list)) else [
        b for b in branches.split(',') if b]
    target_assigner = _build_target_assigner(cfg)
    net = _build_net(cfg, branch_list, target_assigner).cuda()
    torchplus.train.try_restore_latest_checkpoints(str(model_dir), [net])
    net.eval()
    print(f"restored checkpoint at global_step={net.get_global_step()}")

    dataset = _build_dataset(cfg, branch_list, target_assigner, training=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=merge_second_batch)

    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    dt_annos = []
    t0 = time.time()
    with torch.no_grad():
        seen = 0
        for example in loader:
            if seen >= n_samples:
                break
            example_torch = example_convert_to_torch(example)
            dt_annos += predict_kitti_to_anno(
                net, example_torch, cfg['class_names'], cfg['pc_range'])
            seen += example['image_idx'].shape[0]
    print(f"evaluated {seen} frames in {time.time() - t0:.1f}s")

    gt_annos = [info["annos"] for info in dataset.kitti_infos[:seen]]
    result = get_official_eval_result(gt_annos, dt_annos, cfg['class_names'])
    print(result)
    result = get_coco_eval_result(gt_annos, dt_annos, cfg['class_names'])
    print(result)


if __name__ == '__main__':
    fire.Fire()
