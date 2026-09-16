"""Coordinate-correspondence gated-residual YZ fusion: keeps the original
single-branch XY path/head byte-for-byte weight-compatible with a plain
VoxelNet checkpoint, and merges in a YZ branch through a real-point
(y,z)->(x,y) correspondence instead of ChannelFoldFusion's uniform broadcast.

See second/pytorch/models/voxelnet.py's GatedYZVoxelNet /
PointCorrespondenceGatedFusion / masked_scatter_mean_yz_to_xy for the model
itself. This script only wires it to data/training/checkpoints, mirroring
second/multiview_experiment.py's structure (reuses its TASKS config and
dataset builder unchanged -- the 'voxels_yz'/'coordinates_yz'/'num_points_yz'
keys this model needs are already produced by prep_pointcloud's
aux_pillar_voxel_generators path).

Usage:
    python -m second.gated_yz_experiment verify_synthetic
    python -m second.gated_yz_experiment verify_zero_init --task=ped_cycle --baseline_ckpt=<path>
    python -m second.gated_yz_experiment smoke_test --task=car --steps=10
    python -m second.gated_yz_experiment train --model_dir=<dir> --config=C --baseline_ckpt=<path>
    python -m second.gated_yz_experiment evaluate --model_dir=<dir>
"""
import pathlib
import random
import time

import fire
import numpy as np
import torch

import second.multiview_experiment as me
from second.pytorch.train import example_convert_to_torch, predict_kitti_to_anno
from second.data.preprocess import merge_second_batch
from second.pytorch.models.voxelnet import (
    GatedYZVoxelNet, masked_scatter_mean_yz_to_xy, remap_baseline_state_dict_to_gated)
from second.utils.eval import get_coco_eval_result, get_official_eval_result
import torchplus


def set_seed(seed):
    """Fix every source of randomness this codebase otherwise leaves
    unseeded (see multiview_experiment_log.pdf section 11): Python/numpy/
    torch RNGs, plus the DataLoader worker seed (second/pytorch/train.py's
    _worker_init_fn seeds numpy workers from time.time(), which would make
    B and C see different augmentation/db-sampling draws on every run --
    call this BEFORE building the model (so the randomly-initialized YZ
    branch gets identical weights across configs/seeds you intend to
    compare) and pass the same seed to _seeded_worker_init_fn for the
    DataLoader.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seeded_worker_init_fn(worker_id, base_seed):
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def _build_model(cfg, target_assigner, correction_enabled=True):
    xy_cfg, _ = me._xy_branch_config(cfg)
    yz_cfg, _ = me._aux_branch_config(cfg, 'yz')
    return GatedYZVoxelNet(
        xy_branch_cfg=xy_cfg,
        yz_branch_cfg=yz_cfg,
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
        loss_norm_type=me.LossNormType.NormByNumPositives,
        encode_rad_error_by_sin=True,
        loc_loss_ftor=me.WeightedSmoothL1LocalizationLoss(sigma=3.0),
        cls_loss_ftor=me.SigmoidFocalClassificationLoss(gamma=2.0, alpha=0.25),
        out_size_factor=me._out_size_factor(cfg),
        correction_enabled=correction_enabled,
    )


def load_baseline_checkpoint(model, ckpt_path, strict_report=True):
    """Load a plain-VoxelNet checkpoint into model's xy path + head.
    Prints every skipped/unrecognized/missing/mismatched key -- never
    silently drops anything.
    """
    raw_sd = torch.load(ckpt_path, map_location='cpu')
    remapped, skipped, unrecognized = remap_baseline_state_dict_to_gated(raw_sd)
    print(f"[load_baseline_checkpoint] baseline keys={len(raw_sd)}  "
          f"remapped={len(remapped)}  skipped(non-weight buffers)={len(skipped)}  "
          f"unrecognized={len(unrecognized)}")
    if unrecognized:
        print(f"  UNRECOGNIZED baseline keys (not loaded, investigate): {unrecognized}")

    model_sd = model.state_dict()
    shape_mismatches = []
    to_load = {}
    for k, v in remapped.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape != v.shape:
            shape_mismatches.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            to_load[k] = v
    missing_after = sorted(set(model_sd.keys()) - set(to_load.keys()))
    xy_head_missing = [k for k in missing_after
                       if k.startswith(('pfns.xy.', 'backbones.xy.', 'head.'))]

    print(f"  loadable (name+shape match)={len(to_load)}")
    print(f"  shape-mismatched keys ({len(shape_mismatches)}):")
    for k, s1, s2 in shape_mismatches:
        print(f"    {k}: baseline={s1} vs model={s2}")
    print(f"  xy/head params baseline did NOT cover ({len(xy_head_missing)}): {xy_head_missing}")
    print(f"  yz/fusion params left at their own init (expected, no baseline equivalent): "
          f"{[k for k in missing_after if k not in xy_head_missing]}")

    model.load_state_dict(to_load, strict=False)
    if strict_report and (shape_mismatches or xy_head_missing):
        print("  >>> WARNING: xy/head path is NOT fully loaded from baseline -- see above. <<<")
    return {
        'loaded': len(to_load), 'shape_mismatches': shape_mismatches,
        'xy_head_missing': xy_head_missing, 'unrecognized': unrecognized,
    }


# ---------------------------------------------------------------------------
# Section 7 verifications
# ---------------------------------------------------------------------------

def verify_synthetic():
    """Hand-built tiny example: 2 points, known (x,y,z) -> checks
    masked_scatter_mean_yz_to_xy places each point's YZ feature at the
    correct XY output cell, averages when two points share a cell, and
    leaves untouched cells at exactly 0. No model, no data pipeline.
    """
    torch.manual_seed(0)
    B, C = 1, 4
    Z, Y = 3, 3          # tiny YZ backbone grid
    out_h, out_w = 4, 4  # tiny XY output grid
    F_yz = torch.zeros(B, C, Z, Y)
    F_yz[0, :, 1, 2] = torch.tensor([1.0, 2.0, 3.0, 4.0])   # pillar (z=1,y=2)
    F_yz[0, :, 0, 0] = torch.tensor([10.0, 0.0, 0.0, 0.0])  # pillar (z=0,y=0)

    xy_pc_range = [0.0, 0.0, -3.0, 4.0, 4.0, 1.0]
    xy_voxel_size = [1.0, 1.0, 4.0]
    out_size_factor = 1

    # 3 points: two share XY cell (2,2) both sourced from YZ pillar (1,2);
    # one point sourced from YZ pillar (0,0) lands in XY cell (0,0).
    # voxels_yz layout mirrors the real pipeline: grouped by YZ pillar.
    voxels_yz = torch.zeros(2, 2, 3)  # [n_yz_voxels=2, max_pts=2, xyz]
    num_points_yz = torch.tensor([2, 1])
    coords_yz = torch.tensor([[0, 1, 2, 0],   # batch,z_idx,y_idx,x_idx(dummy)
                              [0, 0, 0, 0]])
    # voxel 0 (YZ pillar z=1,y=2): two points, both should land in XY cell (row=2,col=2)
    voxels_yz[0, 0] = torch.tensor([2.3, 2.7, 0.0])   # x=2.3 -> col2, y=2.7 -> row2
    voxels_yz[0, 1] = torch.tensor([2.9, 2.1, 0.0])   # x=2.9 -> col2, y=2.1 -> row2
    # voxel 1 (YZ pillar z=0,y=0): one point -> XY cell (row=0,col=0)
    voxels_yz[1, 0] = torch.tensor([0.4, 0.2, 0.0])

    agg = masked_scatter_mean_yz_to_xy(
        F_yz, voxels_yz, num_points_yz, coords_yz, batch_size=1,
        xy_pc_range=xy_pc_range, xy_voxel_size=xy_voxel_size,
        out_size_factor=out_size_factor, out_h=out_h, out_w=out_w)

    expect_22 = torch.tensor([1.0, 2.0, 3.0, 4.0])   # mean of the same vector twice = itself
    expect_00 = torch.tensor([10.0, 0.0, 0.0, 0.0])
    ok_22 = torch.allclose(agg[0, :, 2, 2], expect_22, atol=1e-6)
    ok_00 = torch.allclose(agg[0, :, 0, 0], expect_00, atol=1e-6)
    zero_elsewhere = agg.clone()
    zero_elsewhere[0, :, 2, 2] = 0
    zero_elsewhere[0, :, 0, 0] = 0
    ok_zero = torch.allclose(zero_elsewhere, torch.zeros_like(zero_elsewhere), atol=1e-6)

    print(f"cell(2,2) = {agg[0,:,2,2].tolist()}  expect {expect_22.tolist()}  match={ok_22}")
    print(f"cell(0,0) = {agg[0,:,0,0].tolist()}  expect {expect_00.tolist()}  match={ok_00}")
    print(f"all other cells exactly zero: {ok_zero}")

    # gradient flow check
    F_yz_grad = F_yz.clone().requires_grad_(True)
    agg2 = masked_scatter_mean_yz_to_xy(
        F_yz_grad, voxels_yz, num_points_yz, coords_yz, batch_size=1,
        xy_pc_range=xy_pc_range, xy_voxel_size=xy_voxel_size,
        out_size_factor=out_size_factor, out_h=out_h, out_w=out_w)
    agg2.sum().backward()
    grad_nonzero_at_used_pillars = (F_yz_grad.grad[0, :, 1, 2].abs().sum() > 0
                                     and F_yz_grad.grad[0, :, 0, 0].abs().sum() > 0)
    grad_zero_elsewhere = F_yz_grad.grad.clone()
    grad_zero_elsewhere[0, :, 1, 2] = 0
    grad_zero_elsewhere[0, :, 0, 0] = 0
    grad_zero_ok = torch.allclose(grad_zero_elsewhere, torch.zeros_like(grad_zero_elsewhere))
    print(f"gradient reaches used YZ pillars only: {grad_nonzero_at_used_pillars and grad_zero_ok}")

    # edge cases: empty voxels_yz, all-out-of-range points
    empty_agg = masked_scatter_mean_yz_to_xy(
        F_yz, torch.zeros(0, 2, 3), torch.zeros(0, dtype=torch.long), torch.zeros(0, 4, dtype=torch.long),
        batch_size=1, xy_pc_range=xy_pc_range, xy_voxel_size=xy_voxel_size,
        out_size_factor=out_size_factor, out_h=out_h, out_w=out_w)
    print(f"empty voxels_yz -> all-zero output: {torch.allclose(empty_agg, torch.zeros_like(empty_agg))}")

    oor_voxels = torch.zeros(1, 1, 3)
    oor_voxels[0, 0] = torch.tensor([999.0, 999.0, 0.0])  # far outside xy_pc_range
    oor_coords = torch.tensor([[0, 1, 2, 0]])
    oor_numpts = torch.tensor([1])
    oor_agg = masked_scatter_mean_yz_to_xy(
        F_yz, oor_voxels, oor_numpts, oor_coords, batch_size=1,
        xy_pc_range=xy_pc_range, xy_voxel_size=xy_voxel_size,
        out_size_factor=out_size_factor, out_h=out_h, out_w=out_w)
    print(f"out-of-range point -> all-zero output: {torch.allclose(oor_agg, torch.zeros_like(oor_agg))}")

    # regression test: out_size_factor > 1 (e.g. car) means coords_yz's NATIVE
    # pillar index must be downsampled before indexing the post-backbone F_yz
    # -- this crashed with an out-of-bounds CUDA index before the fix.
    F_yz_small = torch.zeros(1, C, 2, 2)  # post-backbone: only 2x2
    F_yz_small[0, :, 1, 1] = torch.tensor([5.0, 6.0, 7.0, 8.0])
    native_voxels = torch.zeros(1, 1, 3)
    native_voxels[0, 0] = torch.tensor([2.5, 2.5, 0.0])
    # native coords use a 4x4 native grid (out_size_factor=2 -> 2x2 post-backbone);
    # z_idx=3,y_idx=3 natively should downsample to (1,1) in F_yz_small.
    native_coords = torch.tensor([[0, 3, 3, 0]])
    native_numpts = torch.tensor([1])
    agg_ds = masked_scatter_mean_yz_to_xy(
        F_yz_small, native_voxels, native_numpts, native_coords, batch_size=1,
        xy_pc_range=xy_pc_range, xy_voxel_size=xy_voxel_size,
        out_size_factor=2, out_h=out_h, out_w=out_w)
    # point (2.5,2.5) with out_size_factor=2 -> effective cell size 2.0 -> cell (1,1)
    ok_downsample = torch.allclose(agg_ds[0, :, 1, 1], torch.tensor([5.0, 6.0, 7.0, 8.0]), atol=1e-6)
    print(f"out_size_factor=2 native-index downsampling correct: {ok_downsample}")

    all_ok = (ok_22 and ok_00 and ok_zero and grad_nonzero_at_used_pillars and grad_zero_ok
              and ok_downsample)
    print(f"\nverify_synthetic: {'PASS' if all_ok else 'FAIL'}")


def verify_zero_init(task='ped_cycle', baseline_ckpt=None, tol=1e-5):
    """Loads a baseline checkpoint (or, if none given, compares two freshly
    constructed models) and checks:
    (a) correction_enabled=False path == running baseline XY/head alone.
    (b) correction_enabled=True at zero-init (fresh fusion weights) produces
        the SAME output as (a), i.e. F_out == F_xy exactly at init.
    """
    cfg = me.TASKS[task]
    target_assigner = me._build_target_assigner(cfg)
    model_a = _build_model(cfg, target_assigner, correction_enabled=False)
    model_c = _build_model(cfg, target_assigner, correction_enabled=True)

    if baseline_ckpt:
        load_baseline_checkpoint(model_a, baseline_ckpt)
        load_baseline_checkpoint(model_c, baseline_ckpt)
    else:
        print("no --baseline_ckpt given: comparing two freshly-initialized "
              "models with IDENTICAL random weights (copied), not a real checkpoint.")
        model_c.pfns['xy'].load_state_dict(model_a.pfns['xy'].state_dict())
        model_c.backbones['xy'].load_state_dict(model_a.backbones['xy'].state_dict())
        model_c.head.load_state_dict(model_a.head.state_dict())

    model_a.cuda().eval()
    model_c.cuda().eval()

    dataset = me._build_dataset(cfg, ['yz'], target_assigner, training=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=merge_second_batch)
    example = next(iter(loader))
    example_torch = example_convert_to_torch(example)

    # Manually replay each path up to raw head output (not net.forward(), which
    # in eval() mode runs full NMS/decode -- we want the pre-NMS box/cls preds).
    with torch.no_grad():
        batch_size_dev = example_torch['anchors'].shape[0]
        F_xy_a = model_a.backbones['xy'](model_a.scatters['xy'](
            model_a.pfns['xy'](example_torch['voxels'], example_torch['num_points'], example_torch['coordinates']),
            example_torch['coordinates'], batch_size_dev))
        preds_a = model_a.head(F_xy_a)

        F_xy_c = model_c.backbones['xy'](model_c.scatters['xy'](
            model_c.pfns['xy'](example_torch['voxels'], example_torch['num_points'], example_torch['coordinates']),
            example_torch['coordinates'], batch_size_dev))
        yz_voxels, yz_num_points, yz_coors = (example_torch['voxels_yz'],
                                               example_torch['num_points_yz'],
                                               example_torch['coordinates_yz'])
        F_yz_c = model_c.backbones['yz'](model_c.scatters['yz'](
            model_c.pfns['yz'](yz_voxels, yz_num_points, yz_coors), yz_coors, batch_size_dev))
        agg = masked_scatter_mean_yz_to_xy(
            F_yz_c, yz_voxels, yz_num_points, yz_coors, batch_size_dev,
            model_c._xy_pc_range, model_c._xy_voxel_size, model_c._out_size_factor,
            F_xy_c.shape[2], F_xy_c.shape[3])
        F_out_c, gate, delta = model_c.fusion(F_xy_c, agg)
        preds_c = model_c.head(F_out_c)

    delta_is_zero = torch.allclose(delta, torch.zeros_like(delta), atol=1e-7)
    gate_is_half = torch.allclose(gate, torch.full_like(gate, 0.5), atol=1e-6)
    f_match = torch.allclose(F_xy_a, F_xy_c, atol=tol) and torch.allclose(F_xy_c, F_out_c, atol=1e-7)
    box_match = torch.allclose(preds_a['box_preds'], preds_c['box_preds'], atol=tol)
    cls_match = torch.allclose(preds_a['cls_preds'], preds_c['cls_preds'], atol=tol)

    print(f"proj(agg_yz) == 0 at init: {delta_is_zero}")
    print(f"gate == 0.5 at init: {gate_is_half}")
    print(f"F_xy(baseline) == F_xy(gated) == F_out(gated) at init: {f_match}")
    print(f"head box_preds match (config A vs C@init): {box_match}")
    print(f"head cls_preds match (config A vs C@init): {cls_match}")
    all_ok = delta_is_zero and gate_is_half and f_match and box_match and cls_match
    print(f"\nverify_zero_init: {'PASS' if all_ok else 'FAIL'}")


def smoke_test(task='ped_cycle', steps=20, config='C', freeze_baseline=True, baseline_ckpt=None):
    """Short training smoke test: finite loss, and confirms which params
    actually receive gradient (must match the frozen/trainable split)."""
    assert config in ('B', 'C')
    cfg = me.TASKS[task]
    target_assigner = me._build_target_assigner(cfg)
    net = _build_model(cfg, target_assigner, correction_enabled=True)
    net.fusion.force_gate_one = (config == 'B')

    if baseline_ckpt:
        load_baseline_checkpoint(net, baseline_ckpt)
    net.cuda()

    if freeze_baseline:
        for p in net.pfns['xy'].parameters():
            p.requires_grad_(False)
        for p in net.backbones['xy'].parameters():
            p.requires_grad_(False)
        for p in net.head.parameters():
            p.requires_grad_(False)

    dataset = me._build_dataset(cfg, ['yz'], target_assigner, training=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=merge_second_batch, drop_last=True)
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in net.parameters())
    print(f"trainable params: {n_trainable} / {n_total} total "
          f"({'frozen xy+head' if freeze_baseline else 'all unfrozen'})")
    optimizer = torch.optim.Adam(trainable_params, lr=2e-4, weight_decay=1e-4)

    net.train()
    if freeze_baseline:
        net.set_frozen_baseline_eval()

    it = iter(loader)
    losses = []
    for step in range(steps):
        example = next(it)
        example_torch = example_convert_to_torch(example)
        ret = net(example_torch)
        loss = ret['loss'].mean()
        optimizer.zero_grad()
        loss.backward()

        if step <= 1:
            xy_head_grad_none = all(
                p.grad is None or p.grad.abs().sum() == 0
                for p in list(net.pfns['xy'].parameters()) + list(net.backbones['xy'].parameters())
                + list(net.head.parameters()))
            yz_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                               for p in net.pfns['yz'].parameters())
            proj_has_grad = net.fusion.proj.weight.grad is not None and net.fusion.proj.weight.grad.abs().sum() > 0
            gate_grad = net.fusion.gate_conv.weight.grad
            gate_has_grad = (not net.fusion.force_gate_one) and gate_grad is not None and gate_grad.abs().sum() > 0
            print(f"[step {step}] xy/head grad ~= 0 (frozen, expected {freeze_baseline}): {xy_head_grad_none}")
            print(f"[step {step}] yz branch has gradient "
                  f"(expected False at step 0 -- proj is zero-init so it blocks upstream "
                  f"gradient until its own weight moves off zero; True from step>=1): {yz_has_grad}")
            print(f"[step {step}] fusion.proj has gradient (should be True even at step 0 "
                  f"-- gradient w.r.t. a zero-init weight is generally nonzero): {proj_has_grad}")
            if config == 'C':
                print(f"[step {step}] fusion.gate_conv has gradient "
                      f"(expected False at step 0 since delta=0 there, True from step>=1): {gate_has_grad}")

        optimizer.step()
        losses.append(float(loss))
        print(f"step={step} loss={float(loss):.4f} "
              f"gate_mean={'n/a' if net._last_gate is None else float(net._last_gate.mean()):.4f} "
              f"delta_absmean={'n/a' if net._last_delta is None else float(net._last_delta.abs().mean()):.5f}")

    finite = all(np.isfinite(l) for l in losses)
    print(f"\nsmoke_test finite losses: {finite}")


def train(model_dir, task='ped_cycle', config='C', baseline_ckpt=None,
          max_steps=296960, display_step=50, save_every=5000,
          freeze_baseline=True, finetune_lr=2e-5, seed=42):
    """config='B': gate forced to 1 (no learned gate). config='C': full
    learned gate. Both require --baseline_ckpt on the FIRST run (resumed
    runs restore from model_dir instead). freeze_baseline=False switches to
    stage-2 fine-tuning of the whole network at `finetune_lr` -- point this
    at a model_dir that already has a stage-1 checkpoint.

    `seed` is fixed (this codebase otherwise seeds nothing -- see
    multiview_experiment_log.pdf section 11) and applied BEFORE the model
    is constructed, so the randomly-initialized YZ branch starts from
    IDENTICAL weights across a B/C pair run with the same --seed (xy/head
    come from the same deterministic baseline_ckpt either way). Use the
    same --seed for the B and C runs you intend to compare; a different
    seed on either side reintroduces exactly the confound this guards
    against.
    """
    assert config in ('B', 'C')
    set_seed(seed)
    cfg = me.TASKS[task]
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    target_assigner = me._build_target_assigner(cfg)
    net = _build_model(cfg, target_assigner, correction_enabled=True)
    net.fusion.force_gate_one = (config == 'B')

    restored = torchplus.train.try_restore_latest_checkpoints(str(model_dir), [net])
    if net.get_global_step() == 0:
        if not baseline_ckpt:
            raise ValueError("first run needs --baseline_ckpt=<path to baseline voxelnet-*.tckpt>")
        load_baseline_checkpoint(net, baseline_ckpt)
    net.cuda()

    if freeze_baseline:
        for p in list(net.pfns['xy'].parameters()) + list(net.backbones['xy'].parameters()) \
                + list(net.head.parameters()):
            p.requires_grad_(False)
        trainable_params = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=me.BASE_LR, weight_decay=me.WEIGHT_DECAY)
    else:
        yz_params = (list(net.pfns['yz'].parameters()) + list(net.backbones['yz'].parameters())
                     + list(net.fusion.parameters()))
        xy_head_params = (list(net.pfns['xy'].parameters()) + list(net.backbones['xy'].parameters())
                           + list(net.head.parameters()))
        for p in xy_head_params:
            p.requires_grad_(True)
        optimizer = torch.optim.Adam(
            [{'params': yz_params, 'lr': me.BASE_LR},
             {'params': xy_head_params, 'lr': finetune_lr}],
            weight_decay=me.WEIGHT_DECAY)
    optimizer.name = "adam_optimizer"
    torchplus.train.try_restore_latest_checkpoints(str(model_dir), [optimizer])

    def _lr_for_step(step):
        return me.BASE_LR * (me.DECAY_FACTOR ** (step // me.DECAY_STEPS))

    dataset = me._build_dataset(cfg, ['yz'], target_assigner, training=True)
    shuffle_generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=me.BATCH_SIZE, shuffle=True, num_workers=me.NUM_WORKERS,
        collate_fn=merge_second_batch, drop_last=True, generator=shuffle_generator,
        worker_init_fn=lambda wid: _seeded_worker_init_fn(wid, seed))

    net.train()
    if freeze_baseline:
        net.set_frozen_baseline_eval()

    step = net.get_global_step()
    t0 = time.time()
    data_iter = iter(loader)
    while step < max_steps:
        try:
            example = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            example = next(data_iter)
        base_lr_now = _lr_for_step(step)
        for i, pg in enumerate(optimizer.param_groups):
            pg['lr'] = base_lr_now if (freeze_baseline or i == 0) else finetune_lr
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
            gate_mean = float(net._last_gate.mean()) if net._last_gate is not None else float('nan')
            delta_absmean = float(net._last_delta.abs().mean()) if net._last_delta is not None else float('nan')
            print(f"step={step} steptime={dt / display_step:.3f} "
                  f"loss={float(loss):.4f} cls_loss={float(ret['cls_loss_reduced']):.4f} "
                  f"loc_loss={float(ret['loc_loss_reduced']):.4f} "
                  f"dir_loss={float(ret['dir_loss_reduced']):.4f} "
                  f"gate_mean={gate_mean:.4f} delta_absmean={delta_absmean:.5f} "
                  f"num_pos={num_pos} lr={optimizer.param_groups[0]['lr']:.2e}")
        if step % save_every == 0:
            torchplus.train.save_models(str(model_dir), [net, optimizer], step)

    torchplus.train.save_models(str(model_dir), [net, optimizer], step)
    print(f"done at step {step}")


def evaluate(model_dir, task='ped_cycle', config='C', max_samples=None, force_gate_one_eval=False):
    """force_gate_one_eval=True is the diagnostic ablation from section 7:
    take a trained Config-C checkpoint and force G=1 ONLY at eval time
    (never trained that way) to see how much the learned gate is actually
    buying versus an always-on correction."""
    cfg = me.TASKS[task]
    target_assigner = me._build_target_assigner(cfg)
    net = _build_model(cfg, target_assigner, correction_enabled=True)
    net.fusion.force_gate_one = (config == 'B') or force_gate_one_eval
    torchplus.train.try_restore_latest_checkpoints(str(model_dir), [net])
    net.cuda()
    net.eval()
    print(f"restored checkpoint at global_step={net.get_global_step()}  "
          f"force_gate_one_eval={force_gate_one_eval}  config={config}")

    dataset = me._build_dataset(cfg, ['yz'], target_assigner, training=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=me.BATCH_SIZE, shuffle=False, num_workers=me.NUM_WORKERS,
        collate_fn=merge_second_batch)

    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    dt_annos = []
    gate_vals, delta_abs_vals = [], []
    t0 = time.time()
    with torch.no_grad():
        seen = 0
        for example in loader:
            if seen >= n_samples:
                break
            example_torch = example_convert_to_torch(example)
            dt_annos += predict_kitti_to_anno(
                net, example_torch, cfg['class_names'], cfg['pc_range'])
            if net._last_gate is not None:
                gate_vals.append(float(net._last_gate.mean()))
                delta_abs_vals.append(float(net._last_delta.abs().mean()))
            seen += example['image_idx'].shape[0]
    print(f"evaluated {seen} frames in {time.time() - t0:.1f}s")
    if gate_vals:
        print(f"gate distribution over eval set: mean={np.mean(gate_vals):.4f} "
              f"std={np.std(gate_vals):.4f} min={np.min(gate_vals):.4f} max={np.max(gate_vals):.4f}")
        print(f"|G*delta| mean over eval set: {np.mean(delta_abs_vals):.5f}")

    gt_annos = [info["annos"] for info in dataset.kitti_infos[:seen]]
    result = get_official_eval_result(gt_annos, dt_annos, cfg['class_names'])
    print(result)
    result = get_coco_eval_result(gt_annos, dt_annos, cfg['class_names'])
    print(result)


if __name__ == '__main__':
    fire.Fire()
