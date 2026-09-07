"""Utilities for the pillar-axis ablation study (PP-XY / PP-XZ / PP-YZ).

The rest of the codebase (data augmentation, ground-truth boxes, anchor
generation, box coding, NMS, KITTI eval) all operate in real LiDAR (x, y, z)
coordinates and assume the detection plane is XY. We keep all of that
untouched. The only thing "view" changes is which two axes the point cloud
gets grouped into pillars on; the resulting pseudo-image is resized back to
the canonical XY grid before it reaches the RPN, so every anchor/matching/NMS
code path sees exactly the tensor shapes it does today.

Axis indices follow the usual LiDAR convention: 0=x, 1=y, 2=z.
"""

VIEW_SPATIAL_AXES = {
    'xy': (0, 1),  # baseline PointPillars: collapse Z
    'xz': (0, 2),  # PP-XZ: collapse Y
    'yz': (1, 2),  # PP-YZ: collapse X
}


def spatial_axes_for_view(view):
    if view not in VIEW_SPATIAL_AXES:
        raise ValueError(f"unknown view {view!r}, must be one of {list(VIEW_SPATIAL_AXES)}")
    return VIEW_SPATIAL_AXES[view]


def make_pillar_voxel_config(voxel_size, point_cloud_range, view):
    """Derive the (voxel_size, spatial_axes) used to actually form pillars.

    `voxel_size`/`point_cloud_range` are the canonical XY config values
    (e.g. from the .proto). `point_cloud_range` is never changed -- it is a
    real-world extent, independent of which plane we pillar on. Only
    `voxel_size` changes: the axis being collapsed gets a bin spanning its
    full range (so it grids to a single cell, exactly how the existing XY
    config collapses Z), and any newly-spatial axis that didn't have a
    "pillar-sized" resolution in the canonical config (i.e. Z, whose
    canonical voxel_size just spans the whole height range) is given the
    same fine resolution used for X/Y.
    """
    spatial_axes = spatial_axes_for_view(view)
    collapsed_axis = ({0, 1, 2} - set(spatial_axes)).pop()

    fine_res = voxel_size[0]
    pillar_voxel_size = list(voxel_size)
    for axis in spatial_axes:
        # Z's canonical voxel_size is sized to collapse the whole height
        # range; if Z becomes spatial here it needs a real pillar resolution.
        if pillar_voxel_size[axis] >= (point_cloud_range[axis + 3] - point_cloud_range[axis]):
            pillar_voxel_size[axis] = fine_res

    span = point_cloud_range[collapsed_axis + 3] - point_cloud_range[collapsed_axis]
    pillar_voxel_size[collapsed_axis] = span + 1.0  # +1 margin: guarantee exactly one bin

    return pillar_voxel_size, spatial_axes
