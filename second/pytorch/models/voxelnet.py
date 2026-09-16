import time
from enum import Enum
from functools import reduce

import numpy as np
try:
    import sparseconvnet as scn
except ImportError:
    # Only needed by the sparse-conv ("SECOND") middle extractor; PointPillars
    # uses PointPillarsScatter instead and never touches this module.
    scn = None
import torch
from torch import nn
from torch.nn import functional as F

import torchplus
from torchplus import metrics
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args
from second.pytorch.core import box_torch_ops
from second.pytorch.core.losses import (WeightedSigmoidClassificationLoss,
                                          WeightedSmoothL1LocalizationLoss,
                                          WeightedSoftmaxClassificationLoss)
from second.pytorch.models.pointpillars import PillarFeatureNet, PointPillarsScatter
from second.pytorch.utils import get_paddings_indicator


def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, name='vfe'):
        super(VFELayer, self).__init__()
        self.name = name
        self.units = int(out_channels / 2)
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        pointwise = F.relu(x)
        # [K, T, units]

        aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        # [K, 1, units]
        repeated = aggregated.repeat(1, voxel_count, 1)

        concatenated = torch.cat([pointwise, repeated], dim=2)
        # [K, T, 2 * units]
        return concatenated


class VoxelFeatureExtractor(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        assert len(num_filters) == 2
        num_input_features += 3  # add mean features
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        self.vfe1 = VFELayer(num_input_features, num_filters[0], use_norm)
        self.vfe2 = VFELayer(num_filters[0], num_filters[1], use_norm)
        self.linear = Linear(num_filters[1], num_filters[1])
        # var_torch_init(self.linear.weight)
        # var_torch_init(self.linear.bias)
        self.norm = BatchNorm1d(num_filters[1])

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat(
                [features, features_relative, points_dist], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        # mask = features.max(dim=2, keepdim=True)[0] != 0
        x = self.vfe1(features)
        x *= mask
        x = self.vfe2(x)
        x *= mask
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = F.relu(x)
        x *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = torch.max(x, dim=1)[0]
        return voxelwise


class VoxelFeatureExtractorV2(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractorV2, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        assert len(num_filters) > 0
        num_input_features += 3
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        num_filters = [num_input_features] + num_filters
        filters_pairs = [[num_filters[i], num_filters[i + 1]]
                         for i in range(len(num_filters) - 1)]
        self.vfe_layers = nn.ModuleList(
            [VFELayer(i, o, use_norm) for i, o in filters_pairs])
        self.linear = Linear(num_filters[-1], num_filters[-1])
        # var_torch_init(self.linear.weight)
        # var_torch_init(self.linear.bias)
        self.norm = BatchNorm1d(num_filters[-1])

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat(
                [features, features_relative, points_dist], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        for vfe in self.vfe_layers:
            features = vfe(features)
            features *= mask
        features = self.linear(features)
        features = self.norm(features.permute(0, 2, 1).contiguous()).permute(
            0, 2, 1).contiguous()
        features = F.relu(features)
        features *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = torch.max(features, dim=1)[0]
        return voxelwise


class SparseMiddleExtractor(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SparseMiddleExtractor'):
        super(SparseMiddleExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        self.voxel_output_shape = output_shape
        middle_layers = []

        num_filters = [num_input_features] + num_filters_down1
        # num_filters = [64] + num_filters_down1
        filters_pairs_d1 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]

        for i, o in filters_pairs_d1:
            middle_layers.append(scn.SubmanifoldConvolution(3, i, o, 3, False))
            middle_layers.append(scn.BatchNormReLU(o, eps=1e-3, momentum=0.99))
        middle_layers.append(
            scn.Convolution(
                3,
                num_filters[-1],
                num_filters[-1], (3, 1, 1), (2, 1, 1),
                bias=False))
        middle_layers.append(
            scn.BatchNormReLU(num_filters[-1], eps=1e-3, momentum=0.99))
        # assert len(num_filters_down2) > 0
        if len(num_filters_down1) == 0:
            num_filters = [num_filters[-1]] + num_filters_down2
        else:
            num_filters = [num_filters_down1[-1]] + num_filters_down2
        filters_pairs_d2 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]
        for i, o in filters_pairs_d2:
            middle_layers.append(scn.SubmanifoldConvolution(3, i, o, 3, False))
            middle_layers.append(scn.BatchNormReLU(o, eps=1e-3, momentum=0.99))
        middle_layers.append(
            scn.Convolution(
                3,
                num_filters[-1],
                num_filters[-1], (3, 1, 1), (2, 1, 1),
                bias=False))
        middle_layers.append(
            scn.BatchNormReLU(num_filters[-1], eps=1e-3, momentum=0.99))
        middle_layers.append(scn.SparseToDense(3, num_filters[-1]))
        self.middle_conv = Sequential(*middle_layers)

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()[:, [1, 2, 3, 0]]
        ret = self.scn_input((coors.cpu(), voxel_features, batch_size))
        ret = self.middle_conv(ret)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class ZeroPad3d(nn.ConstantPad3d):
    def __init__(self, padding):
        super(ZeroPad3d, self).__init__(padding, 0)


class MiddleExtractor(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='MiddleExtractor'):
        super(MiddleExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm3d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm3d)
            # BatchNorm3d = change_default_args(
            #     group=32, eps=1e-3, momentum=0.01)(GroupBatchNorm3d)
            Conv3d = change_default_args(bias=False)(nn.Conv3d)
        else:
            BatchNorm3d = Empty
            Conv3d = change_default_args(bias=True)(nn.Conv3d)
        self.voxel_output_shape = output_shape
        self.middle_conv = Sequential(
            ZeroPad3d(1),
            Conv3d(num_input_features, 64, 3, stride=(2, 1, 1)),
            BatchNorm3d(64),
            nn.ReLU(),
            ZeroPad3d([1, 1, 1, 1, 0, 0]),
            Conv3d(64, 64, 3, stride=1),
            BatchNorm3d(64),
            nn.ReLU(),
            ZeroPad3d(1),
            Conv3d(64, 64, 3, stride=(2, 1, 1)),
            BatchNorm3d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        output_shape = [batch_size] + self.voxel_output_shape[1:]
        ret = scatter_nd(coors.long(), voxel_features, output_shape)
        # print('scatter_nd fw:', time.time() - t)
        ret = ret.permute(0, 4, 1, 2, 3)
        ret = self.middle_conv(ret)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret


class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_filters=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 name='rpn'):
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        if use_bev:
            self.bev_extractor = Sequential(
                Conv2d(6, 32, 3, padding=1),
                BatchNorm2d(32),
                nn.ReLU(),
                # nn.MaxPool2d(2, 2),
                Conv2d(32, 64, 3, padding=1),
                BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            block2_input_filters += 64

        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_filters, num_filters[0], 3, stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1)

    def forward(self, x, bev=None):
        x = self.block1(x)
        up1 = self.deconv1(x)
        if self._use_bev:
            bev[:, -1] = torch.clamp(
                torch.log(1 + bev[:, -1]) / np.log(16.0), max=1.0)
            x = torch.cat([x, self.bev_extractor(bev)], dim=1)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict


class RPNBackbone(nn.Module):
    """The conv/deconv feature-pyramid part of `RPN`, without the detection
    heads (conv_cls/conv_box/conv_dir_cls) -- i.e. exactly RPN's block1-3 +
    deconv1-3 + concat, extracted so it can be instantiated once per branch
    in a multi-view fusion model (see `MultiViewVoxelNet`). This is a NEW,
    standalone class -- `RPN` itself is untouched, so the existing
    single-branch path (and its checkpoints) are unaffected by this file.
    """
    def __init__(self,
                 use_norm=True,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_filters=128,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 name='rpn_backbone'):
        super().__init__()
        self.name = name
        self._use_bev = use_bev
        self.num_output_filters = sum(num_upsample_filters)
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        block2_input_filters = num_filters[0]
        if use_bev:
            self.bev_extractor = Sequential(
                Conv2d(6, 32, 3, padding=1),
                BatchNorm2d(32),
                nn.ReLU(),
                Conv2d(32, 64, 3, padding=1),
                BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            block2_input_filters += 64

        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_filters, num_filters[0], 3, stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )

    def forward(self, x, bev=None):
        x = self.block1(x)
        up1 = self.deconv1(x)
        if self._use_bev:
            bev[:, -1] = torch.clamp(
                torch.log(1 + bev[:, -1]) / np.log(16.0), max=1.0)
            x = torch.cat([x, self.bev_extractor(bev)], dim=1)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        return torch.cat([up1, up2, up3], dim=1)


class ChannelFoldFusion(nn.Module):
    """Fuses a primary XY backbone feature map with 0+ auxiliary (XZ/YZ)
    backbone feature maps, at native resolution, with no interpolation and
    no pooling-to-a-vector: each auxiliary map's collapsed axis (Z) is
    reshaped into the channel dimension (every Z-bin's channels kept,
    nothing averaged/maxed away) and broadcast (not resized) across
    whichever XY axis that branch doesn't natively have, then concatenated
    and 1x1-conv'd down to `out_channels` for the shared RPN head.

    aux_z_sizes: e.g. {'xz': Z_after_xz_backbone, 'yz': Z_after_yz_backbone}
    -- must be known at construction time to size the reduce-conv.
    """
    def __init__(self, xy_channels, aux_channels, aux_z_sizes, out_channels, debug=False):
        super().__init__()
        self._aux_names = list(aux_z_sizes.keys())
        self._aux_z_sizes = dict(aux_z_sizes)
        self._xy_channels = xy_channels
        self._aux_channels = aux_channels
        self._out_channels = out_channels
        self.debug = debug
        total_in = xy_channels + sum(aux_channels * z for z in aux_z_sizes.values())
        self._total_in = total_in
        self.reduce = nn.Conv2d(total_in, out_channels, 1)

    def forward(self, f_xy, f_xz=None, f_yz=None):
        b, c, Y, X = f_xy.shape
        if self.debug:
            print(f"[ChannelFoldFusion] F_xy'  shape={tuple(f_xy.shape)}")
        parts = [f_xy]
        if f_xz is not None:
            bb, cc, z, x = f_xz.shape
            assert x == X, f"xz width {x} != xy width {X} -- branch backbone strides don't match"
            if self.debug:
                print(f"[ChannelFoldFusion] F_xz'  shape={tuple(f_xz.shape)} "
                      f"(Z'={z}, matches xy width X={x}: {x == X})")
            folded = f_xz.reshape(bb, cc * z, 1, x).expand(bb, cc * z, Y, x)
            if self.debug:
                print(f"[ChannelFoldFusion] xz folded+broadcast shape={tuple(folded.shape)} "
                      f"(channels = C'{cc} x Z'{z} = {cc * z})")
            parts.append(folded)
        if f_yz is not None:
            bb, cc, z, y = f_yz.shape
            assert y == Y, f"yz height {y} != xy height {Y} -- branch backbone strides don't match"
            if self.debug:
                print(f"[ChannelFoldFusion] F_yz'  shape={tuple(f_yz.shape)} "
                      f"(Z'={z}, matches xy height Y={y}: {y == Y})")
            folded = f_yz.reshape(bb, cc * z, 1, y).expand(bb, cc * z, X, y).permute(0, 1, 3, 2)
            if self.debug:
                print(f"[ChannelFoldFusion] yz folded+broadcast shape={tuple(folded.shape)} "
                      f"(channels = C'{cc} x Z'{z} = {cc * z})")
            parts.append(folded)
        fused = torch.cat(parts, dim=1)
        if self.debug:
            print(f"[ChannelFoldFusion] concat shape={tuple(fused.shape)} "
                  f"(expected total_in={self._total_in}: {fused.shape[1] == self._total_in})")
        out = self.reduce(fused)
        if self.debug:
            print(f"[ChannelFoldFusion] after 1x1 conv shape={tuple(out.shape)} "
                  f"(out_channels={self._out_channels})")
        return out


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"


class VoxelNet(nn.Module):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 num_input_features=4,
                 vfe_class_name="VoxelFeatureExtractor",
                 vfe_num_filters=[32, 128],
                 with_distance=False,
                 middle_class_name="SparseMiddleExtractor",
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 rpn_class_name="RPN",
                 rpn_layer_nums=[3, 5, 5],
                 rpn_layer_strides=[2, 2, 2],
                 rpn_num_filters=[128, 128, 256],
                 rpn_upsample_strides=[1, 2, 4],
                 rpn_num_upsample_filters=[256, 256, 256],
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_sparse_rpn=False,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_threshold=0.5,
                 nms_pre_max_size=1000,
                 nms_post_max_size=20,
                 nms_iou_threshold=0.1,
                 target_assigner=None,
                 use_bev=False,
                 lidar_only=False,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 pillar_spatial_axes=(0, 1),
                 canonical_hw=None,
                 name='voxelnet'):
        super().__init__()
        self.name = name
        self._num_class = num_class
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_score_threshold = nms_score_threshold
        self._nms_pre_max_size = nms_pre_max_size
        self._nms_post_max_size = nms_post_max_size
        self._nms_iou_threshold = nms_iou_threshold
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_sparse_rpn = use_sparse_rpn
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0
        self._num_input_features = num_input_features
        self._box_coder = target_assigner.box_coder
        self._lidar_only = lidar_only
        self.target_assigner = target_assigner
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._loss_norm_type = loss_norm_type
        self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()

        self._loc_loss_ftor = loc_loss_ftor
        self._cls_loss_ftor = cls_loss_ftor
        self._direction_loss_weight = direction_loss_weight
        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight
        self._pillar_spatial_axes = tuple(pillar_spatial_axes)
        # (H, W) the RPN/anchors expect; when the pillar grid was formed on a
        # non-XY plane (see second/core/view_transform.py) its native size
        # differs and gets resized to this before entering the RPN.
        self._canonical_hw = tuple(canonical_hw) if canonical_hw is not None else None

        vfe_class_dict = {
            "VoxelFeatureExtractor": VoxelFeatureExtractor,
            "VoxelFeatureExtractorV2": VoxelFeatureExtractorV2,
            "PillarFeatureNet": PillarFeatureNet
        }
        vfe_class = vfe_class_dict[vfe_class_name]
        if vfe_class_name == "PillarFeatureNet":
            self.voxel_feature_extractor = vfe_class(
                num_input_features,
                use_norm,
                num_filters=vfe_num_filters,
                with_distance=with_distance,
                voxel_size=voxel_size,
                pc_range=pc_range,
                spatial_axes=self._pillar_spatial_axes
            )
        else:
            self.voxel_feature_extractor = vfe_class(
                num_input_features,
                use_norm,
                num_filters=vfe_num_filters,
                with_distance=with_distance)

        print("middle_class_name", middle_class_name)
        if middle_class_name == "PointPillarsScatter":
            self.middle_feature_extractor = PointPillarsScatter(output_shape=output_shape,
                                                                num_input_features=vfe_num_filters[-1],
                                                                spatial_axes=self._pillar_spatial_axes)
            num_rpn_input_filters = self.middle_feature_extractor.nchannels
        else:
            mid_class_dict = {
                "MiddleExtractor": MiddleExtractor,
                "SparseMiddleExtractor": SparseMiddleExtractor,
            }
            mid_class = mid_class_dict[middle_class_name]
            self.middle_feature_extractor = mid_class(
                output_shape,
                use_norm,
                num_input_features=vfe_num_filters[-1],
                num_filters_down1=middle_num_filters_d1,
                num_filters_down2=middle_num_filters_d2)
            if len(middle_num_filters_d2) == 0:
                if len(middle_num_filters_d1) == 0:
                    num_rpn_input_filters = int(vfe_num_filters[-1] * 2)
                else:
                    num_rpn_input_filters = int(middle_num_filters_d1[-1] * 2)
            else:
                num_rpn_input_filters = int(middle_num_filters_d2[-1] * 2)

        rpn_class_dict = {
            "RPN": RPN,
        }
        rpn_class = rpn_class_dict[rpn_class_name]
        self.rpn = rpn_class(
            use_norm=True,
            num_class=num_class,
            layer_nums=rpn_layer_nums,
            layer_strides=rpn_layer_strides,
            num_filters=rpn_num_filters,
            upsample_strides=rpn_upsample_strides,
            num_upsample_filters=rpn_num_upsample_filters,
            num_input_filters=num_rpn_input_filters,
            num_anchor_per_loc=target_assigner.num_anchors_per_location,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_bev=use_bev,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=target_assigner.box_coder.code_size)

        self.rpn_acc = metrics.Accuracy(
            dim=-1, encode_background_as_zeros=encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=use_sigmoid_score,
            encode_background_as_zeros=encode_background_as_zeros)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        self.register_buffer("global_step", torch.LongTensor(1).zero_())

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def forward(self, example):
        """module's forward should always accept dict and return loss.
        """
        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        batch_anchors = example["anchors"]
        batch_size_dev = batch_anchors.shape[0]
        t = time.time()
        # features: [num_voxels, max_num_points_per_voxel, 7]
        # num_points: [num_voxels]
        # coors: [num_voxels, 4]
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        if self._use_sparse_rpn:
            preds_dict = self.sparse_rpn(voxel_features, coors, batch_size_dev)
        else:
            spatial_features = self.middle_feature_extractor(
                voxel_features, coors, batch_size_dev)
            if (self._canonical_hw is not None
                    and tuple(spatial_features.shape[-2:]) != self._canonical_hw):
                # Pillars were formed on a non-XY plane (see
                # second/core/view_transform.py): its native pseudo-image
                # resolution doesn't match the canonical XY grid the
                # RPN/anchors expect, so resample it to that size. This is a
                # known, deliberately simple approximation -- it does not
                # attempt to reproject XZ/YZ features into true XY spatial
                # positions.
                spatial_features = F.interpolate(
                    spatial_features, size=self._canonical_hw,
                    mode='bilinear', align_corners=False)
            if self._use_bev:
                preds_dict = self.rpn(spatial_features, example["bev_map"])
            else:
                preds_dict = self.rpn(spatial_features)
        # preds_dict["voxel_features"] = voxel_features
        # preds_dict["spatial_features"] = spatial_features
        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]
        self._total_forward_time += time.time() - t
        if self.training:
            labels = example['labels']
            reg_targets = example['reg_targets']

            cls_weights, reg_weights, cared = prepare_loss_weights(
                labels,
                pos_cls_weight=self._pos_cls_weight,
                neg_cls_weight=self._neg_cls_weight,
                loss_norm_type=self._loss_norm_type,
                dtype=voxels.dtype)
            cls_targets = labels * cared.type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)

            loc_loss, cls_loss = create_loss(
                self._loc_loss_ftor,
                self._cls_loss_ftor,
                box_preds=box_preds,
                cls_preds=cls_preds,
                cls_targets=cls_targets,
                cls_weights=cls_weights,
                reg_targets=reg_targets,
                reg_weights=reg_weights,
                num_class=self._num_class,
                encode_rad_error_by_sin=self._encode_rad_error_by_sin,
                encode_background_as_zeros=self._encode_background_as_zeros,
                box_code_size=self._box_coder.code_size,
            )
            loc_loss_reduced = loc_loss.sum() / batch_size_dev
            loc_loss_reduced *= self._loc_loss_weight
            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
            cls_pos_loss /= self._pos_cls_weight
            cls_neg_loss /= self._neg_cls_weight
            cls_loss_reduced = cls_loss.sum() / batch_size_dev
            cls_loss_reduced *= self._cls_loss_weight
            loss = loc_loss_reduced + cls_loss_reduced
            dir_loss = torch.zeros_like(loss)
            if self._use_direction_classifier:
                dir_targets = get_direction_target(example['anchors'],
                                                   reg_targets)
                dir_logits = preds_dict["dir_cls_preds"].view(
                    batch_size_dev, -1, 2)
                weights = (labels > 0).type_as(dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self._dir_loss_ftor(
                    dir_logits, dir_targets, weights=weights)
                dir_loss = dir_loss.sum() / batch_size_dev
                loss += dir_loss * self._direction_loss_weight

            return {
                "loss": loss,
                "cls_loss": cls_loss,
                "loc_loss": loc_loss,
                "cls_pos_loss": cls_pos_loss,
                "cls_neg_loss": cls_neg_loss,
                "cls_preds": cls_preds,
                "dir_loss_reduced": dir_loss,
                "cls_loss_reduced": cls_loss_reduced,
                "loc_loss_reduced": loc_loss_reduced,
                "cared": cared,
            }
        else:
            return self.predict(example, preds_dict)

    def predict(self, example, preds_dict):
        t = time.time()
        batch_size = example['anchors'].shape[0]
        batch_anchors = example["anchors"].view(batch_size, -1, 7)

        self._total_inference_count += batch_size
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
        batch_imgidx = example['image_idx']

        self._total_forward_time += time.time() - t
        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)
        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1

        batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                               num_class_with_bg)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
                                                       batch_anchors)
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
                batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask
        ):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            if self._use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                # print(dir_preds.shape)
                dir_labels = torch.max(dir_preds, dim=-1)[1]
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            # Apply NMS in birdeye view
            if self._use_rotate_nms:
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms
            selected_boxes = None
            selected_labels = None
            selected_scores = None
            selected_dir_labels = None

            if self._multiclass_nms:
                # curently only support class-agnostic boxes.
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)
                boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
                selected_per_class = box_torch_ops.multiclass_nms(
                    nms_func=nms_func,
                    boxes=boxes_for_mcnms,
                    scores=total_scores,
                    num_class=self._num_class,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                    score_thresh=self._nms_score_threshold,
                )
                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []
                for i, selected in enumerate(selected_per_class):
                    if selected is not None:
                        num_dets = selected.shape[0]
                        selected_boxes.append(box_preds[selected])
                        selected_labels.append(
                            torch.full([num_dets], i, dtype=torch.int64))
                        if self._use_direction_classifier:
                            selected_dir_labels.append(dir_labels[selected])
                        selected_scores.append(total_scores[selected, i])
                if len(selected_boxes) > 0:
                    selected_boxes = torch.cat(selected_boxes, dim=0)
                    selected_labels = torch.cat(selected_labels, dim=0)
                    selected_scores = torch.cat(selected_scores, dim=0)
                    if self._use_direction_classifier:
                        selected_dir_labels = torch.cat(
                            selected_dir_labels, dim=0)
                else:
                    selected_boxes = None
                    selected_labels = None
                    selected_scores = None
                    selected_dir_labels = None
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long)
                else:
                    top_scores, top_labels = torch.max(total_scores, dim=-1)

                if self._nms_score_threshold > 0.0:
                    thresh = torch.tensor(
                        [self._nms_score_threshold],
                        device=total_scores.device).type_as(total_scores)
                    top_scores_keep = (top_scores >= thresh)
                    top_scores = top_scores.masked_select(top_scores_keep)
                if top_scores.shape[0] != 0:
                    if self._nms_score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self._use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                    if not self._use_rotate_nms:
                        box_preds_corners = box_torch_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4])
                        boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                            box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self._nms_pre_max_size,
                        post_max_size=self._nms_post_max_size,
                        iou_threshold=self._nms_iou_threshold,
                    )
                else:
                    selected = None
                if selected is not None:
                    selected_boxes = box_preds[selected]
                    if self._use_direction_classifier:
                        selected_dir_labels = dir_labels[selected]
                    selected_labels = top_labels[selected]
                    selected_scores = top_scores[selected]
            # finally generate predictions.

            if selected_boxes is not None:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))
                    # box_preds[..., -1] += (
                    #     ~(dir_labels.byte())).type_as(box_preds) * np.pi
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c)
                locs = final_box_preds_camera[:, :3]
                dims = final_box_preds_camera[:, 3:6]
                angles = final_box_preds_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_torch_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = box_torch_ops.project_to_image(
                    box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = torch.min(box_corners_in_image, dim=1)[0]
                maxxy = torch.max(box_corners_in_image, dim=1)[0]
                # minx = torch.min(box_corners_in_image[..., 0], dim=1)[0]
                # maxx = torch.max(box_corners_in_image[..., 0], dim=1)[0]
                # miny = torch.min(box_corners_in_image[..., 1], dim=1)[0]
                # maxy = torch.max(box_corners_in_image[..., 1], dim=1)[0]
                # box_2d_preds = torch.stack([minx, miny, maxx, maxy], dim=1)
                box_2d_preds = torch.cat([minxy, maxxy], dim=1)
                # predictions
                predictions_dict = {
                    "bbox": box_2d_preds,
                    "box3d_camera": final_box_preds_camera,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "image_idx": img_idx,
                }
            else:
                predictions_dict = {
                    "bbox": None,
                    "box3d_camera": None,
                    "box3d_lidar": None,
                    "scores": None,
                    "label_preds": None,
                    "image_idx": img_idx,
                }
            predictions_dicts.append(predictions_dict)
        self._total_postprocess_time += time.time() - t
        return predictions_dicts

    @property
    def avg_forward_time(self):
        return self._total_forward_time / self._total_inference_count

    @property
    def avg_postprocess_time(self):
        return self._total_postprocess_time / self._total_inference_count

    def clear_time_metrics(self):
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0

    def metrics_to_float(self):
        self.rpn_acc.float()
        self.rpn_metrics.float()
        self.rpn_cls_loss.float()
        self.rpn_loc_loss.float()
        self.rpn_total_loss.float()

    def update_metrics(self,
                       cls_loss,
                       loc_loss,
                       cls_preds,
                       labels,
                       sampled):
        batch_size = cls_preds.shape[0]
        num_class = self._num_class
        if not self._encode_background_as_zeros:
            num_class += 1
        cls_preds = cls_preds.view(batch_size, -1, num_class)
        rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()[0]
        prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
        prec = prec.numpy()
        recall = recall.numpy()
        rpn_cls_loss = self.rpn_cls_loss(cls_loss).numpy()[0]
        rpn_loc_loss = self.rpn_loc_loss(loc_loss).numpy()[0]
        ret = {
            "cls_loss": float(rpn_cls_loss),
            "cls_loss_rt": float(cls_loss.data.cpu().numpy()),
            'loc_loss': float(rpn_loc_loss),
            "loc_loss_rt": float(loc_loss.data.cpu().numpy()),
            "rpn_acc": float(rpn_acc),
        }
        for i, thresh in enumerate(self.rpn_metrics.thresholds):
            ret[f"prec@{int(thresh*100)}"] = float(prec[i])
            ret[f"rec@{int(thresh*100)}"] = float(recall[i])
        return ret

    def clear_metrics(self):
        self.rpn_acc.clear()
        self.rpn_metrics.clear()
        self.rpn_cls_loss.clear()
        self.rpn_loc_loss.clear()
        self.rpn_total_loss.clear()

    @staticmethod
    def convert_norm_to_float(net):
        '''
        BatchNorm layers to have parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        '''
        if isinstance(net, torch.nn.modules.batchnorm._BatchNorm):
            net.float()
        for child in net.children():
            VoxelNet.convert_norm_to_float(net)
        return net


class DetectionHead(nn.Module):
    """Just RPN's conv_cls/conv_box/conv_dir_cls + the permute logic (see
    `RPN.__init__`/`forward` above), extracted so `MultiViewVoxelNet` can
    apply a single shared head to a fused feature map without also getting
    a second backbone. Standalone -- `RPN` is untouched.
    """
    def __init__(self,
                 num_input_filters,
                 num_class,
                 num_anchor_per_loc,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 box_code_size=7):
        super().__init__()
        self._use_direction_classifier = use_direction_classifier
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(num_input_filters, num_cls, 1)
        self.conv_box = nn.Conv2d(
            num_input_filters, num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                num_input_filters, num_anchor_per_loc * 2, 1)

    def forward(self, x):
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {"box_preds": box_preds, "cls_preds": cls_preds}
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict


class MultiViewVoxelNet(nn.Module):
    """PointPillars with 1-2 auxiliary (XZ/YZ) pillar branches fused into
    the primary XY branch before a shared detection head. See
    /home/bohyeon/.claude/plans/tidy-purring-sutton.md for the design.

    Each branch keeps its own native 2D resolution through its own
    PillarFeatureNet + PointPillarsScatter + RPNBackbone (no interpolation
    anywhere). Fusion (`ChannelFoldFusion`) folds each auxiliary branch's
    collapsed axis (Z) into channels and broadcasts across whichever XY
    axis that branch lacks -- no pooling, no resize. Detection stays fully
    XY: the shared `target_assigner`/anchors/box_coder are the standard
    7-dim (x,y,z,w,l,h,r) ones VoxelNet already uses; XZ/YZ contribute
    features only, not their own box predictions.

    This is a NEW, independent class -- `VoxelNet` is completely untouched,
    so the existing single-branch path and its checkpoints are unaffected
    by this class existing.
    """
    def __init__(self,
                 branch_configs,
                 num_class=2,
                 num_input_features=4,
                 vfe_num_filters=[64],
                 with_distance=False,
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_threshold=0.5,
                 nms_pre_max_size=1000,
                 nms_post_max_size=20,
                 nms_iou_threshold=0.1,
                 target_assigner=None,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 fused_channels=128,
                 aux_reduce_channels=64,
                 fusion_debug=False,
                 name='multiview_voxelnet'):
        """
        branch_configs: dict keyed by 'xy' (required), 'xz', 'yz'. Each
        value is a dict with keys: output_shape, voxel_size, pc_range,
        spatial_axes, rpn_layer_nums, rpn_layer_strides, rpn_num_filters,
        rpn_upsample_strides, rpn_num_upsample_filters -- i.e. exactly the
        per-branch args second_builder.py already computes for the
        single-branch VoxelNet (view_transform.make_pillar_voxel_config
        etc.), just one dict per branch instead of one set of globals.
        Design each branch's rpn_layer_strides/upsample_strides so its
        native H (for xz/yz) or W (for xz) ends up matching XY's H/W after
        its own backbone -- that's what makes concatenation in
        `ChannelFoldFusion` valid without interpolation (see plan).

        `aux_reduce_channels`: each auxiliary branch's post-backbone output
        (e.g. 384 channels) is 1x1-conv'd down to this many channels BEFORE
        channel-folding -- folding keeps every Z'-bin's channels, so with
        two aux branches at full backbone width the fold+concat tensor is
        enormous (measured: 15744 channels at 248x296 -> OOM on a 24GB GPU
        during backward). This does not re-introduce Z pooling -- Z' stays
        a real, fully preserved dimension; only the per-Z-bin channel width
        shrinks, same as XY's own backbone already summarizing 64 pillar-
        feature channels into fewer effective directions than raw points.
        """
        super().__init__()
        assert 'xy' in branch_configs, "branch_configs must include 'xy'"
        assert not multiclass_nms, "multiclass NMS not yet wired up for MultiViewVoxelNet"
        self.name = name
        self._num_class = num_class
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_score_threshold = nms_score_threshold
        self._nms_pre_max_size = nms_pre_max_size
        self._nms_post_max_size = nms_post_max_size
        self._nms_iou_threshold = nms_iou_threshold
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_direction_classifier = use_direction_classifier
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0
        self._box_coder = target_assigner.box_coder
        self.target_assigner = target_assigner
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._loss_norm_type = loss_norm_type
        self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()
        self._loc_loss_ftor = loc_loss_ftor
        self._cls_loss_ftor = cls_loss_ftor
        self._direction_loss_weight = direction_loss_weight
        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight
        self.register_buffer("global_step", torch.LongTensor(1).zero_())

        self._branch_names = list(branch_configs.keys())
        self.pfns = nn.ModuleDict()
        self.scatters = nn.ModuleDict()
        self.backbones = nn.ModuleDict()
        for view, cfg in branch_configs.items():
            self.pfns[view] = PillarFeatureNet(
                num_input_features, use_norm, num_filters=vfe_num_filters,
                with_distance=with_distance, voxel_size=cfg['voxel_size'],
                pc_range=cfg['pc_range'], spatial_axes=cfg['spatial_axes'])
            self.scatters[view] = PointPillarsScatter(
                output_shape=cfg['output_shape'],
                num_input_features=vfe_num_filters[-1],
                spatial_axes=cfg['spatial_axes'])
            self.backbones[view] = RPNBackbone(
                use_norm=use_norm,
                layer_nums=cfg['rpn_layer_nums'],
                layer_strides=cfg['rpn_layer_strides'],
                num_filters=cfg['rpn_num_filters'],
                upsample_strides=cfg['rpn_upsample_strides'],
                num_upsample_filters=cfg['rpn_num_upsample_filters'],
                num_input_filters=self.scatters[view].nchannels,
                use_groupnorm=use_groupnorm, num_groups=num_groups)

        # Probe each branch's post-backbone native shape with a dummy zero
        # tensor -- needed to size ChannelFoldFusion's reduce-conv, and to
        # sanity-check (Step 2/4 of the plan) that shapes/axes are what we
        # expect before ever touching real data.
        self.branch_shapes = {}
        with torch.no_grad():
            for view, cfg in branch_configs.items():
                out_shape = cfg['output_shape']  # [1, z_dim, y_dim, x_dim, C]
                nx, ny = self.scatters[view].nx, self.scatters[view].ny
                dummy = torch.zeros(1, self.scatters[view].nchannels, ny, nx)
                out = self.backbones[view](dummy)
                self.branch_shapes[view] = tuple(out.shape)
                print(f"[MultiViewVoxelNet] branch={view} native_scatter_shape="
                      f"(1,{self.scatters[view].nchannels},{ny},{nx}) "
                      f"post_backbone_shape={tuple(out.shape)}")

        xy_channels = self.branch_shapes['xy'][1]
        self.aux_reduce = nn.ModuleDict()
        aux_z_sizes = {}
        for view in self._branch_names:
            if view == 'xy':
                continue
            c, h = self.branch_shapes[view][1], self.branch_shapes[view][2]
            self.aux_reduce[view] = nn.Conv2d(c, aux_reduce_channels, 1)
            aux_z_sizes[view] = h  # post-backbone Z length for that branch
            print(f"[MultiViewVoxelNet] branch={view} aux_reduce: {c} -> "
                  f"{aux_reduce_channels} channels (Z'={h} preserved, not pooled)")

        self.fusion = ChannelFoldFusion(
            xy_channels=xy_channels, aux_channels=aux_reduce_channels,
            aux_z_sizes=aux_z_sizes, out_channels=fused_channels, debug=fusion_debug)
        self.head = DetectionHead(
            num_input_filters=fused_channels,
            num_class=num_class,
            num_anchor_per_loc=target_assigner.num_anchors_per_location,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            box_code_size=target_assigner.box_coder.code_size)

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def forward(self, example):
        batch_size_dev = example['anchors'].shape[0]
        branch_feats = {}
        for view in self._branch_names:
            suffix = '' if view == 'xy' else f'_{view}'
            voxels = example['voxels' + suffix]
            num_points = example['num_points' + suffix]
            coors = example['coordinates' + suffix]
            voxel_features = self.pfns[view](voxels, num_points, coors)
            spatial_features = self.scatters[view](voxel_features, coors, batch_size_dev)
            feat = self.backbones[view](spatial_features)
            if view != 'xy':
                feat = self.aux_reduce[view](feat)  # channel width only; Z' untouched
            branch_feats[view] = feat

        fused = self.fusion(
            branch_feats['xy'], branch_feats.get('xz'), branch_feats.get('yz'))
        preds_dict = self.head(fused)
        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]

        if self.training:
            labels = example['labels']
            reg_targets = example['reg_targets']
            cls_weights, reg_weights, cared = prepare_loss_weights(
                labels,
                pos_cls_weight=self._pos_cls_weight,
                neg_cls_weight=self._neg_cls_weight,
                loss_norm_type=self._loss_norm_type,
                dtype=box_preds.dtype)
            cls_targets = labels * cared.type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)

            loc_loss, cls_loss = create_loss(
                self._loc_loss_ftor,
                self._cls_loss_ftor,
                box_preds=box_preds,
                cls_preds=cls_preds,
                cls_targets=cls_targets,
                cls_weights=cls_weights,
                reg_targets=reg_targets,
                reg_weights=reg_weights,
                num_class=self._num_class,
                encode_rad_error_by_sin=self._encode_rad_error_by_sin,
                encode_background_as_zeros=self._encode_background_as_zeros,
                box_code_size=self._box_coder.code_size,
            )
            loc_loss_reduced = loc_loss.sum() / batch_size_dev
            loc_loss_reduced *= self._loc_loss_weight
            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
            cls_pos_loss /= self._pos_cls_weight
            cls_neg_loss /= self._neg_cls_weight
            cls_loss_reduced = cls_loss.sum() / batch_size_dev
            cls_loss_reduced *= self._cls_loss_weight
            loss = loc_loss_reduced + cls_loss_reduced
            dir_loss = torch.zeros_like(loss)
            if self._use_direction_classifier:
                dir_targets = get_direction_target(example['anchors'], reg_targets)
                dir_logits = preds_dict["dir_cls_preds"].view(batch_size_dev, -1, 2)
                weights = (labels > 0).type_as(dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self._dir_loss_ftor(dir_logits, dir_targets, weights=weights)
                dir_loss = dir_loss.sum() / batch_size_dev
                loss += dir_loss * self._direction_loss_weight

            return {
                "loss": loss,
                "cls_loss": cls_loss,
                "loc_loss": loc_loss,
                "cls_pos_loss": cls_pos_loss,
                "cls_neg_loss": cls_neg_loss,
                "cls_preds": cls_preds,
                "dir_loss_reduced": dir_loss,
                "cls_loss_reduced": cls_loss_reduced,
                "loc_loss_reduced": loc_loss_reduced,
                "cared": cared,
            }
        else:
            return self.predict(example, preds_dict)

    def predict(self, example, preds_dict):
        """Verbatim adaptation of `VoxelNet.predict`'s non-multiclass path
        (see that method) -- same decode/NMS logic, just reading from this
        class's own attributes. `multiclass_nms=True` is rejected in
        __init__ since that path isn't ported here yet.
        """
        t = time.time()
        batch_size = example['anchors'].shape[0]
        batch_anchors = example["anchors"].view(batch_size, -1, 7)

        self._total_inference_count += batch_size
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        batch_anchors_mask = [None] * batch_size
        batch_imgidx = example['image_idx']

        self._total_forward_time += time.time() - t
        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1, self._box_coder.code_size)
        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1
        batch_cls_preds = batch_cls_preds.view(batch_size, -1, num_class_with_bg)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds, batch_anchors)
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
                batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask):
            if self._use_direction_classifier:
                dir_labels = torch.max(dir_preds, dim=-1)[1]
            if self._encode_background_as_zeros:
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            else:
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            if self._use_rotate_nms:
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms
            selected_boxes = selected_labels = selected_scores = selected_dir_labels = None

            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = torch.zeros(
                    total_scores.shape[0], device=total_scores.device, dtype=torch.long)
            else:
                top_scores, top_labels = torch.max(total_scores, dim=-1)

            if self._nms_score_threshold > 0.0:
                thresh = torch.tensor(
                    [self._nms_score_threshold], device=total_scores.device).type_as(total_scores)
                top_scores_keep = (top_scores >= thresh)
                top_scores = top_scores.masked_select(top_scores_keep)
            if top_scores.shape[0] != 0:
                if self._nms_score_threshold > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    if self._use_direction_classifier:
                        dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(box_preds_corners)
                selected = nms_func(
                    boxes_for_nms, top_scores,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold)
            else:
                selected = None
            if selected is not None:
                selected_boxes = box_preds[selected]
                if self._use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]

            if selected_boxes is not None:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                    box_preds[..., -1] += torch.where(
                        opp_labels, torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c)
                locs = final_box_preds_camera[:, :3]
                dims = final_box_preds_camera[:, 3:6]
                angles = final_box_preds_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_torch_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = box_torch_ops.project_to_image(box_corners, P2)
                minxy = torch.min(box_corners_in_image, dim=1)[0]
                maxxy = torch.max(box_corners_in_image, dim=1)[0]
                box_2d_preds = torch.cat([minxy, maxxy], dim=1)
                predictions_dict = {
                    "bbox": box_2d_preds,
                    "box3d_camera": final_box_preds_camera,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "image_idx": img_idx,
                }
            else:
                predictions_dict = {
                    "bbox": None, "box3d_camera": None, "box3d_lidar": None,
                    "scores": None, "label_preds": None, "image_idx": img_idx,
                }
            predictions_dicts.append(predictions_dict)
        self._total_postprocess_time += time.time() - t
        return predictions_dicts


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
        boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                box_code_size=7):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = torchplus.nn.one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    return loc_losses, cls_losses


def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         loss_norm_type=LossNormType.NormByNumPositives,
                         dtype=torch.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = cared.type(dtype).sum(1, keepdim=True)
        num_examples = torch.clamp(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
        normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
        # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
        normalizer = torch.clamp(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    else:
        raise ValueError(
            f"unknown loss norm type. available: {list(LossNormType)}")
    return cls_weights, reg_weights, cared


def assign_weight_to_each_class(labels,
                                weight_per_class,
                                norm_by_num=True,
                                dtype=torch.float32):
    weights = torch.zeros(labels.shape, dtype=dtype, device=labels.device)
    for label, weight in weight_per_class:
        positives = (labels == label).type(dtype)
        weight_class = weight * positives
        if norm_by_num:
            normalizer = positives.sum()
            normalizer = torch.clamp(normalizer, min=1.0)
            weight_class /= normalizer
        weights += weight_class
    return weights


def get_direction_target(anchors, reg_targets, one_hot=True):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, 7)
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = (rot_gt > 0).long()
    if one_hot:
        dir_cls_targets = torchplus.nn.one_hot(
            dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets


# =============================================================================
# Point-correspondence gated-residual YZ fusion (replaces channel-fold
# broadcast for the case of exactly one auxiliary branch, YZ). See
# GatedYZVoxelNet below for how this is wired up; ChannelFoldFusion above is
# left completely untouched so old xy+xz/xy+yz/xy+xz+yz runs stay reproducible
# for comparison.
# =============================================================================

def masked_scatter_mean_yz_to_xy(F_yz, voxels_yz, num_points_yz, coords_yz,
                                  batch_size, xy_pc_range, xy_voxel_size,
                                  out_size_factor, out_h, out_w):
    """Map each real LiDAR point's own (y, z) -> its YZ backbone feature,
    then that SAME point's own (x, y) -> the XY output cell it lands in
    (at F_xy's resolution, i.e. after `out_size_factor` downsampling), and
    masked-mean all points landing in the same XY cell.

    This does not attempt to recover distinct per-X-position information
    that was already summarized when the YZ pillar was formed (all points
    in one YZ pillar share one feature vector, regardless of their spread
    along X) -- it only routes that shared vector to each point's own real
    XY position instead of broadcasting it identically to every X column,
    which is the concrete difference from the old ChannelFoldFusion path.

    Args:
      F_yz: [B, C, Z', Y'] YZ branch backbone output.
      voxels_yz: [N_yz_voxels, max_pts, >=2] raw per-point features for
        each YZ pillar (column 0 = real-world x, column 1 = real-world y --
        the same `voxels_*` tensor the YZ PillarFeatureNet already
        consumes, so no new data-pipeline fields are needed).
      num_points_yz: [N_yz_voxels] valid-point count per YZ pillar (the
        rest of `max_pts` is zero-padding) -- same tensor PillarFeatureNet
        uses for its own padding mask.
      coords_yz: [N_yz_voxels, 4] = [batch, z_idx, y_idx, x_idx] (x_idx is
        always 0, the collapsed axis for this branch) in this codebase's
        fixed coordinate convention (see PillarFeatureNet/PointPillarsScatter
        docstrings) -- used only to look up *which* YZ pillar's feature
        (F_yz[batch, :, z_idx, y_idx]) each of its own points should read.
      xy_pc_range, xy_voxel_size: the canonical XY point_cloud_range (6,)
        and voxel_size (3,) -- real-world units, not resolution-specific.
      out_size_factor: XY backbone's native-grid -> output-grid downsample
        factor (1 for ped_cycle, 2 for car in this repo's configs) -- so
        the effective output cell size is `xy_voxel_size * out_size_factor`.
      out_h, out_w: F_xy's spatial size (rows=Y, cols=X). The returned map
        matches this exactly, by construction, for any task/config -- no
        hardcoded resolution.

    Returns:
      agg: [B, C, out_h, out_w], masked mean of F_yz features scattered by
        real point position; exactly 0 in cells no point maps into.
        Differentiable w.r.t. F_yz (indexing + index_add_ both support
        autograd), so gradients reach the YZ branch through this path.
    """
    device = F_yz.device
    dtype = F_yz.dtype
    C = F_yz.shape[1]
    n_voxels, max_pts = voxels_yz.shape[0], voxels_yz.shape[1]

    if n_voxels == 0:
        return torch.zeros(batch_size, C, out_h, out_w, device=device, dtype=dtype)

    b_idx = coords_yz[:, 0].long()
    # coords_yz indexes the YZ branch's NATIVE (pre-backbone) pillar grid,
    # but F_yz is that branch's POST-backbone output -- downsampled by the
    # same out_size_factor as the xy branch, since aux branches are built
    # with the same rpn_layer_strides (see multiview_experiment.py's
    # _aux_branch_config). Native indices must be scaled down to match, or
    # this indexes out of bounds whenever out_size_factor > 1 (e.g. car).
    z_native = coords_yz[:, 1].long()
    y_native = coords_yz[:, 2].long()
    z_idx = (z_native // out_size_factor).clamp(max=F_yz.shape[2] - 1)
    y_idx = (y_native // out_size_factor).clamp(max=F_yz.shape[3] - 1)
    # One feature vector per YZ pillar -> [n_voxels, C].
    voxel_feat = F_yz[b_idx, :, z_idx, y_idx]
    # Every point in that pillar starts from the same vector.
    point_feat = voxel_feat.unsqueeze(1).expand(n_voxels, max_pts, C)

    valid = get_paddings_indicator(num_points_yz, max_pts, axis=0)  # [n_voxels, max_pts] bool

    px = voxels_yz[..., 0]
    py = voxels_yz[..., 1]
    eff_vx = xy_voxel_size[0] * out_size_factor
    eff_vy = xy_voxel_size[1] * out_size_factor
    x_out = torch.floor((px - xy_pc_range[0]) / eff_vx).long()
    y_out = torch.floor((py - xy_pc_range[1]) / eff_vy).long()

    in_range = valid & (x_out >= 0) & (x_out < out_w) & (y_out >= 0) & (y_out < out_h)

    b_flat = b_idx.unsqueeze(1).expand(n_voxels, max_pts)
    flat_idx = (b_flat * out_h + y_out) * out_w + x_out  # [n_voxels, max_pts]

    flat_idx = flat_idx[in_range]        # [N_valid]
    feats_valid = point_feat[in_range]   # [N_valid, C]

    total_cells = batch_size * out_h * out_w
    sum_buf = torch.zeros(total_cells, C, device=device, dtype=dtype)
    cnt_buf = torch.zeros(total_cells, device=device, dtype=dtype)
    if flat_idx.numel() > 0:
        sum_buf.index_add_(0, flat_idx, feats_valid)
        cnt_buf.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=dtype))

    mean_buf = sum_buf / cnt_buf.clamp(min=1.0).unsqueeze(1)
    agg = mean_buf.view(batch_size, out_h, out_w, C).permute(0, 3, 1, 2).contiguous()
    return agg


class PointCorrespondenceGatedFusion(nn.Module):
    """`F_out = F_xy + G * proj(agg_yz)`, `G = sigmoid(Conv1x1(cat(F_xy, proj(agg_yz))))`.

    `proj` and `gate_conv` are both zero-initialized (weight AND bias), so
    at construction `proj(agg_yz) == 0` identically regardless of `agg_yz`,
    hence `F_out == F_xy` exactly -- the model starts identical to running
    the XY path alone. `force_gate_one` is a pure eval-time/diagnostic
    override (see GatedYZVoxelNet.forward): with it set, G is a constant 1
    with no gradient to gate_conv, used for Config B (learned correspondence
    fusion, un-gated) and for the "force a trained C's gate to 1" ablation
    (never both trained together as if they were the same run).
    """
    def __init__(self, xy_channels, yz_channels):
        super().__init__()
        self.proj = nn.Conv2d(yz_channels, xy_channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gate_conv = nn.Conv2d(xy_channels * 2, 1, 1)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)
        self.force_gate_one = False

    def forward(self, F_xy, agg_yz):
        delta = self.proj(agg_yz)  # ΔF_yz, linear output (can be +/-)
        if self.force_gate_one:
            gate = torch.ones(F_xy.shape[0], 1, F_xy.shape[2], F_xy.shape[3],
                               device=F_xy.device, dtype=F_xy.dtype)
        else:
            gate_logit = self.gate_conv(torch.cat([F_xy, delta], dim=1))
            gate = torch.sigmoid(gate_logit)
        f_out = F_xy + gate * delta
        return f_out, gate, delta


class GatedYZVoxelNet(nn.Module):
    """PointPillars with the original single-branch XY path/head UNCHANGED
    (weight-compatible with a plain `VoxelNet` checkpoint -- see
    `remap_baseline_state_dict_to_gated` below) plus a YZ branch whose
    native-resolution backbone features are routed to their real XY
    position via `masked_scatter_mean_yz_to_xy` and merged in with a
    zero-init learned gated residual (`PointCorrespondenceGatedFusion`).

    Unlike `MultiViewVoxelNet`, `self.head` sees the XY branch's full
    384-channel backbone output directly -- there is no shared reduce-conv
    in the XY path, so a baseline checkpoint's `rpn.*` weights map onto
    this model's `backbones['xy']`/`head` with identical shapes.
    """
    def __init__(self,
                 xy_branch_cfg,
                 yz_branch_cfg,
                 num_class=2,
                 num_input_features=4,
                 vfe_num_filters=[64],
                 with_distance=False,
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_threshold=0.5,
                 nms_pre_max_size=1000,
                 nms_post_max_size=20,
                 nms_iou_threshold=0.1,
                 target_assigner=None,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 out_size_factor=1,
                 correction_enabled=True,
                 name='gated_yz_voxelnet'):
        super().__init__()
        assert not multiclass_nms, "multiclass NMS not wired up for GatedYZVoxelNet"
        self.name = name
        self._num_class = num_class
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_score_threshold = nms_score_threshold
        self._nms_pre_max_size = nms_pre_max_size
        self._nms_post_max_size = nms_post_max_size
        self._nms_iou_threshold = nms_iou_threshold
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_direction_classifier = use_direction_classifier
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0
        self._box_coder = target_assigner.box_coder
        self.target_assigner = target_assigner
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._loss_norm_type = loss_norm_type
        self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()
        self._loc_loss_ftor = loc_loss_ftor
        self._cls_loss_ftor = cls_loss_ftor
        self._direction_loss_weight = direction_loss_weight
        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight
        self.register_buffer("global_step", torch.LongTensor(1).zero_())

        # correspondence bookkeeping -- real-world units, resolution-independent
        self._xy_pc_range = list(xy_branch_cfg['pc_range'])
        self._xy_voxel_size = list(xy_branch_cfg['voxel_size'])
        self._out_size_factor = out_size_factor
        self.correction_enabled = correction_enabled

        self.pfns = nn.ModuleDict()
        self.scatters = nn.ModuleDict()
        self.backbones = nn.ModuleDict()
        for view, cfg in (('xy', xy_branch_cfg), ('yz', yz_branch_cfg)):
            self.pfns[view] = PillarFeatureNet(
                num_input_features, use_norm, num_filters=vfe_num_filters,
                with_distance=with_distance, voxel_size=cfg['voxel_size'],
                pc_range=cfg['pc_range'], spatial_axes=cfg['spatial_axes'])
            self.scatters[view] = PointPillarsScatter(
                output_shape=cfg['output_shape'],
                num_input_features=vfe_num_filters[-1],
                spatial_axes=cfg['spatial_axes'])
            self.backbones[view] = RPNBackbone(
                use_norm=use_norm,
                layer_nums=cfg['rpn_layer_nums'],
                layer_strides=cfg['rpn_layer_strides'],
                num_filters=cfg['rpn_num_filters'],
                upsample_strides=cfg['rpn_upsample_strides'],
                num_upsample_filters=cfg['rpn_num_upsample_filters'],
                num_input_filters=self.scatters[view].nchannels,
                use_groupnorm=use_groupnorm, num_groups=num_groups)

        # probe real output shapes -- nothing hardcoded
        self.branch_shapes = {}
        with torch.no_grad():
            for view, cfg in (('xy', xy_branch_cfg), ('yz', yz_branch_cfg)):
                nx, ny = self.scatters[view].nx, self.scatters[view].ny
                dummy = torch.zeros(1, self.scatters[view].nchannels, ny, nx)
                out = self.backbones[view](dummy)
                self.branch_shapes[view] = tuple(out.shape)
                print(f"[GatedYZVoxelNet] branch={view} native_scatter_shape="
                      f"(1,{self.scatters[view].nchannels},{ny},{nx}) "
                      f"post_backbone_shape={tuple(out.shape)}")

        xy_channels = self.branch_shapes['xy'][1]
        yz_channels = self.branch_shapes['yz'][1]
        self.fusion = PointCorrespondenceGatedFusion(xy_channels, yz_channels)

        # head sees the FULL xy backbone width -- no reduce-conv, so a
        # baseline VoxelNet checkpoint's rpn.conv_* weights load unchanged.
        self.head = DetectionHead(
            num_input_filters=xy_channels,
            num_class=num_class,
            num_anchor_per_loc=target_assigner.num_anchors_per_location,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            box_code_size=target_assigner.box_coder.code_size)

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def set_frozen_baseline_eval(self):
        """Stage-1 training helper: put the XY PFN/backbone and head into
        eval() (frozen BatchNorm running stats) while the rest of the model
        stays in train() mode. Call this AFTER `net.train()` on every step
        (or once per epoch is not enough -- `net.train()` flips every
        submodule back to train mode, so this must run after it), since
        nn.Module has no notion of "train() except these children".
        """
        self.pfns['xy'].eval()
        self.scatters['xy'].eval()
        self.backbones['xy'].eval()
        self.head.eval()

    def forward(self, example):
        batch_size_dev = example['anchors'].shape[0]

        xy_voxels = example['voxels']
        xy_num_points = example['num_points']
        xy_coors = example['coordinates']
        xy_feat = self.pfns['xy'](xy_voxels, xy_num_points, xy_coors)
        xy_spatial = self.scatters['xy'](xy_feat, xy_coors, batch_size_dev)
        F_xy = self.backbones['xy'](xy_spatial)

        if self.correction_enabled:
            yz_voxels = example['voxels_yz']
            yz_num_points = example['num_points_yz']
            yz_coors = example['coordinates_yz']
            yz_feat = self.pfns['yz'](yz_voxels, yz_num_points, yz_coors)
            yz_spatial = self.scatters['yz'](yz_feat, yz_coors, batch_size_dev)
            F_yz = self.backbones['yz'](yz_spatial)

            out_h, out_w = F_xy.shape[2], F_xy.shape[3]
            agg_yz = masked_scatter_mean_yz_to_xy(
                F_yz, yz_voxels, yz_num_points, yz_coors, batch_size_dev,
                self._xy_pc_range, self._xy_voxel_size, self._out_size_factor,
                out_h, out_w)
            F_out, gate, delta = self.fusion(F_xy, agg_yz)
            self._last_gate = gate.detach()
            self._last_delta = delta.detach()
        else:
            # Config-A-equivalent path for numerical-parity verification:
            # YZ branch is never run, F_out is exactly F_xy.
            F_out = F_xy
            self._last_gate = None
            self._last_delta = None

        preds_dict = self.head(F_out)
        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]

        if self.training:
            labels = example['labels']
            reg_targets = example['reg_targets']
            cls_weights, reg_weights, cared = prepare_loss_weights(
                labels,
                pos_cls_weight=self._pos_cls_weight,
                neg_cls_weight=self._neg_cls_weight,
                loss_norm_type=self._loss_norm_type,
                dtype=box_preds.dtype)
            cls_targets = labels * cared.type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)

            loc_loss, cls_loss = create_loss(
                self._loc_loss_ftor,
                self._cls_loss_ftor,
                box_preds=box_preds,
                cls_preds=cls_preds,
                cls_targets=cls_targets,
                cls_weights=cls_weights,
                reg_targets=reg_targets,
                reg_weights=reg_weights,
                num_class=self._num_class,
                encode_rad_error_by_sin=self._encode_rad_error_by_sin,
                encode_background_as_zeros=self._encode_background_as_zeros,
                box_code_size=self._box_coder.code_size,
            )
            loc_loss_reduced = loc_loss.sum() / batch_size_dev
            loc_loss_reduced *= self._loc_loss_weight
            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
            cls_pos_loss /= self._pos_cls_weight
            cls_neg_loss /= self._neg_cls_weight
            cls_loss_reduced = cls_loss.sum() / batch_size_dev
            cls_loss_reduced *= self._cls_loss_weight
            loss = loc_loss_reduced + cls_loss_reduced
            dir_loss = torch.zeros_like(loss)
            if self._use_direction_classifier:
                dir_targets = get_direction_target(example['anchors'], reg_targets)
                dir_logits = preds_dict["dir_cls_preds"].view(batch_size_dev, -1, 2)
                weights = (labels > 0).type_as(dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self._dir_loss_ftor(dir_logits, dir_targets, weights=weights)
                dir_loss = dir_loss.sum() / batch_size_dev
                loss += dir_loss * self._direction_loss_weight

            return {
                "loss": loss,
                "cls_loss": cls_loss,
                "loc_loss": loc_loss,
                "cls_pos_loss": cls_pos_loss,
                "cls_neg_loss": cls_neg_loss,
                "cls_preds": cls_preds,
                "dir_loss_reduced": dir_loss,
                "cls_loss_reduced": cls_loss_reduced,
                "loc_loss_reduced": loc_loss_reduced,
                "cared": cared,
            }
        else:
            return MultiViewVoxelNet.predict(self, example, preds_dict)


def remap_baseline_state_dict_to_gated(baseline_state_dict):
    """Rename a plain-`VoxelNet` checkpoint's keys onto `GatedYZVoxelNet`'s
    xy-path submodules. Returns (remapped_state_dict, skipped_keys,
    unrecognized_keys) -- call site is responsible for checking those are
    what it expects before loading (see gated_yz_experiment.py).
    """
    metric_prefixes = ('rpn_acc.', 'rpn_precision.', 'rpn_recall.', 'rpn_metrics.',
                        'rpn_cls_loss.', 'rpn_loc_loss.', 'rpn_total_loss.')
    remapped, skipped, unrecognized = {}, [], []
    for k, v in baseline_state_dict.items():
        if k.startswith('voxel_feature_extractor.'):
            remapped[k.replace('voxel_feature_extractor.', 'pfns.xy.', 1)] = v
        elif k.startswith(('rpn.conv_cls.', 'rpn.conv_box.', 'rpn.conv_dir_cls.')):
            remapped[k.replace('rpn.', 'head.', 1)] = v
        elif k.startswith('rpn.'):
            remapped[k.replace('rpn.', 'backbones.xy.', 1)] = v
        elif k.startswith('middle_feature_extractor.') or k == 'global_step' \
                or k.startswith(metric_prefixes):
            skipped.append(k)
        else:
            unrecognized.append(k)
    return remapped, skipped, unrecognized
