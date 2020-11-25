import torch
import numpy as np
from losses.commons import focal_loss, IOULoss, BoxSimilarity

INF = 1e8


class BoxCoder(object):
    def __init__(self):
        super(BoxCoder, self).__init__()

    @staticmethod
    def full_encoder(grids, gt_boxes):
        """
        :param grids: [num_grids,2] (xc,yc)
        :param gt_boxes: [num_gts,4] (x1,y1,x2,y2)
        :return: [num_grids,num_gts,4] (l,t,r,b)
        """
        left = grids[:, None, 0] - gt_boxes[None, :, 0]
        top = grids[:, None, 1] - gt_boxes[None, :, 1]
        right = gt_boxes[None, :, 2] - grids[:, None, 0]
        bottom = gt_boxes[None, :, 3] - grids[:, None, 1]
        reg_target_per_img = torch.stack([left, top, right, bottom], dim=2)
        return reg_target_per_img

    @staticmethod
    def encoder(grids, gt_boxes):
        """

        :param grids:[num,2]
        :param gt_boxes:[num,4](x1,y1,x2,y2)
        :return:
        """
        left_top = grids[..., [0, 1]] - gt_boxes[..., [0, 1]]
        right_bottom = gt_boxes[..., [2, 3]] - grids[..., [0, 1]]
        return torch.cat([left_top, right_bottom], dim=-1)

    @staticmethod
    def decoder(predicts, grids):
        predicts[..., :2] = grids - predicts[..., :2]
        predicts[..., 2:] = grids + predicts[..., 2:]
        return predicts


class Matcher(object):
    BELOW_LOW_THRESHOLD = -1

    def __init__(self, radius, strides, layer_limits):
        self.radius = radius
        self.box_coder = BoxCoder()
        self.strides = torch.tensor(strides)
        expand_limits = np.array(layer_limits)[None].repeat(2).tolist()
        self.layer_limits = torch.tensor([-1.] + expand_limits + [INF]).view(-1, 2)

    def __call__(self, grids, gt_boxes):
        ret = list()
        device = grids[0].device
        if self.strides.device != device:
            self.strides = self.strides.to(device)
        if self.layer_limits.device != device:
            self.layer_limits = self.layer_limits.to(device)

        expand_grid = torch.cat(
            [torch.cat([grid, layer_limit.expand_as(grid), stride.expand_as(grid[..., [0]])],
                       dim=-1).view(-1, 5)
             for grid, layer_limit, stride in zip(grids, self.layer_limits, self.strides)
             ], dim=0)
        for bid, gt in enumerate(gt_boxes):
            if len(gt) == 0:
                continue
            reg_target_per_img = self.box_coder.full_encoder(expand_grid, gt[:, 1:])
            if self.radius == 0:
                valid_in_box = reg_target_per_img.min(dim=2)[0] > 0
            else:
                limit_gt_xy = gt[:, [1, 2]] + gt[:, [3, 4]] / 2
                limit_gt_min_xy = limit_gt_xy[None, :, :] - expand_grid[:, None, [4, 4]] * self.radius
                limit_gt_max_xy = limit_gt_xy[None, :, :] + expand_grid[:, None, [4, 4]] * self.radius
                limit_gt_min_xy = torch.where(limit_gt_min_xy > gt[None, :, [1, 2]],
                                              limit_gt_min_xy, gt[None, :, [1, 2]])
                limit_gt_max_xy = torch.where(limit_gt_max_xy < gt[None, :, [3, 4]],
                                              limit_gt_max_xy, gt[None, :, [3, 4]])
                left_top = expand_grid[:, None, [0, 1]] - limit_gt_min_xy
                right_bottom = limit_gt_max_xy - expand_grid[:, None, [0, 1]]
                valid_in_box = torch.cat([left_top, right_bottom], dim=2).min(dim=2)[0] > 0
            max_reg_targets_per_im = reg_target_per_img.max(dim=2)[0]
            is_card_in_level = (max_reg_targets_per_im >= expand_grid[:, [2]]) & (
                    max_reg_targets_per_im <= expand_grid[:, [3]])
            gt_area = (gt[:, 3] - gt[:, 1]) * (gt[:, 4] - gt[:, 2])
            locations_to_gt_area = gt_area[None, :].repeat(len(expand_grid), 1)
            locations_to_gt_area[~valid_in_box] = INF
            locations_to_gt_area[~is_card_in_level] = INF
            min_area, gt_idx = locations_to_gt_area.min(dim=1)
            gt_idx[min_area == INF] = self.BELOW_LOW_THRESHOLD
            ret.append((bid, gt_idx))
        return ret


class FCOSLoss(object):
    def __init__(self,
                 strides,
                 layer_limits,
                 radius=0,
                 alpha=0.25,
                 gamma=2.0,
                 iou_type="giou",
                 iou_loss_type="centerness",
                 iou_loss_weight=0.5,
                 reg_loss_weight=1.3):
        self.alpha = alpha
        self.gamma = gamma
        self.matcher = Matcher(radius=radius, strides=strides, layer_limits=layer_limits)
        self.iou_loss = IOULoss(iou_type=iou_type, coord_type="ltrb")
        self.box_similarity = BoxSimilarity(iou_type="iou", coord_type="ltrb")
        self.iou_loss_type = iou_loss_type
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.iou_loss_weight = iou_loss_weight
        self.reg_loss_weight = reg_loss_weight

    @staticmethod
    def build_centerness(reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, cls_predicts, reg_predicts, iou_predicts, grids, targets):
        cls_predicts = torch.cat([item for item in cls_predicts], dim=1)
        reg_predicts = torch.cat([item for item in reg_predicts], dim=1)
        iou_predicts = torch.cat([item for item in iou_predicts], dim=1)
        all_grids = torch.cat([item for item in grids], dim=0)
        gt_boxes = targets['target'].split(targets['batch_len'])
        matches = self.matcher(grids, gt_boxes)
        match_bidx = list()
        match_grid_idx = list()
        match_gt_idx = list()
        for bid, match in matches:
            grid_idx = (match >= 0).nonzero(as_tuple=False).squeeze(-1)
            match_grid_idx.append(grid_idx)
            match_gt_idx.append(match[grid_idx])
            match_bidx.append(bid)
        if cls_predicts.dtype == torch.float16:
            cls_predicts = cls_predicts.float()
        if iou_predicts.dtype == torch.float16:
            iou_predicts = iou_predicts.float()
        cls_batch_idx = sum([[i] * len(j) for i, j in zip(match_bidx, match_grid_idx)], [])
        cls_grid_idx = torch.cat(match_grid_idx)
        cls_label_idx = torch.cat([gt_boxes[i][:, 0][j].long() for i, j in zip(match_bidx, match_gt_idx)])
        num_pos = len(cls_batch_idx)
        cls_targets = torch.zeros_like(cls_predicts)
        cls_targets[cls_batch_idx, cls_grid_idx, cls_label_idx] = 1.0
        all_cls_loss = focal_loss(cls_predicts.sigmoid(), cls_targets, alpha=self.alpha,
                                  gamma=self.gamma).sum() / num_pos

        all_box_targets = self.matcher.box_coder.encoder(all_grids[cls_grid_idx],
                                                         torch.cat(
                                                             [gt_boxes[i][:, 1:][j] for i, j in
                                                              zip(match_bidx, match_gt_idx)]
                                                             , dim=0))
        all_box_predicts = reg_predicts[cls_batch_idx, cls_grid_idx]
        if self.iou_loss_type == "centerness":
            iou_targets = self.build_centerness(all_box_targets)
        elif self.iou_loss_type == "iou":
            iou_targets = self.box_similarity(all_box_predicts.detach(), all_box_targets)
        else:
            raise NotImplementedError("iou_loss_type: {:s} is not support now".format(self.iou_loss_type))
        all_iou_loss = self.iou_loss_weight * self.bce(
            iou_predicts[cls_batch_idx, cls_grid_idx, 0], iou_targets) / num_pos
        all_box_loss = self.reg_loss_weight * (
                self.iou_loss(all_box_predicts, all_box_targets) * iou_targets).sum() / (iou_targets.sum())
        return all_cls_loss, all_box_loss, all_iou_loss, num_pos
