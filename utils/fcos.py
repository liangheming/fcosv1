import torch
from torchvision.ops.boxes import nms


def non_max_suppression(prediction: torch.Tensor,
                             conf_thresh=0.05,
                             iou_thresh=0.5,
                             max_det=300,
                             max_box=2048,
                             max_layer_num=1000
                             ):
    """
    :param max_layer_num:
    :param prediction:
    :param conf_thresh:
    :param iou_thresh:
    :param max_det:
    :param max_box:
    :return: (x1,y1,x2,y2,score,cls_id)
    """
    for i in range(len(prediction)):
        if prediction[i].dtype == torch.float16:
            prediction[i] = prediction[i].float()
    bs = prediction[0].shape[0]
    out = [None] * bs
    for bi in range(bs):
        batch_predicts_list = [torch.zeros(size=(0, 6), device=prediction[0].device).float()] * len(prediction)
        for lj in range(len(prediction)):
            one_layer_bath_predict = prediction[lj][bi]
            reg_predicts = one_layer_bath_predict[:, :4]
            center_predicts = one_layer_bath_predict[:, 4].sigmoid()
            cls_predicts = one_layer_bath_predict[:, 5:].sigmoid()

            max_val, max_idx = cls_predicts.max(dim=1)
            valid_bool_idx = max_val > conf_thresh
            if valid_bool_idx.sum() == 0:
                continue
            valid_val = max_val[valid_bool_idx] * center_predicts[valid_bool_idx]
            sorted_idx = valid_val.argsort(descending=True)
            valid_val = valid_val[sorted_idx]
            valid_box = reg_predicts[valid_bool_idx, :][sorted_idx]
            valid_cls = max_idx[valid_bool_idx][sorted_idx]
            if 0 < max_layer_num < valid_box.shape[0]:
                valid_val = valid_val[:max_layer_num]
                valid_box = valid_box[:max_layer_num, :]
                valid_cls = valid_cls[:max_layer_num]
            batch_predicts = torch.cat([valid_box, valid_val[:, None], valid_cls[:, None]], dim=-1)
            batch_predicts_list[lj] = batch_predicts
        x = torch.cat(batch_predicts_list, dim=0)
        if x.shape[0] == 0:
            continue
        c = x[:, 5:6] * max_box
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms(boxes, scores, iou_thresh)
        if i.shape[0] > max_det:
            i = i[:max_det]
        out[bi] = x[i]
    return out


class BoxCoder(object):
    def __init__(self):
        super(BoxCoder, self).__init__()

    @staticmethod
    def encoder(grids, gt_boxes):
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
    def decoder(predicts, grids):
        predicts[..., :2] = grids - predicts[..., :2]
        predicts[..., 2:] = grids + predicts[..., 2:]
        return predicts
