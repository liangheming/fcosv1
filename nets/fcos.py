import torch
import math
from nets import resnet
from torch import nn
from nets.common import FPN, CGR, CBR
from torchvision.ops.boxes import nms
from losses.fcos_loss import BoxCoder, FCOSLoss


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
            cls_predicts = one_layer_bath_predict[:, 4:].sigmoid()
            center_predicts = cls_predicts[:, 0]
            cls_predicts = cls_predicts[:, 1:]

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


class Scale(nn.Module):
    def __init__(self, init_val=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(data=init_val), requires_grad=True)

    def forward(self, x):
        return x * self.scale


class SequenceCNR(nn.Module):
    def __init__(self,
                 in_channel,
                 inner_channel,
                 kennel_size=3,
                 stride=1,
                 num=4,
                 padding=None,
                 bias=True,
                 block_type='CGR'):
        super(SequenceCNR, self).__init__()
        self.bones = list()
        for i in range(num):
            if i == 0:
                block = eval(block_type)(in_channel, inner_channel, kennel_size, stride, padding=padding, bias=bias)
            else:
                block = eval(block_type)(inner_channel, inner_channel, kennel_size, stride, padding=padding, bias=bias)
            self.bones.append(block)
        self.bones = nn.Sequential(*self.bones)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.bones(x)


class RetinaClsHead(nn.Module):
    def __init__(self,
                 in_channel=256,
                 num_anchors=9, num_cls=80):
        super(RetinaClsHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_cls = num_cls
        self.cls = nn.Conv2d(in_channel, self.num_anchors * self.num_cls, 3, 1, 1)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.constant_(self.cls.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        x = self.cls(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() \
            .view(bs, h, w, self.num_anchors, self.num_cls) \
            .view(bs, -1, self.num_cls)
        return x


class RetinaRegHead(nn.Module):
    def __init__(self, in_channel=256, num_anchors=9):
        super(RetinaRegHead, self).__init__()
        self.num_anchors = num_anchors
        self.reg = nn.Conv2d(in_channel, self.num_anchors * 4, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.reg(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() \
            .view(bs, h, w, self.num_anchors, 4) \
            .view(x.size(0), -1, 4)
        return x


class IOUPredHead(nn.Module):
    def __init__(self, in_channel=256, num_anchors=9):
        super(IOUPredHead, self).__init__()
        self.num_anchors = num_anchors
        self.iou_head = nn.Conv2d(in_channel, self.num_anchors, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.iou_head(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() \
            .view(bs, h, w, self.num_anchors, 1) \
            .view(x.size(0), -1, 1)
        return x


class FCOSHead(nn.Module):
    def __init__(self, in_channel,
                 inner_channel,
                 strides,
                 num_cls=80,
                 num_convs=4,
                 layer_num=5,
                 block_type="CGR",
                 on_reg=True):
        super(FCOSHead, self).__init__()
        self.num_cls = num_cls
        self.layer_num = layer_num
        self.strides = strides
        self.on_reg = on_reg
        self.box_coder = BoxCoder()

        self.scales = nn.ModuleList([Scale(init_val=1.0) for _ in range(self.layer_num)])
        self.grids = [torch.zeros(size=(0, 2))] * self.layer_num
        self.cls_bones = SequenceCNR(in_channel, inner_channel,
                                     kennel_size=3, stride=1,
                                     num=num_convs, block_type=block_type)
        self.reg_bones = SequenceCNR(in_channel, inner_channel,
                                     kennel_size=3, stride=1,
                                     num=num_convs, block_type=block_type)
        self.cls_head = RetinaClsHead(inner_channel, 1, num_cls)
        self.reg_head = RetinaRegHead(inner_channel, 1)
        self.iou_head = IOUPredHead(inner_channel, 1)

    def build_grids(self, feature_maps):
        assert len(self.strides) == len(feature_maps)
        assert self.layer_num == len(feature_maps)

        grids = list()
        for i in range(self.layer_num):
            feature_map = feature_maps[i]
            stride = self.strides[i]
            _, _, ny, nx = feature_map.shape
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid = torch.stack([xv,
                                yv], dim=2)
            grid = (grid + 0.5) * stride
            grids.append(grid.view(-1, 2))
        return grids

    def forward(self, xs):
        cls_outputs = list()
        reg_outputs = list()
        center_outputs = list()
        for i, x in enumerate(xs):
            cls_tower = self.cls_bones(x)
            reg_tower = self.reg_bones(x)
            cls_outputs.append(self.cls_head(cls_tower))
            reg_outputs.append((self.scales[i](self.reg_head(reg_tower))).exp())
            if self.on_reg:
                center_outputs.append(self.iou_head(reg_tower))
            else:
                center_outputs.append(self.iou_head(cls_tower))
        if self.grids[0] is None or self.grids[0].shape[0] != cls_outputs[0].shape[1]:
            with torch.no_grad():
                grids = self.build_grids(xs)
                assert len(grids) == len(self.grids)
                for i, grid in enumerate(grids):
                    self.grids[i] = grid.to(xs[0].device)
        if self.training:
            return cls_outputs, reg_outputs, center_outputs, self.grids
        else:
            output = list()
            for cls_predict, reg_predict, center_predict, grid in zip(cls_outputs, reg_outputs, center_outputs,
                                                                      self.grids):
                grid_output = grid[None]
                reg_output = self.box_coder.decoder(reg_predict, grid_output)
                cat_output = torch.cat([reg_output, center_predict, cls_predict], dim=-1)
                output.append(cat_output)
            return output


default_cfg = {
    "num_cls": 80,
    "strides": [8, 16, 32, 64, 128],
    "backbone": "resnet18",
    "pretrained": True,
    "fpn_channel": 256,
    "head_conv_num": 4,
    "block_type": "CGR",
    "on_reg": True,
    # loss
    "radius": 7,
    "alpha": 0.25,
    "gamma": 2.0,
    "iou_type": "giou",
    "layer_limits": [64, 128, 256, 512],
    "iou_loss_type": "centerness",
    "iou_loss_weight": 0.5,
    "reg_loss_weight": 1.3,
    # predicts
    "conf_thresh": 0.01,
    "nms_iou_thresh": 0.5,
    "max_det": 300,

}


class FCOS(nn.Module):
    def __init__(self, **kwargs):
        self.cfg = {**default_cfg, **kwargs}
        super(FCOS, self).__init__()
        self.backbones = getattr(resnet, self.cfg['backbone'])(pretrained=self.cfg['pretrained'])
        c3, c4, c5 = self.backbones.inner_channels
        self.neck = FPN(c3, c4, c5, self.cfg['fpn_channel'])
        self.head = FCOSHead(in_channel=self.cfg['fpn_channel'],
                             inner_channel=self.cfg['fpn_channel'],
                             num_cls=self.cfg['num_cls'],
                             num_convs=self.cfg['head_conv_num'],
                             layer_num=5,
                             strides=self.cfg['strides'],
                             block_type=self.cfg['block_type'],
                             on_reg=self.cfg['on_reg'])
        self.loss = FCOSLoss(
            strides=self.cfg['strides'],
            radius=self.cfg['radius'],
            alpha=self.cfg['alpha'],
            gamma=self.cfg['gamma'],
            iou_type=self.cfg['iou_type'],
            layer_limits=self.cfg['layer_limits'],
            iou_loss_type=self.cfg['iou_loss_type'],
            iou_loss_weight=self.cfg['iou_loss_weight'],
            reg_loss_weight=self.cfg['reg_loss_weight']
        )

    def forward(self, x, targets=None):
        c3, c4, c5 = self.backbones(x)
        p3, p4, p5, p6, p7 = self.neck([c3, c4, c5])
        out = self.head([p3, p4, p5, p6, p7])
        ret = dict()
        if self.training:
            assert targets is not None
            cls_outputs, reg_outputs, iou_outputs, grids = out
            all_cls_loss, all_box_loss, all_iou_loss, num_pos = self.loss(
                cls_outputs, reg_outputs, iou_outputs, grids, targets)
            ret['cls_loss'] = all_cls_loss
            ret['box_loss'] = all_box_loss
            ret['iou_loss'] = all_iou_loss
            ret['match_num'] = num_pos
        else:
            _, _, h, w = x.shape
            for pred in out:
                pred[:, [0, 2]] = pred[:, [0, 2]].clamp(min=0, max=w)
                pred[:, [1, 3]] = pred[:, [0, 2]].clamp(min=0, max=h)
            predicts = non_max_suppression(out,
                                           conf_thresh=self.cfg['conf_thresh'],
                                           iou_thresh=self.cfg['nms_iou_thresh'],
                                           max_det=self.cfg['max_det']
                                           )
            ret['predicts'] = predicts
        return ret


if __name__ == '__main__':
    input_tensor = torch.rand(size=(4, 3, 640, 640)).float()
    net = FCOS()
    net(input_tensor, 1)
    net(input_tensor, 1)
#     # mcls_output, mreg_output, miou_output, mgrid = net(input_tensor, 1)
#     # for cls_out, reg_out, iou_out, gird_out in zip(mcls_output, mreg_output, miou_output, mgrid):
#     #     print(cls_out.shape, reg_out.shape, iou_out.shape, gird_out.shape)
# # # out = net(input_tensor)
# # for item in out:
# #     print(item.shape)
