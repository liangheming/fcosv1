import torch
from nets.fcos import FCOS
from losses.fcos_loss import FCOSLoss
from datasets.coco import COCODataSets
from torch.utils.data.dataloader import DataLoader

if __name__ == '__main__':
    dataset = COCODataSets(img_root="/home/huffman/data/val2017",
                           annotation_path="/home/huffman/data/annotations/instances_val2017.json",
                           use_crowd=True,
                           augments=True,
                           remove_blank=True,
                           img_size=640
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)
    net = FCOS(
        backbone="resnet18"
    )
    creterion = FCOSLoss()
    for img_input, targets, _ in dataloader:
        _, _, h, w = img_input.shape
        targets[:, 3:] = targets[:, 3:] * torch.tensor(data=[w, h, w, h])
        cls_outputs, reg_outputs, center_outputs, grids = net(img_input)
        total_loss, detail_loss, num_match = creterion(cls_outputs, reg_outputs, center_outputs, grids,
                                                       targets)
        cls_loss, reg_loss, center_loss = detail_loss
        print(total_loss)
        print(cls_loss, reg_loss, center_loss)
        break
