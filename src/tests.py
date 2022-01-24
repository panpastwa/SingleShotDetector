from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import torch
import time

from model import SSD
from dataset import CocoDataset, PascalVOCDataset
from utils import save_image_with_boxes, collate_fn

# root = "/media/panpastwa/Vincent/val2017"
# annotations_file = "/home/panpastwa/Downloads/instances_val2017.json"
# dataset = CocoDataset(root, annotations_file)
# image, target = dataset[0]
# boxes = target["boxes"]
# image = ToPILImage()(image)
# save_image_with_boxes(image, "image1.png", boxes)

root = "/media/panpastwa/Vincent/PascalVOC"
dataset = PascalVOCDataset(root)
image, target = dataset[0]
# boxes = target["boxes"]
# image = ToPILImage()(image)
# save_image_with_boxes(image, "image2.png", boxes)


dataloader = DataLoader(dataset, batch_size=16, num_workers=2, collate_fn=collate_fn)

device = 'cuda'
ssd = SSD(num_classes=21)
ssd.to(device)

# Warm-up
for i, (images, targets) in enumerate(dataloader):
    images = images.to(device)
    ssd(images, targets)
    if i == 3:
        break

for images, targets in dataloader:
    images = images.to(device)
    start = time.time()
    ssd(images, targets)
    end = time.time()
    print(f"Inference time per image ({device}) : {(end-start)*1000/images.shape[0]:.3f} ms.")
    break
