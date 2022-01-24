import torch
import time

from model import SSD
from dataset import CocoDataset, PascalVOCDataset
from utils import save_image_with_boxes

root = "/media/panpastwa/Vincent/val2017"
annotations_file = "/home/panpastwa/Downloads/instances_val2017.json"
dataset = CocoDataset(root, annotations_file)
image, target = dataset[0]
boxes = []
for obj in target:
    bbox = obj['bbox']
    bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
    boxes.append(bbox)
save_image_with_boxes(image, "image1.png", boxes)

root = "/media/panpastwa/Vincent/PascalVOC"
dataset = PascalVOCDataset(root)
image, target = dataset[0]
boxes = []
for obj in target["annotation"]["object"]:
    box = obj["bndbox"]
    boxes.append([int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])])
save_image_with_boxes(image, "image2.png", boxes)


device = 'cuda'
ssd = SSD(num_classes=91)
ssd.to(device)
ssd.eval()
x = torch.rand((1, 3, 300, 300)).to(device)

# Warm-up
ssd(x)

start = time.time()
output = ssd(x)
end = time.time()
print(f"Inference time: {(end-start)*1000:.3f} ms.")
print(output.shape)
