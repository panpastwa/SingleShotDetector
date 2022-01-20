from torchvision.datasets import CocoDetection
import torch
import time

from model import SSD
from utils import save_image_with_boxes


image_dir = "/media/panpastwa/Vincent/val2017"
annotations_file = "/home/panpastwa/Downloads/instances_val2017.json"
dataset = CocoDetection(image_dir, annotations_file)

item = dataset[0]
image = item[0]
boxes = []
for obj in item[1]:
    if obj['category_id'] == 1:
        boxes.append(obj['bbox'])
save_image_with_boxes(image, "image.png", boxes)

device = 'cuda'
ssd = SSD(num_classes=91)
ssd.to(device)
x = torch.rand((1, 3, 300, 300)).to(device)

# Warm-up
ssd(x)

start = time.time()
output = ssd(x)
end = time.time()
print(f"Inference time: {(end-start)*1000:.3f} ms.")
print(output.shape)
