from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes, save_image
from torchvision.transforms import ConvertImageDtype
import torch

from model import SSD
from dataset import PascalVOCDataset
from utils import collate_fn

DEVICE = 'cuda'

root = "/media/panpastwa/Vincent/PascalVOC"
dataset = PascalVOCDataset(root, image_set="val")
dataloader = DataLoader(dataset, batch_size=8, num_workers=2, collate_fn=collate_fn)

num_classes = len(dataset.classes)
ssd = SSD(num_classes=num_classes)
ssd.eval()
ssd.to(DEVICE)
ssd.load_state_dict(torch.load("../weights/weights2"))

with torch.no_grad():
    for i, (images, targets) in enumerate(dataloader):
        images = images.to(DEVICE)
        detections = ssd(images, targets)

        for j, (image, detection) in enumerate(zip(images, detections)):
            image_uint8 = ConvertImageDtype(torch.uint8)(image)
            image_with_boxes = draw_bounding_boxes(image_uint8.cpu(), detection["boxes"].cpu()*300)
            save_image(ConvertImageDtype(torch.float)(image_with_boxes), f"../outputs/image{i}{j}.png")

