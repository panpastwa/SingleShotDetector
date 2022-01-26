from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import torch
import time

from model import SSD
from dataset import CocoDataset, PascalVOCDataset
from utils import save_image_with_boxes, collate_fn
from torchvision.utils import draw_bounding_boxes, save_image
from torchvision.transforms import ConvertImageDtype

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

# dataloader = DataLoader(dataset, batch_size=1, num_workers=2, collate_fn=collate_fn)

device = 'cuda'
ssd = SSD(num_classes=21)
ssd.to(device)

# Warm-up
# for i, (images, targets) in enumerate(dataloader):
#     images = images.to(device)
#     ssd(images, targets)
#     if i == 3:
#         break
#
# for images, targets in dataloader:
#     images = images.to(device)
#     start = time.time()
#     ssd(images, targets)
#     end = time.time()
#     print(f"Inference time per image ({device}) : {(end-start)*1000/images.shape[0]:.3f} ms.")
#     break

optimizer = torch.optim.SGD(params=ssd.parameters(), lr=0.01)
image = image.to(device).unsqueeze(dim=0)
image_uint8 = ConvertImageDtype(torch.uint8)(image[0]).cpu()


# Training on same image
for epoch in range(100):

    print(f"Epoch {epoch:3d}")
    ssd.train()
    loss = ssd(image, [target])
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss:5.2f} | Allocated memory: {torch.cuda.memory_allocated('cuda')}")
    print('-'*50)

ssd.eval()
detections = ssd(image)
print(target)
scores = detections[0]["scores"]
boxes = detections[0]["boxes"]
classes = detections[0]["class_ids"]
mask = scores > 0.5
scores = scores[mask]
boxes = boxes[mask]
classes = classes[mask]
print(scores, classes, boxes)
image_with_boxes = draw_bounding_boxes(image_uint8, boxes.cpu()*300)
save_image(ConvertImageDtype(torch.float)(image_with_boxes), f"../outputs/image.png")
