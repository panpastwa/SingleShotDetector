from torchvision.datasets import CocoDetection, VOCDetection
from torchvision.transforms import Compose, Resize, PILToTensor, ConvertImageDtype
from torch.utils.data import Dataset
import torch


class CocoDataset(Dataset):

    def __init__(self, root: str, annotations_file: str):
        self.dataset = CocoDetection(root, annotations_file)
        self.transforms = Compose([Resize((300, 300)), PILToTensor(), ConvertImageDtype(torch.float)])

    def __getitem__(self, item):
        image, target = self.dataset[item]
        target = self.preprocess_target(target)
        image, target = self.transform(image, target)
        return image, target

    def __len__(self):
        return len(self.dataset)

    def preprocess_target(self, target):
        class_ids, boxes = [], []
        for obj in target:
            class_id = obj["category_id"]
            bbox = obj['bbox']
            bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            class_ids.append(class_id)
            boxes.append(bbox)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        class_ids = torch.tensor(class_ids, dtype=torch.long)
        target = {"class_ids": class_ids, "boxes": boxes}
        return target

    def transform(self, image, target):
        w, h = image.width, image.height
        boxes = target["boxes"]
        boxes[:, (0, 2)] /= w
        boxes[:, (1, 3)] /= h
        image = self.transforms(image)
        return image, target


class PascalVOCDataset(Dataset):

    def __init__(self, root: str, image_set: str = "train", download: bool = False):
        self.dataset = VOCDetection(root, image_set=image_set, download=download)
        self.transforms = Compose([Resize((300, 300)), PILToTensor(), ConvertImageDtype(torch.float)])
        classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                   "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                   "sheep", "sofa", "train", "tvmonitor"]
        self.classes = {k: v for v, k in enumerate(classes)}

    def __getitem__(self, item):
        image, target = self.dataset[item]
        target = self.preprocess_target(target)
        image, target = self.transform(image, target)
        return image, target

    def __len__(self):
        return len(self.dataset)

    def preprocess_target(self, target):
        class_ids, boxes = [], []
        for obj in target["annotation"]["object"]:
            class_id = self.classes[obj["name"]]
            box = obj["bndbox"]
            box = [int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])]
            class_ids.append(class_id)
            boxes.append(box)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        class_ids = torch.tensor(class_ids, dtype=torch.long)
        target = {"class_ids": class_ids, "boxes": boxes}
        return target

    def transform(self, image, target):
        w, h = image.width, image.height
        boxes = target["boxes"]
        boxes[:, (0, 2)] /= w
        boxes[:, (1, 3)] /= h
        image = self.transforms(image)
        return image, target
