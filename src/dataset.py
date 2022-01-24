from torchvision.datasets import CocoDetection, VOCDetection
from torch.utils.data import Dataset


class CocoDataset(Dataset):

    def __init__(self, root: str, annotations_file: str):
        self.dataset = CocoDetection(root, annotations_file)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


class PascalVOCDataset(Dataset):

    def __init__(self, root: str, image_set: str = "train", download: bool = False):
        self.dataset = VOCDetection(root, image_set=image_set, download=download)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)
