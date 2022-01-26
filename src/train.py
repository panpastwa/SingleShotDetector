import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from model import SSD
from dataset import PascalVOCDataset
from utils import collate_fn

DEVICE = 'cuda'

root = "/media/panpastwa/Vincent/PascalVOC"
dataset = PascalVOCDataset(root)
dataloader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True, collate_fn=collate_fn)

num_classes = len(dataset.classes)
ssd = SSD(num_classes=num_classes)
ssd.to(DEVICE)

optimizer = torch.optim.SGD(params=ssd.parameters(), lr=0.001)

iterator = tqdm(dataloader)
for epoch in range(5):
    losses = []
    for i, (images, targets) in enumerate(iterator):
        images = images.to(DEVICE)
        loss = ssd(images, targets)
        loss.backward()
        optimizer.step()
        iterator.set_description(f"Loss: {loss:.2f}")
        losses.append(loss)

    torch.save(ssd.state_dict(), f"../weights/weights{epoch+1}")
    losses = torch.tensor(losses).cpu()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()
