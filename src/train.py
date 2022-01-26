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

mean_losses = []
for epoch in range(3):
    losses = []
    iterator = tqdm(dataloader)
    for i, (images, targets) in enumerate(iterator):
        images = images.to(DEVICE)
        loss = ssd(images, targets)
        loss.backward()
        optimizer.step()
        iterator.set_description(f"Loss: {loss:.2f}")
        losses.append(loss)
        if i == 10:
            break

    torch.save(ssd.state_dict(), f"../weights/weights{epoch+1}")

    losses = torch.tensor(losses).cpu()
    mean_losses.append(losses.mean())

    plt.figure()
    plt.plot(losses)
    plt.title(f"Epoch {epoch+1:2d}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(f"../figures/training_epoch_{epoch+1:2d}")

plt.figure()
plt.plot(mean_losses)
plt.title(f"Training curve")
plt.xlabel("Epoch")
plt.ylabel("Mean loss")
plt.savefig("../figures/training_curve.png")
