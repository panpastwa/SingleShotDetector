import torch
import time

from model import SSD


device = 'cuda'
ssd = SSD(num_classes=91)
ssd.to(device)
print(ssd)
x = torch.rand((2, 3, 300, 300)).to(device)
s = time.time()
output = ssd(x)
print(f"Inference time: {(time.time()-s)*1000:.2f} ms.")
print(output.shape)
