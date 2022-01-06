import torch


class SSD(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = torch.nn.ModuleList([

            # VGG to Conv4_3
            torch.nn.Sequential(

                # Conv1
                torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),

                # Conv2
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),

                # Conv3
                torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # ceil_mode=True for same size as in paper

                # Conv4
                torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2),
            ),

            # VGG to Conv5_3 and converted FC to Conv
            torch.nn.Sequential(

                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling from previous conv layer

                # Conv5
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                # torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # As noted by authors in paper on page 7

                # FC changed to Conv as noted by authors in paper on page 7
                torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
                torch.nn.ReLU(),
            ),

            # Extra feature layers
            torch.nn.Sequential(
                torch.nn.Conv2d(1024, 256, kernel_size=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
            ),

            torch.nn.Sequential(
                torch.nn.Conv2d(512, 128, kernel_size=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
            ),

            torch.nn.Sequential(
                torch.nn.Conv2d(256, 128, kernel_size=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 256, kernel_size=3),
                torch.nn.ReLU(),
            ),

            torch.nn.Sequential(
                torch.nn.Conv2d(256, 128, kernel_size=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 256, kernel_size=3),
                torch.nn.ReLU(),
            )
        ])

    def debug_shape(self, x):
        for block in self.model:
            x = block(x)
            print(x.shape)

    def forward(self, x):

        for block in self.model:
            x = block(x)
        return x
