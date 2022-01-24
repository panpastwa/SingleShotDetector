import torch

from anchors import Anchors


class SSD(torch.nn.Module):

    def __init__(self, num_classes, anchors: Anchors = None):
        super().__init__()

        self.num_classes = num_classes

        if anchors is None:
            self.anchors = Anchors()
        else:
            self.anchors = anchors

        self.module_list = torch.nn.ModuleList([

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

        # Convolutional classifiers that map features to detections
        self.classifiers = torch.nn.ModuleList([

            # Temporary changing shape of classifiers for convinient anchor testing

            torch.nn.Sequential(
                # torch.nn.Conv2d(512, 4 * (self.num_classes+4), kernel_size=3, padding=1),
                torch.nn.Conv2d(512, 6 * (self.num_classes+4), kernel_size=3, padding=1),
                torch.nn.ReLU(),
            ),

            torch.nn.Sequential(
                torch.nn.Conv2d(1024, 6 * (self.num_classes + 4), kernel_size=3, padding=1),
                torch.nn.ReLU(),
            ),

            torch.nn.Sequential(
                torch.nn.Conv2d(512, 6 * (self.num_classes + 4), kernel_size=3, padding=1),
                torch.nn.ReLU(),
            ),

            torch.nn.Sequential(
                torch.nn.Conv2d(256, 6 * (self.num_classes + 4), kernel_size=3, padding=1),
                torch.nn.ReLU(),
            ),

            torch.nn.Sequential(
                # torch.nn.Conv2d(256, 4 * (self.num_classes + 4), kernel_size=3, padding=1),
                torch.nn.Conv2d(256, 6 * (self.num_classes + 4), kernel_size=3, padding=1),
                torch.nn.ReLU(),
            ),

            torch.nn.Sequential(
                # torch.nn.Conv2d(256, 4 * (self.num_classes + 4), kernel_size=3, padding=1),
                torch.nn.Conv2d(256, 6 * (self.num_classes + 4), kernel_size=3, padding=1),
                torch.nn.ReLU(),
            ),
        ])

    def reshape_detections(self, detections):
        """
        Reshape classifier output to standard detection shape
        :param detections: Tensor contatining output from classifier in shape [batch_size, K*(num_classes+4), N, N]
        :return: Tensor containing reshaped detections in shape [batch_size, num_detections, num_classes+4]
        """

        batch_size, K, H, W = detections.shape
        K = K // (self.num_classes+4)

        # As we expect input as images in shape (300, 300), we expect H == W
        assert H == W

        # Split second dimension to K*(num_classes+4)
        detections = detections.view(batch_size, K, self.num_classes+4, H, W)

        # Change order in detections tensor
        detections = detections.permute(0, 3, 4, 1, 2)

        # Reshape detections to desired shape (batch_size, num_detections, num_classes+4)
        detections = detections.reshape(batch_size, K*H*W, self.num_classes+4)

        return detections

    def forward(self, x, target=None):

        if self.training and target is None:
            raise ValueError("Target is required during training")

        features = x
        output = []

        for feature_extractor, classifier in zip(self.module_list, self.classifiers):

            features = feature_extractor(features)
            detections = classifier(features)
            detections = self.reshape_detections(detections)

            output.append(detections)

        # Merge tensors together and return detections
        output = torch.cat(output, dim=1)
        return output
