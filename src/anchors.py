import torch


class Anchors:

    def __init__(self, feature_map_sizes=None, scales=None, aspect_ratios=None):

        if feature_map_sizes is None:
            # Default SSD feature map sizes
            self.feature_map_sizes = torch.tensor([38, 19, 10, 5, 3, 1], dtype=torch.long)
        else:
            self.feature_map_sizes = feature_map_sizes

        if scales is None:
            # Default SSD scales
            self.scales = torch.linspace(0.2, 0.9, self.feature_map_sizes.shape[0])
        else:
            self.scales = scales
        if aspect_ratios is None:
            # Default SSD aspect ratios
            self.aspect_ratios = torch.tensor([1, 2, 1/2, 3, 1/3])
        else:
            self.aspect_ratios = aspect_ratios

        # Calculate widths and heights
        self.widths = self.scales.unsqueeze(dim=1) * self.aspect_ratios.sqrt()
        self.heights = self.scales.unsqueeze(dim=1) / self.aspect_ratios.sqrt()

        # For the aspect ratio of 1, we also add a default box whose scale is sqrt(scale[k]*scale[k+1]
        # resulting in 6 default boxes per feature map location.
        additional_scale = [self.scales[k]*self.scales[k+1] for k in range(self.scales.shape[0]-1)]
        additional_scale.append(self.scales[-1])
        additional_scale = torch.tensor(additional_scale).sqrt().unsqueeze(dim=1)

        # Append "bonus" default box (since aspect ratio is 1, we don't need to multiply by it)
        self.widths = torch.hstack([additional_scale, self.widths])
        self.heights = torch.hstack([additional_scale, self.heights])

        # Create tensor of pairs (width, height)
        self.hw_pairs = torch.stack([self.widths.unsqueeze(dim=2), self.heights.unsqueeze(dim=2)], dim=-1)
        self.hw_pairs = self.hw_pairs.squeeze(dim=2)

        # Tensor containing centers of each default box
        self.centers = torch.tensor([((i+0.5)/k, (j+0.5)/k)
                                     for k in self.feature_map_sizes
                                     for j in range(k) for i in range(k)
                                     for _ in range(self.feature_map_sizes.shape[0])])

        # Tensor containing width and height for each default box
        self.hw_grid = torch.tensor([(w, h)
                                     for i, k in enumerate(self.feature_map_sizes)
                                     for _ in range(k*k)
                                     for w, h in self.hw_pairs[i]])

        # Grid of default boxes
        self.grid = torch.stack([self.centers, self.hw_grid], dim=1).reshape((-1, 4))
