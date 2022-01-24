from PIL import Image, ImageDraw
import torch


def collate_fn(data):
    images = torch.cat([d[0].unsqueeze(dim=0) for d in data])
    targets = [d[1] for d in data]
    return images, targets


def save_image_with_boxes(image: Image, save_path: str, boxes):
    """
    Save image with drawn boxes onto it.
    :param image: PIL Image
    :param save_path: Path where new image will be saved
    :param boxes: List of boxes
    :return: None
    """

    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    new_boxes = boxes.clone()
    new_boxes[:, (0, 2)] *= image.width
    new_boxes[:, (1, 3)] *= image.height
    for bbox in new_boxes:
        draw.rectangle(bbox.tolist())
    image_with_boxes.save(save_path)
