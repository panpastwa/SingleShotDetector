from PIL import Image, ImageDraw


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
    for bbox in boxes:
        bbox = (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
        draw.rectangle(bbox)
    image_with_boxes.save(save_path)
