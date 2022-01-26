from torchvision.ops import box_iou, box_convert, nms
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

        self.smooth_l1_loss = torch.nn.SmoothL1Loss(reduction="sum")
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")

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

        # Xavier init
        for module_list in (self.module_list, self.classifiers):
            for module in module_list:
                for layer in module:
                    if isinstance(layer, torch.nn.Conv2d):
                        torch.nn.init.xavier_uniform_(layer.weight)

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

    def match_ground_truth_to_anchors(self, target):

        matches = []
        for target in target:

            class_ids, ground_truth_boxes = target["class_ids"], target["boxes"]
            anchors = self.anchors.default_boxes_xyxy
            classes_ids, anchors_ids, gt_bbox_ids = [], [], []

            ious = box_iou(ground_truth_boxes, anchors)

            # Get anchors of highest IoU with given ground truth box
            filtered_anchor_ids = torch.argmax(ious, dim=1)
            indices = torch.arange(filtered_anchor_ids.shape[0])
            classes_ids.extend(class_ids)
            anchors_ids.extend(filtered_anchor_ids)
            gt_bbox_ids.extend(indices)

            # Filter anchors of IoU > 0.5 with ground truth boxes
            mask = ious > 0.5
            mask = torch.nonzero(mask)

            # Remove duplicate pairs, that were added in step above
            avoid_duplicates_mask = torch.stack([indices, filtered_anchor_ids], dim=1)
            remove_mask = torch.zeros(mask.shape[0], dtype=torch.bool)
            for pair in avoid_duplicates_mask:
                remove_mask |= (mask == pair).all(dim=1)
            mask = mask[~remove_mask]

            # Add matches for IoU > 0.5
            acceptable_class_ids = class_ids[mask[:, 0]]
            classes_ids.extend(acceptable_class_ids)
            anchors_ids.extend(mask[:, 1])
            gt_bbox_ids.extend(mask[:, 0])

            # Convert to tensor
            classes_ids = torch.tensor(classes_ids, dtype=torch.long)
            anchors_ids = torch.tensor(anchors_ids, dtype=torch.long)
            gt_bbox_ids = torch.tensor(gt_bbox_ids, dtype=torch.long)

            matches.append({"class_ids": classes_ids,
                            "anchors_ids": anchors_ids,
                            "gt_bbox_ids": gt_bbox_ids})

        return matches

    def regress_offsets(self, matches, target):

        target_offsets = []
        for matches_dict, target_dict in zip(matches, target):

            ground_truth_boxes = box_convert(target_dict["boxes"], "xyxy", "cxcywh")
            gt_bbox_ids = matches_dict["gt_bbox_ids"]
            ground_truth_boxes = ground_truth_boxes[gt_bbox_ids]

            anchor_ids = matches_dict["anchors_ids"]
            default_boxes = self.anchors.default_boxes_cxcywh[anchor_ids]
            diffs = torch.empty_like(ground_truth_boxes)
            diffs[:, :2] = (ground_truth_boxes[:, :2] - default_boxes[:, :2]) / default_boxes[:, 2:]
            diffs[:, 2:] = torch.log(ground_truth_boxes[:, 2:] / default_boxes[:, 2:])
            target_offsets.append(diffs)

        return target_offsets

    def calculate_loss(self, detections, matches, offsets, alpha=1.0):

        loss = torch.tensor(0.0, device=detections.device)
        batch_size = detections.shape[0]

        for detections, matches, offsets in zip(detections, matches, offsets):

            positive_ids = matches["anchors_ids"]
            class_ids = matches["class_ids"]

            positives_detections = detections[positive_ids]

            detection_offsets = positives_detections[:, :4]
            class_score_logits = detections[:, 4:]

            # Debug prints
            # class_probs = class_score_logits.softmax(dim=-1)
            # background_detections = (class_probs[:, 0] > 0.5).count_nonzero()
            # class_detections = (class_probs[:, 1:] > 0.5).count_nonzero()
            # print(f"Number of detections: {detections.shape[0]}")
            # print(f"Number of background detections: {background_detections.item()}")
            # print(f"Number of class detections: {class_detections.item()}")
            # print(f"Best score: {class_probs.max().item():.3f}")

            localization_loss = self.smooth_l1_loss(detection_offsets, offsets.to(detection_offsets.device))

            class_targets = torch.zeros(detections.shape[0], dtype=torch.long)
            class_targets[positive_ids] = class_ids

            confidence_loss = self.cross_entropy_loss(class_score_logits, class_targets.to(detections.device))

            mask = torch.ones_like(confidence_loss, dtype=torch.bool)
            mask[positive_ids] = False

            # Hard negative mining
            negatives = confidence_loss[mask]
            highest_confidence_negatives_ids = torch.argsort(negatives, descending=True)
            highest_confidence_negatives_ids = highest_confidence_negatives_ids[:3*positive_ids.shape[0]]  # neg:pos 3:1
            highest_confidence_negatives = negatives[highest_confidence_negatives_ids]
            # print(f"Confidence loss (pos): {confidence_loss[positive_ids].sum():5.2f} | "
            #       f"Confidence loss (neg): {highest_confidence_negatives.sum():5.2f}")

            confidence_loss = confidence_loss[positive_ids].sum() + highest_confidence_negatives.sum()
            # print(f"Localization loss: {localization_loss:5.2f} | Confidence loss: {confidence_loss:5.2f}")

            loss += 1/positive_ids.shape[0] * (confidence_loss + alpha*localization_loss)

        loss /= batch_size
        return loss

    def prepare_output(self, detections, max_detections=10):

        # Filter boxes with low confidence score
        scores = detections[:, :, 4:]
        scores = scores.softmax(dim=-1)
        best_scores_values, best_scores_indices = torch.max(scores[:, :, 1:], dim=-1)
        score_mask = best_scores_values > 0.01

        output = []

        for i, detection in enumerate(detections):

            print(f"Number of filtered boxes (confidence < 0.01): {(~score_mask[i]).count_nonzero()}")

            mask = score_mask[i]

            # Get scores
            score = best_scores_values[i, mask]
            class_ids = best_scores_indices[i, mask] + 1  # Adding one because we omit background class

            boxes = self.anchors.default_boxes_cxcywh.to(detections.device)
            boxes = boxes[mask]
            detection = detection[mask]

            # Add offsets to default boxes
            # boxes += detection[:, :4]
            boxes[:, :2] += detection[:, :2]*boxes[:, 2:]
            boxes[:, 2:] *= torch.exp(detection[:, 2:4])

            boxes = box_convert(boxes, "cxcywh", "xyxy")
            boxes = torch.clip(boxes, 0.0, 1.0)
            best_boxes = nms(boxes, score, iou_threshold=0.45)

            # Leave only N boxes
            best_boxes = best_boxes[:max_detections]

            output.append({"scores": score[best_boxes], "class_ids": class_ids[best_boxes], "boxes": boxes[best_boxes]})

        return output

    def forward(self, x, target=None):

        if len(x.shape) != 4 or x.shape[1] != 3 or x.shape[2] != 300 or x.shape[3] != 300:
            raise ValueError("Input should have shape [BATCH_SIZE, 3, 300, 300]")

        if self.training and target is None:
            raise ValueError("Target is required during training")

        if self.training and x.shape[0] != len(target):
            raise ValueError(f"Target should have same number of items as in batch "
                             f"(batch size: {x.shape[0]}, target size: {len(target)})")

        features = x
        output = []

        for feature_extractor, classifier in zip(self.module_list, self.classifiers):

            features = feature_extractor(features)
            detections = classifier(features)
            detections = self.reshape_detections(detections)

            output.append(detections)

        # Merge tensors together and return detections
        detections = torch.cat(output, dim=1)

        if self.training:
            matches = self.match_ground_truth_to_anchors(target)
            offsets = self.regress_offsets(matches, target)
            loss = self.calculate_loss(detections, matches, offsets)
            return loss

        else:
            output = self.prepare_output(detections)
            return output
