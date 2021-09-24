from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform


def get_detector(backbone, num_classes=2, max_size=800, min_size=1333, **kwargs):

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    detector = FasterRCNN(backbone, num_classes, rpn_anchor_generator=anchor_generator, **kwargs)
    detector.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean=[0.485], image_std=[0.229])
    return detector


__all__ = ['FasterRCNN', 'AnchorGenerator', 'GeneralizedRCNNTransform', 'get_detector']
