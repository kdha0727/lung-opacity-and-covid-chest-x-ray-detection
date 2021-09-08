from . import efficientnet

# # get number of input features for the classifier
# in_features = model.roi_heads.box_predictor.cls_score.in_features
#
# # replace the pre-trained head with a new one
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


def get_full_network(num_classes=4, trainable_backbone_layers=5, pretrained_backbone=True, **kwargs):

    assert 0 <= trainable_backbone_layers <= 5

    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    effnet_backbone = efficientnet.get_efficientnet_backbone(
        depth=4, in_channels=1, image_size=None, pretrained=pretrained_backbone,
    )
    model = FasterRCNN(effnet_backbone, num_classes, rpn_anchor_generator=anchor_generator, **kwargs)
    return model
