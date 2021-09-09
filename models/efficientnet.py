from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import GlobalParams
# from efficientnet_pytorch.utils import get_model_params


def get_efficientnet_backbone(depth=4, in_channels=1, image_size=256, pretrained=True, **override_params):

    assert 0 <= depth <= 8
    if image_size is not None:
        override_params.update(image_size=image_size)
    model_name = 'efficientnet-b{}'.format(depth)
    if pretrained:
        net = EfficientNet.from_pretrained(model_name, in_channels=in_channels, **override_params)
        global_params = net._global_params
        override_global_params = GlobalParams(
            width_coefficient=global_params.width_coefficient,
            depth_coefficient=global_params.depth_coefficient,
            image_size=global_params.image_size,
            dropout_rate=global_params.dropout_rate,
            num_classes=global_params.num_classes,
            batch_norm_momentum=global_params.batch_norm_momentum,
            batch_norm_epsilon=global_params.batch_norm_epsilon,
            drop_connect_rate=global_params.drop_connect_rate,
            depth_divisor=global_params.depth_divisor,
            min_depth=global_params.min_depth,
            include_top=False,
        )
        net._global_params = override_global_params
        del net._dropout, net._fc
    else:
        net = EfficientNet.from_name(model_name, include_top=False, **override_params)
        net._change_in_channels(in_channels)
    net.out_channels = net._bn1.num_features
    return net
