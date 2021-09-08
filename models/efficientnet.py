from efficientnet_pytorch import EfficientNet
# from efficientnet_pytorch.utils import get_model_params


def get_efficientnet(depth=4, in_channels=3, image_size=None, **override_params):

    assert 0 <= depth <= 8
    if image_size is not None:
        override_params.update(image_size=image_size)
    net = EfficientNet.from_name('efficientnet-b{}'.format(depth), include_top=False, **override_params)
    if in_channels != 3:
        net._change_in_channels(3)
    net.out_channels = net._bn1.num_features
    return net
