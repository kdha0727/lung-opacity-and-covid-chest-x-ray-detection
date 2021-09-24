from .classifier import *
from .detector import *
from .efficientnet import *
from .metrics import *

from . import classifier
from . import detector
from . import efficientnet
from . import metrics


__all__ = ['freeze_backbone_gradient']
for _ in (classifier, detector, efficientnet, metrics):
    __all__ += _.__all__


def freeze_backbone_gradient(backbone, startswith=None):
    if startswith:
        def check(parameter_name):
            return any(map(lambda s: parameter_name.startswith(s), startswith))
        if isinstance(startswith, str):
            startswith = (startswith, )
    else:
        def check(parameter_name):
            return bool(parameter_name)
    for name, param in backbone.named_parameters():
        if check(name):
            param.requires_grad_(False)
