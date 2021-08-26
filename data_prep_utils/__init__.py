# ****************************** Enhanced Dataset Wrappers ******************************
#
# Usage:
#
# import data_prep_utils
# data_prep_utils.set_root("/content/drive/Shareddrives/{your-data-root-folder}/")
# from data_prep_utils import covid_19_radiography_dataset
# data = covid_19_radiography_dataset.get_torch_dataset(transform=transforms.ToTensor())
# loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
# ...
#


#
# modules
#

from . import dataset


#
# root directory getters and setters
#
# set_root: this function configures root data directory. default is "./data" directory.
# get_root: this function returns configured root data directory ""lazily"".
from ._internal import get_root, set_root, lazy_init as init, _register_init_hook


#
# lazy data wrappers
#

from . import wrapper

covid_19_radiography_dataset = wrapper.COVID19RadiologyDataset()
rsna_pneumonia_detection_challenge = wrapper.RSNAPneumoniaDetectionChallenge()

for w in (
        covid_19_radiography_dataset,
        rsna_pneumonia_detection_challenge
):
    # lambda is lazy-evaluated, so do not use lambda expression.
    def __hook(__obj=w):
        return __obj.__data_root__  # warning: this may cause recursion while initialization
    _register_init_hook(__hook)
del wrapper, w, _register_init_hook, __hook  # remove redundant classes from namespace


__all__ = [

    # modules,
    'dataset',

    # root directory getters and setters,
    'get_root', 'set_root', 'init',

    # and wrapped datasets.
    'rsna_pneumonia_detection_challenge',
    'covid_19_radiography_dataset'

]
