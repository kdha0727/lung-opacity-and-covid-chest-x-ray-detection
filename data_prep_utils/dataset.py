import numpy as np
import pandas as pd
import pydicom
import os
import typing
from PIL import Image
from torchvision.datasets import VisionDataset, ImageFolder as _ImageFolder


# loader

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # issue: to use one-channel image, we use "L" conversion.
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        return img.convert('L')


def dicom_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        dcm = pydicom.dcmread(f)
        arr = dcm.pixel_array
        return Image.fromarray(arr)


def default_loader(path: str):
    if os.path.splitext(path)[-1].lower() == '.dcm':
        return dicom_loader(path)
    else:
        return pil_loader(path)


# dataset class

class ImageWithPandas(VisionDataset):
    """A generic data loader where the image path and label is given as pandas DataFrame.

    Args:
        dataframe (pandas.DataFrame): A data table that contains image path, target class,
            and extra outputs.
        label_path (string): Data frame`s image path label string.
        label_target (string): Data frame`s target class label string.
        label_extras (tuple[string] or string, optional): Data frame`s label that will
            be used for extra outputs.
        root (string, optional): Root directory path. Use unless data frame`s column
            contains file folders.
        extension (string, optional): An extension that will be concatenated after
            image file name. Use unless data frame`s column contains extension.
        class_to_idx (dict[str, int], optional): A mapping table that converts class
            label string into integer value. If not given, sorted index value will
            be used as class integer value.
        transform (callable, optional): A function/transform that takes in an image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        extras_transform (callable, optional): A function/transform that takes in the
            extra outputs and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(
            self,
            dataframe: pd.DataFrame,
            label_path: str,
            label_target: str,
            label_extras: typing.Optional[typing.Union[typing.Iterable[str], str]] = None,
            root: typing.Optional[typing.Union[str, os.PathLike]] = None,
            extension: typing.Optional[str] = None,
            class_to_idx: typing.Optional[typing.Dict[typing.Any, int]] = None,
            transform: typing.Optional[typing.Callable] = None,
            target_transform: typing.Optional[typing.Callable] = None,
            extras_transform: typing.Optional[typing.Callable] = None,
            loader: typing.Callable[[str], typing.Any] = default_loader,
    ) -> None:

        super(ImageWithPandas, self).__init__(root, None, transform, target_transform)

        self.extras_transform = extras_transform
        self.loader = loader
        self.label_path = label_path
        self.label_target = label_target

        labels = [label_path, label_target]
        if isinstance(label_extras, str):
            labels.append(label_extras)
            label_extras = [label_extras]
        elif label_extras is not None:
            label_extras = list(label_extras)
            labels.extend(label_extras)

        self.label_extras = label_extras

        samples = dataframe[labels].copy(deep=True)

        assert extension.startswith('.') or extension is None
        if root is not None:
            root = os.path.expanduser(root)
        if root is not None or extension is not None:
            samples[label_path] = samples[label_path].map(
                (lambda x: os.path.join(root, x + extension or ''))
                if root is not None else (lambda x: x + extension)
            )

        classes = sorted(samples[label_target].unique())
        if class_to_idx is None:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        samples[label_target] = samples[label_target].map(lambda x: class_to_idx[x])
        samples = samples.drop_duplicates()

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.num_classes = len(class_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> ...:
        row = self.samples.iloc[index]
        path, target = row[self.label_path], row[self.label_target]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.label_extras:
            if self.extras_transform is not None:
                extra = map(self.extras_transform, row[self.label_extras].values)
            else:
                extra = iter(row[self.label_extras].values)
        else:
            extra = ()
        return tuple((sample, target, *extra))


class ImageFolder(_ImageFolder):
    """A generic data loader where the images are arranged in root folder.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: typing.Union[str, os.PathLike],
            class_to_idx: typing.Optional[typing.Dict[str, int]] = None,
            transform: typing.Optional[typing.Callable] = None,
            target_transform: typing.Optional[typing.Callable] = None,
            loader: typing.Callable[[str], typing.Any] = default_loader,
            is_valid_file: typing.Optional[typing.Callable[[str], bool]] = None,
    ):
        self.class_to_idx = class_to_idx
        super(ImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)

    def _find_classes(self, directory: str) -> typing.Tuple[typing.List[str], typing.Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            directory (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        try:
            class_to_idx = self.class_to_idx
        except AttributeError:
            class_to_idx = None
        if class_to_idx is None:
            class_to_idx = self.class_to_idx or {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


__all__ = ['pil_loader', 'dicom_loader', 'ImageWithPandas', 'ImageFolder']
