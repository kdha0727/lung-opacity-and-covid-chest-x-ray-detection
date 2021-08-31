import pydicom
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms

from . import dataset
from ._internal import get_root

#
# lazy data wrapper classes
#


class DataWrapper:
    # set data-subdirectory name as __root__,
    # then full directory can be accessed as __data_root__ property.
    # __data_root__ property is lazily evaluated,
    # so you can set root directory before, by calling set_root().
    __root__: str = ...

    @property
    def __data_root__(self):
        if self.__data_root_cache is not None:
            return self.__data_root_cache
        cache = get_root() / self.__root__
        if not cache.is_dir():
            raise ValueError("Data sub-directory not exists: %s" % cache)
        self.__data_root_cache = cache
        return cache

    __data_root_cache = None


class RSNAPneumoniaDetectionChallenge(DataWrapper):
    __root__ = "rsna-pneumonia-detection-challenge"
    class_to_idx = {1: 1, 0: 0}

    @property
    def image_path(self):
        return self.__data_root__ / "stage_2_train_images"

    @property
    def image_path_str(self):
        return str(self.image_path)

    @property
    def class_info_csv(self):
        path = self.__data_root__ / "stage_2_detailed_class_info.csv"
        return pd.read_csv(path, encoding='utf-8')

    @property
    def train_labels_csv(self):
        path = self.__data_root__ / "stage_2_train_labels.csv"
        return pd.read_csv(path, encoding='utf-8')

    @property
    def classification_csv(self):
        return self.train_labels_csv[["patientId", "Target"]].drop_duplicates()

    @property
    def full_csv(self):
        df1 = pd.DataFrame(data=self.class_info_csv)
        df2 = pd.DataFrame(data=self.train_labels_csv)
        df2 = df2.drop([df2.columns[0]], axis=1)
        result = pd.concat([df1, df2], axis=1)
        return result

    @property
    def lung_opacity_csv(self):
        full_csv = self.full_csv
        df_lung_opacity = full_csv[full_csv['Target'] == 1]
        df_lung_opacity.index = list(range(len(df_lung_opacity)))
        return df_lung_opacity

    def get_random_patient_id(self):
        return os.path.splitext(random.choice(os.listdir(self.image_path_str)))[0]

    def get_patient_path(self, patient_id: str) -> str:
        return str(self.image_path / "{}.dcm".format(patient_id))

    def get_patient_dicom(self, patient_id: str) -> pydicom.FileDataset:
        return pydicom.dcmread(self.get_patient_path(patient_id))

    def get_patient_metadata(self, patient_id: str) -> str:
        return str(self.get_patient_dicom(patient_id))

    def get_patient_image(self, patient_id: str) -> np.ndarray:
        return self.get_patient_dicom(patient_id).pixel_array

    def show_patient_image(self, patient_id):
        plt.imshow(self.get_patient_image(patient_id), cmap=plt.cm.gray)
        plt.show()

    def torch_classification_dataset(self, transform=transforms.ToTensor()):
        return dataset.ImageWithPandas(
            dataframe=self.classification_csv,
            label_id='patientId',
            label_target='Target',
            root=self.image_path,
            extension='.dcm',
            transform=transform,
            loader=dataset.dicom_loader,
            class_to_idx=self.class_to_idx,
        )

    def torch_detection_dataset(self, transforms):
        return dataset.ImageBboxWithPandas(
            dataframe=self.full_csv,
            label_id='patientId',
            label_bbox="x y width height".split(),
            label_target='Target',
            root=self.image_path,
            extension='.dcm',
            transforms=transforms,
            loader=dataset.dicom_loader,
            class_to_idx=self.class_to_idx,
        )


class COVID19RadiologyDataset(DataWrapper):
    __root__ = "COVID-19_Radiography_Dataset"
    class_to_idx = {'Normal': 0, 'Lung_Opacity': 1, 'COVID': 2, 'Viral Pneumonia': 3}

    @property
    def metadata_csv(self):
        # dataframe length: 21164
        # columns: file_name, file_format(identical), image_shape(identical), label
        # file_format: PNG
        # image_shape: (299, 299)
        path = self.__data_root__ / "metadata.csv"
        df = pd.read_csv(path, index_col=0)
        del df["image_data_grayscale"]
        df["Target"] = df["label"].map(lambda x: self.class_to_idx[x])
        return df

    @property
    def classification_csv(self):
        return self.metadata_csv[["file_name", "Target"]]

    @property
    def image_path(self):
        return self.__data_root__ / "COVID-19_Radiography_Dataset"

    def torch_classification_dataset(self, transform=transforms.ToTensor()):
        return dataset.ImageFolder(
            root=self.image_path,
            class_to_idx=self.class_to_idx,
            transform=transform,
            loader=dataset.pil_loader,
        )


__all__ = ['RSNAPneumoniaDetectionChallenge', 'COVID19RadiologyDataset']
