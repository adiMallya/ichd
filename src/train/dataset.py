import os
import numpy as np
import pandas as pd
import cv2

# import pydicom
import torch
from torch.utils.data import Dataset

from utils import window_image


def window_processing(dcm):
    """
    A function to generate brain, subdural and soft-tissue windows
    from a DICOM file & concatenate them into a single three-channel
    image
    """
    # Windowing
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)

    # Standardising
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    # Concatenating
    input_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

    return input_img


# Declaring dataset class
class IntracranialDataset(Dataset):
    def __init__(self, csv_file, path, labels, transform=None):
        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = self.data.loc[idx, "Image"]
        img_name = os.path.join(self.path, img_id + ".png")
        img = cv2.imread(img_name)
        #      img_id = self.data.loc[idx, 'ImageId']

        # try:
        #    img = pydicom.dcmread(self.path, img_id + '.dcm')
        # img = bsb_window(dicom)
        # except:
        #   img = np.zeros((512, 512, 3))

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        if self.labels:

            labels = torch.tensor(
                self.data.loc[
                    idx,
                    [
                        "epidural",
                        "intraparenchymal",
                        "intraventricular",
                        "subarachnoid",
                        "subdural",
                        "any",
                    ],
                ]
            )
            return {"image_id": img_id, "image": img, "labels": labels}
        else:

            return {"image_id": img_id, "image": img}
