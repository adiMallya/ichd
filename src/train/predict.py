import cv2
import numpy as np
import warnings

import torch
import torch.nn as nn
from albumentations import Compose, CenterCrop
from albumentations.pytorch import ToTensorV2

from models import resnext101_32x8d_wsl
import argparse

warnings.filterwarnings("ignore")


def get_model(path, n_classes):
    model = resnext101_32x8d_wsl()
    model.fc = torch.nn.Linear(2048, n_classes)

    model = nn.DataParallel(model)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    return model


def get_tensor(img):
    tfms_img = Compose([CenterCrop(200, 200), ToTensorV2()])
    img = cv2.imread(img)
    return tfms_img(image=img)["image"].unsqueeze(0)


def predict(img):
    model = get_model("src/models/yourmodel", n_classes=6)

    label_list = [
        "epidural",
        "intraparenchymal",
        "intraventricular",
        "subarachnoid",
        "subdural",
        "any",
    ]

    input_tensor = get_tensor(img)

    out = model(input_tensor.float())
    out = torch.sigmoid(out)
    out_trgt = torch.round(out)

    out_np = out_trgt.detach().numpy()

    preds = np.where(out_np == 1)[1]

    label = []
    if len(preds):
        for i in preds:
            label.append(label_list[i])
    else:
        label.append("no ich")

    return out, preds, label


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="predict arguments")
    parser.add_argument("img_path", type=str, help="Image Required")
    args = parser.parse_args()
    img_path = args.img_path

    probas, preds, label = predict(img_path)

    print(probas)
    print(preds)
    print(label)
