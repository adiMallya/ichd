import cv2
from io import BytesIO
import numpy as np
import warnings

import torch
import torch.nn as nn
from albumentations import Compose, CenterCrop
from albumentations.pytorch import ToTensorV2

from train.models import resnext101_32x8d_wsl
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
    # start = time.time()
    model = get_model("/workspace/ichd/models/yourmodel.pt", n_classes=6)
    # print("--- %s seconds ---" % (time.time() - start))
    label_list = [
        "epidural",
        "intraparenchymal",
        "intraventricular",
        "subarachnoid",
        "subdural",
        "any",
    ]

    input_tensor = get_tensor(img).float()

    out = model(input_tensor)
    out = torch.sigmoid(out)
    out_trgt = torch.round(out)

    out_np = out_trgt.detach().numpy()

    preds = np.where(out_np == 1)[1]

    probas = torch.round((out) * 100).detach().numpy()[0]
    # print(probas)

    proba_ord = np.argsort(out.detach().numpy())[0][::-1]
    # print(proba_ord)

    label = []
    if len(preds):
        for i in preds:
            label.append(label_list[i])
    else:
        label.append("no ich")

    arg_s = {}
    for i in proba_ord:
        arg_s[label_list[int(i)]] = probas[int(i)]

    _l = list(arg_s.items())
    cd = [": ".join(map(str, tup)) for tup in _l]
    cd = "-".join(cd)

    return str(label) + '@', str(cd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="predict arguments")
    parser.add_argument("img_path", type=str, help="Image Required")
    args = parser.parse_args()
    img_path = args.img_path

    out = predict(img_path)

    for i in out:
        print(str(i))
