# importing libraries
import os
import glob
import pandas as pd
import numpy as np
import pydicom 
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck
import torch
import torch.optim as optim
from albumentations import Compose, ShiftScaleRotate, HorizontalFlip, RandomBrightnessContrast, CenterCrop
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, Subset
import cv2
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, classification_report, recall_score, f1_score, accuracy_score, precision_score, jaccard_score
# from tqdm import notebook as tqdm

from dataset import IntracranialDataset
from models import resnext101_32x8d_wsl
# declaring file paths
dir_csv = '/workspace/ichd/input'
# test_images_dir = '/workspace/ichd/input/stage_1_test_png_224x/'
train_images_dir = '/workspace/ichd/input/stage_1_train_png_224x/'
# train_metadata_csv = '/workspace/ichd/input/train_metadata_noidx.csv'
# test_metadata_csv = '/workspace/ichd/input/test_metadata_noidx.csv'
tb  = SummaryWriter('runs/ich_detection_experiment_1')


# PARAMS
n_classes = 6
n_epochs = 10
batch_size = 32

COLS = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

# Read train and test data
train = pd.read_csv(os.path.join(dir_csv, 'stage_2_train.csv'))
test = pd.read_csv(os.path.join(dir_csv, 'stage_2_sample_submission.csv'))

# Preparing data
train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
train = train[['Image', 'Diagnosis', 'Label']]
train.drop_duplicates(inplace=True)
train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train['Image'] = 'ID_' + train['Image']

# Remove images that are not saved properly as png
png = glob.glob(os.path.join(train_images_dir, '*.png'))
png = [os.path.basename(png)[:-4] for png in png]
png = np.array(png)


train = train[train['Image'].isin(png)]
train.to_csv('train.csv', index=False)


# Dataloader
transform_train = Compose([CenterCrop(200, 200), HorizontalFlip(),
                           ShiftScaleRotate(),
                           RandomBrightnessContrast(),
                           ToTensorV2()])

train_dataset = IntracranialDataset(
    csv_file='train.csv', path=train_images_dir, transform=transform_train, labels=True)

valid_dataset = IntracranialDataset(
    csv_file='train.csv', path=train_images_dir, transform=transform_train, labels=True)

valid_dataset = torch.utils.data.Subset(valid_dataset, range(0, 134359))
# print(len(valid_dataset))

train_dataset = torch.utils.data.Subset(train_dataset, range(134359, len(train_dataset) - 1))
# print(len(train_dataset))

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
print(len(data_loader_train))

data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
print(len(data_loader_valid))


# Declaring the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Importing the model
model = resnext101_32x8d_wsl()
model.fc = torch.nn.Linear(2048, n_classes)
# Using data parallel to use all GPUs
model = torch.nn.DataParallel(model)

model.to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
# LR Decay scheduler
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training starts
for epoch in range(n_epochs):

    print('Epoch {}/{}'.format(epoch + 1, n_epochs))
    print('-' * 10)

    model.train()
    tr_loss = 0
    tr_correct = 0

    # tk0 = tqdm.tqdm(data_loader_train, desc="Iteration")
    for step, batch in enumerate(data_loader_train):

        inputs = batch["image"]
        # print(inputs.shape)
        labels = batch["labels"]
        # print(labels.shape)

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        outputs = model(inputs)
        # print(outputs.shape)
        loss = criterion(outputs, labels)

        new1 = torch.sigmoid(outputs).detach().cpu() >= 0.5
        # Backpropagation
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        # scaled_loss.backward()
        tr_loss += loss.item()
        # tr_correct += torch.sum(preds == labels)
        # print(tr_correct)
        tr_correct += torch.sum(new1 == labels.cpu())
        optimizer.step()
        optimizer.zero_grad()

        if step % 512 == 0:
            epoch_loss = tr_loss / (step + 1)
            print('Training Loss at {}: {:.4f}'.format(step, epoch_loss))

    epoch_loss = tr_loss / len(data_loader_train)
    print('Training Loss: {:.4f}'.format(epoch_loss))
    print('-----------------------')
    # Tensorboard code for visualisations
    tb.add_scalar("Training Loss", tr_loss, epoch)
    # tb.add_scalar("Training Correct preds", tr_correct, epoch)
    tb.add_scalar("Training Accuracy", tr_correct / len(train_dataset), epoch)
    print('Finished Training!')

    model.eval()
    tr_loss = 0
    tr_correct = 0

    print('Validation starts...')
    # Arrays to store metrics for each batch
    prec = []
    rec = []
    acc = []
    f1 = []
    jacc = []

    for i, x_batch in enumerate(data_loader_valid):
        x_image = x_batch['image']
        x_image = x_image.to(device, dtype=torch.float)
        labels = x_batch["labels"]
        labels = labels.to(device, dtype=torch.float)

        with torch.no_grad():
            pred = model(x_image)
            loss = criterion(pred, labels)

            new1 = torch.sigmoid(pred).detach().cpu() >= 0.5

            tr_loss += loss.item()
            tr_correct += torch.sum(new1 == labels.cpu())

            acc.append(accuracy_score(labels.cpu(), new1.float()))
            f1.append(f1_score(labels.cpu(), new1.float(), average=None, zero_division=1))
            jacc.append(jaccard_score(labels.cpu(), new1.float(), average=None, zero_division=1))
            prec.append(precision_score(labels.cpu(), new1.float(), average=None, zero_division=1))
            rec.append(recall_score(labels.cpu(), new1.float(), average=None, zero_division=1))

    epoch_loss = tr_loss / len(data_loader_valid)
    
    print('Validation  Loss: {:.4f}'.format(epoch_loss))
    print('-----------------------')
    tb.add_scalar("Validation Loss", tr_loss, epoch)
    tb.add_scalar("Validation Accuracy", tr_correct / len(valid_dataset), epoch)
        
    ans = sum(acc) / len(acc)
    ans1 = sum(f1) / len(f1)
    ans2 = sum(rec) / len(rec)
    ans3 = sum(prec) / len(prec)
    ans4 = sum(jacc) / len(jacc)

    print('Accuracy:', ans)
    print('F1:', ans1)
    print("Recall:", ans2)
    print("Precision:", ans3)
    print("Jaccard score:", ans4)

    # Decay learning rate
    # scheduler.step()

# Save model
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
torch.save(checkpoint, '/workspace/ichd/models/yourmodel.pt')
tb.close()
