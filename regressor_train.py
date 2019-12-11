#%%
from __future__ import print_function
from __future__ import division
from PIL import Image, ImageDraw ####
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import cv2
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from efficientnet_pytorch import EfficientNet
import matplotlib.image as mpimg


def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


class trainset(Dataset):

    def __init__(self, data, target, transform):
        self.transform = transform
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        image = mpimg.imread("../data/black_mice/raw_set/"+img_name)
        image = Image.fromarray(image)
        label = torch.tensor(self.target[idx])
        image = self.transform(image)
        return image, label


class validset(Dataset):

    def __init__(self, data, target, transform):
        self.transform = transform
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        image = mpimg.imread("../data/black_mice/raw_set/"+img_name)
        image = Image.fromarray(image)
        label = torch.tensor(self.target[idx])
        image = self.transform(image)
        return image, label

########################################################################################
class Neuralnet(nn.Module):
    
    def __init__(self, model_name, num_classes, feature_extract=True):
        super(Neuralnet, self).__init__()
        self.model = None
        if model_name == 'resnet':
        ##Resnet18
            self.model = models.resnet18(pretrained=True)
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            print("Invalid model name, exiting...")
            exit()

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        output = self.model(x)
        return output

#%%

#Necessary Params
data = np.load('./train_IN.npy',allow_pickle=True)
target = np.load('./train_IN_tar.npy',allow_pickle=True)

model_name = 'resnet'
criterion = nn.CrossEntropyLoss()####%
if _name__ == '__main__':
main()###########################################

input_size = 224
is_inception = False
feature_extract = True

num_classes = 2
num_epochs = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 100

#%%Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#%%Model and it's dict
model = Neuralnet(model_name,num_classes)
model_ft = Neuralnet(model_name,num_classes)

saved_pretrain_model = torch.save(model_ft.state_dict(), './saved_pretrain_model')
model = model.to(device)

#Opt
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer = optim.Adam(params_to_update, lr=0.0001, betas=(0.9, 0.999), eps=1e-07, weight_decay=0, amsgrad=False)

#%% 
## K-fold can be implement. To train a model, I just run one iter at all and output the model.
## valid dataset is spilt by Kfold function
torch.cuda.empty_cache()

KFold(n_splits=2, random_state=123, shuffle=True)
kf = KFold(n_splits=10)
kf.get_n_splits(data)

val_score_list = []

for train_index, test_index in kf.split(data):
    model.load_state_dict(torch.load('./saved_pretrain_model'))
    train_data, valid_data = data[train_index], data[test_index]
    train_target, valid_target = target[train_index], target[test_index]
    train_dataset = trainset(train_data,train_target,train_transform)
    valid_dataset = validset(valid_data,valid_target,test_transform)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    dataloaders = {'train':train_data_loader, 'val':valid_data_loader}

    model, val_acc_history = train_model(model, dataloaders, criterion, optimizer, num_epochs, device, is_inception)
    val_score = np.array(val_acc_history).mean()
    val_score_list.append(val_score)

print("val_score_list: ", val_score_list)
print("val_score_mean: ", np.array(val_score_list).mean())
print("val_score_std: ",np.array(val_score_list).std())
