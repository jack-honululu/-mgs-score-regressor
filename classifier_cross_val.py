#%%
from __future__ import print_function
from __future__ import division
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import pandas as pd
from radam import RAdam
from ranger import Ranger
#from efficientnet_pytorch import EfficientNet

## https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
## The function train_model is reference from pytorch tutorial
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
            
            t = time.time()
            # Iterate over data.

            for i, data in enumerate(dataloaders[phase]):
                inputs = data['image'].to(device, non_blocking=True)
                labels = data['label'].to(device, non_blocking=True)
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
            time_t = time.time() - t
            print('epoch complete in {:.0f}m {:.0f}s'.format(time_t // 60, time_t % 60))

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


class dataset(Dataset):

    def __init__(self, csv_file, index, transform):
        self.transform = transform
        self.data = pd.read_csv(csv_file)
        self.data = self.data.iloc[index]
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join("../data/black_mice/raw_set/",self.data.loc[idx,'index'])
        image = Image.open(img_name)
        label = torch.tensor(self.data.loc[idx,'pain'])
        image = self.transform(image)

        return {'image': image, 'label': label}


class test_dataset(Dataset):

    def __init__(self, csv_file, transform):
        self.transform = transform
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join("../data/black_mice/raw_set/",self.data.loc[idx,'index'])
        image = Image.open(img_name)
        label = torch.tensor(self.data.loc[idx,'pain'])
        image = self.transform(image)

        return {'image': image, 'label': label}


class Neuralnet(nn.Module):
    
    def __init__(self, model_name, num_classes, feature_extract=True):
        super(Neuralnet, self).__init__()
        self.model = None
        #%%Resnet152
        if model_name == 'resnet152':
            self.model = models.resnet152(pretrained=True)
            self.set_parameter_requires_grad(self.model, feature_extract)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        ##Resnet18
        elif model_name == 'resnet':
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
def main():
    experiment = 'KXN_classifier'
    csv_file = './{}_tar.csv'.format(experiment)
    model_name = 'resnet152'
    optim = 'ranger'
    criterion = nn.CrossEntropyLoss()
    input_size = 224
    is_inception = False
    feature_extract = True
    batch_size = 128
    num_classes = 2
    num_epochs = 50
    learning_rate = 0.001
    random_state = 123
    train_a_model = True # False for cross validation

    title = "{}_{}_{}_{}_{}_{}".format(experiment,model_name,
                                    batch_size,learning_rate,
                                    num_epochs,optim)
    print(title)
    #%%Model and it's dict
    model = Neuralnet(model_name,num_classes)
   # torch.save(model.state_dict(), './saved_pretrain_model')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    if next(model.parameters()).is_cuda == False:
        num_workers = 0
    else:
        num_workers = 4
    print(device)
    print(num_workers)
    #Opt
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    # Observe that all parameters are being optimized
    if optim == 'RAdam':
        optimizer = RAdam(params_to_update, lr=learning_rate)
    elif optim == 'Adam':
        optimizer = optim.Adam(params_to_update, lr=learning_rate, betas=(0.9, 0.999), eps=1e-07, weight_decay=0, amsgrad=False)
    elif optim == 'ranger':
        optimizer = Ranger(params_to_update, lr=learning_rate)
    else:
        raise ImportError
    ## K-fold can be implement. To train a model, I just run one iter at all and output the model.
    ## valid dataset is spilt by Kfold function
        #%%Transforms
    train_transform = transforms.Compose([
#        transforms.Resize(256),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data = pd.read_csv(csv_file)
    kf = StratifiedKFold(n_splits=9, random_state=random_state, shuffle=True)

    val_score_list = []
    
    for train_index, valid_index in kf.split(data, data['pain']):
        if train_a_model == False:
            model.load_state_dict(torch.load('./saved_pretrain_model'))
        train_dataset = dataset(csv_file,train_index,train_transform)
        valid_dataset = dataset(csv_file,valid_index,test_transform)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        dataloaders = {'train':train_data_loader, 'val':valid_data_loader}

        model, val_acc_history = train_model(model, dataloaders, criterion, optimizer, num_epochs, device, is_inception)

        if train_a_model == True:
            torch.save(model.state_dict(), './model_'+title)
            np.save('val_acc_history_'+title+'.npy', val_acc_history)
            break
        else:
            val_score = np.array(val_acc_history).max()
            val_score_list.append(val_score)

    if train_a_model == False:
        print("val_score_list: ", val_score_list)
        print("val_score_mean: ", np.array(val_score_list).mean())
        print("val_score_std: ",np.array(val_score_list).std())
        np.save( "val_score_list.npy", np.array(val_score_list))    

#%%
if __name__ == '__main__':
    main()
