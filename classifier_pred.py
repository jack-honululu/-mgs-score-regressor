#%%
from __future__ import print_function
from __future__ import division
from PIL import Image, ImageDraw ####
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch
import classifier_cross_val
import numpy as np
import torchvision
from torchvision import models, transforms
from classifier_cross_val import test_dataset, Neuralnet
import pandas as pd
import matplotlib.image as mpimg

model_name = 'resnet152'
dict_path = '../model_KXN_classifier_resnet152_128_0.001_50_RAdam'
test_data = './KXN_classifier_test.csv'
device = torch.device("cpu")
test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
test_dataset = test_dataset(test_data, test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0, pin_memory=True)

#%%
model = Neuralnet(model_name,2)
model.load_state_dict(torch.load(dict_path,map_location=torch.device('cpu')))
model.eval()
#%%
##Confusion_matrix
## https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial
## the following code is reference to stackoverflow.com
confusion_matrix = torch.zeros(2, 2)
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data['image'].to(device, non_blocking=True)
        classes = data['label'].to(device, non_blocking=True)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

confusion_matrix = np.array(confusion_matrix)
confusion_matrix_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
pd.options.display.float_format = '{:,.3f}'.format
df = pd.DataFrame(confusion_matrix_norm,
             columns=['1', '0'],
             index=['1','0'])
df

#%%
#probabilities
prob = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data['image'].to(device)
        classes = data['label'].to(device)
        outputs = model(inputs)
        sm = torch.nn.Softmax()
        probabilities = sm(outputs) 
        prob.append(probabilities)

# %%
prob
#output[0] => label = 1
# %% val_acc_history_KXN_classifier_resnet152_128_0.001_50
val_acc=[0.8751,
         0.9025,
         0.9059,
         0.9089,
         0.9084,
         0.9118,
         0.9020,
         0.9143,
         0.9128,
         0.8599,
         0.9177,
         0.9138,
         0.9206,
         0.9035,
         0.9157,
         0.9231,
         0.9157,
         0.9216,
         0.9231,
         0.9231,
         0.9221,
         0.9138,
         0.9084,
         0.9250,
         0.9216,
         0.9250,
         0.9138,
         0.9265,
         0.9211,
         0.9211,
         0.9324,
         0.9216,
         0.9250,
         0.9343,
         0.9294,
         0.9339,
         0.9314,
         0.9324,
         0.9334,
         0.9324,
         0.9250,
         0.9319,
         0.9157,
         0.9324,
         0.9339,
         0.9339,
         0.9343,
         0.9373,
         0.9290,
         0.9363]

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

s=np.array(val_acc)
t=np.arange(len(val_acc))

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='epochs', ylabel='val_acc',
       title='KXN_classifier_resnet152_128_0.001_50_Adam')
ax.grid()

fig.savefig("test.png")
plt.show()

#%% KXN_classifier_resnet152_128_0.001_50_RAdam
val_acc=[0.8634,    
         0.8800,    
         0.8874,    
         0.8916,    
         0.8952,    
         0.8943,    
         0.8782,    
         0.8989,    
         0.9040,    
         0.8832,    
         0.9068,    
         0.9049,    
         0.9026,    
         0.9012,    
         0.9096,    
         0.8980,    
         0.9040,    
         0.9151,    
         0.9142,    
         0.9174,    
         0.9160,    
         0.9160,    
         0.9197,    
         0.9174,    
         0.9225,    
         0.9220,    
         0.9183,    
         0.9206,    
         0.9068,    
         0.9211,    
         0.9132,    
         0.9202,    
         0.9257,    
         0.9197,    
         0.9271,    
         0.9211,    
         0.9248,    
         0.9299,    
         0.9220,    
         0.9308,    
         0.9202,    
         0.9275,    
         0.9280,    
         0.9123,    
         0.9243,    
         0.9271,    
         0.9331,    
         0.9317,    
         0.9308,    
         0.9317]
s=np.array(val_acc)
t=np.arange(len(val_acc))

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='epochs', ylabel='val_acc',
       title='KXN_classifier_resnet152_128_0.001_50_RAdam')
ax.grid()

fig.savefig("test.png")
plt.show()

# %%
