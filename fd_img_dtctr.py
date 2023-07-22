import requests
import zipfile
from pathlib import Path

data_path = Path('data')
image_path =data_path/'/content/DataX'

if image_path.is_dir():
  print(f"{image_path} directory exists")
else:
  print(f"Did not find {image_path} directory,creating one....")
  image_path.mkdir(parents=True,exist_ok=True)


  import zipfile

zip_path = '/content/DataX1.zip'

with zipfile.ZipFile(zip_path,'r') as zip_ref:
  zip_ref.extractall('/content')

  #@title Default title text
import os
def walk_through_dir(dir_path):
  for dirpath,dirnames,filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images {(dirpath)}")

walk_through_dir(image_path)

train_dir = image_path/'/content/DataX/Train'
test_dir = image_path/'/content/DataX/Test'
train_dir,test_dir

import torch
from torch import nn
import torchvision
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize(size=(300,300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize(size=(300,300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
from torchvision import datasets

train_data = datasets.ImageFolder(
    root=train_dir,
    transform=train_transforms
)

test_data = datasets.ImageFolder(
    root=test_dir,
    transform=test_transforms
)


from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    dataset = train_data,
    batch_size = 25,
    num_workers = os.cpu_count(),
    shuffle=True
)

test_dataloader = DataLoader(
    dataset = test_data,
    batch_size=25,
    num_workers = os.cpu_count(),
    shuffle=False
)


class_list = train_data.classes
class_list

class tinyvgg1(nn.Module):
  def __init__(self,hidden,input,output):
    super().__init__()
    self.Conv_block1 = nn.Sequential(
        nn.Conv2d(
            in_channels= input,
            out_channels=hidden,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Conv2d(
            in_channels = hidden,
            out_channels =hidden,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.Conv_block2 = nn.Sequential(
        nn.Conv2d(
            in_channels=hidden,
            out_channels=hidden,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Conv2d(
            in_channels=hidden,
            out_channels=hidden,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.Conv_block3 = nn.Sequential(
        nn.Conv2d(
            in_channels=hidden,
            out_channels=hidden,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Conv2d(
            in_channels=hidden,
            out_channels=hidden,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.Conv_block4 = nn.Sequential(
        nn.Conv2d(
            in_channels=hidden,
            out_channels=hidden,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Conv2d(
            in_channels=hidden,
            out_channels=hidden,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=1)
    )
    
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            in_features = hidden*36*36,
            out_features = output
        )
    )

  def forward(self,x):
    x = self.Conv_block1(x)
    #print(x.shape)
    x = self.Conv_block2(x)
    #print(x.shape)
    x = self.Conv_block3(x)
    #print(x.shape)
    x = self.Conv_block4(x)
    #print(x.shape)
    x = self.classifier(x)
    return x
    return  self.Conv_block4(self.Conv_block3(self.Conv_block2(self.Conv_block1(x))))
device = "cuda" if torch.cuda.is_available() else "cpu"
device
model_x = tinyvgg1(input=3,
                   hidden=45,
                   output=2
                   ).to(device)

def accuracy_fn(y_true,y_pred):
  correct = torch.eq(y_true,y_pred).sum().item()
  acc = correct/len(y_pred)*100
  return acc

losfn = nn.CrossEntropyLoss()
optimizer  =torch.optim.Adam(params = model_x.parameters(),
                             lr=0.0001)

import torch
from torch import nn

def train_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               optimizer:torch.optim.Optimizer,
               losfn:torch.nn.Module,
               accuracy_fn,
               device:device):
  train_acc,train_loss=0,0
  model.train()
  for batch,(X,y) in enumerate(dataloader):
    X,y = X.to(device),y.to(device)

    y_pred = model(X)
    loss = losfn(y_pred,y)
    train_loss += loss.item()
    train_acc += accuracy_fn(y_true=y,y_pred=y_pred.argmax(dim=1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  train_acc /= len(dataloader)
  train_loss /= len(dataloader)
  print(f"Epoch {epoch} | Train acc: {train_acc:.2f}% | Train Loss {train_loss:.4f}")

def test_step(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              losfn:torch.nn.Module,
              accuracy_fn,
              device:device
              ):
  test_acc,test_loss = 0,0
  model.eval()
  with torch.inference_mode():
    for batch,(X,y) in enumerate(dataloader):
      X,y = X.to(device),y.to(device)
      test_pred = model(X)
      loss1 = losfn(test_pred,y)
      test_loss += loss1.item()
      test_acc += accuracy_fn(y_true=y,y_pred=test_pred.argmax(dim=1))
    test_acc /= len(dataloader)
    test_loss /= len(dataloader)
    print(f"Test Acc {test_acc:.2f}% | Test Loss {test_loss:.4f}")
from tqdm.auto import tqdm

torch.manual_seed(42)

epochs = 30

from timeit import default_timer as timer

start = timer()

for epoch in tqdm(range(epochs)):
  train_step(model=model_x.to(device),
             dataloader=train_dataloader,
             optimizer=optimizer,
             losfn=losfn,
             accuracy_fn=accuracy_fn,
             device=device)
  test_step(model=model_x.to(device),
            dataloader=test_dataloader,
            losfn=losfn,
            accuracy_fn=accuracy_fn,
            device=device)
end = timer()

print(f"Total time {end-start}")

import requests

custom_image_path = data_path/ ""
if not custom_image_path.is_file():
  with open(custom_image_path,"wb") as f:
    request = requests.get("")
    print(f"Downloading {custom_image_path}.....")
    f.write(request.content)
else:
  print(f"{custom_image_path} already exists,skipping download")

  custom_image_path

import torch
import torchvision

custom_image = torchvision.io.read_image(str(custom_image_path)).to(torch.float32)/255
custom_image

from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize(size=(300,300))
])


custom_image_transformed = image_transform(custom_image)

import matplotlib.pyplot as plt

plt.imshow(custom_image_transformed.permute(1,2,0))


import torchvision
import matplotlib.pyplot as plt
from typing import Dict,List

def plot_pred_image(model:torch.nn.Module,
                    image_path:str,
                    class_names:List[str]=None,
                    transform=None,
                    device:torch.device=None):
  target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
  target_image = target_image/255

  if transform:
    target_image = transform(target_image)

    model.to(device)
    model.eval()
    with torch.inference_mode():
      target_image = target_image.unsqueeze(0)
      target_image_pred = model(target_image.to(device))
    target_image_pred_probs = torch.softmax(target_image_pred,dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs,dim=1)
    plt.imshow(target_image.squeeze().permute(1,2,0))
    if class_names:
               title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob {target_image_pred_probs.max():.3f}"
    else:
               title = f"{target_image_pred_label} | Prob {target_image_pred_probs,max():3f}"
    plt.title(title)
    plt.axis(False)

plot_pred_image(model=model_x,
                image_path=custom_image_path,
                class_names=class_list,
                transform=image_transform,
                device=device)