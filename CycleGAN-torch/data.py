import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
# "/data/torch/CycleGan/summer2winter_yosemite"
def get_data_loader(image_type,image_dir=r"E:\torch学习\data\summer2winter-yosemite\summer2winter_yosemite",
                    image_size=128,batch_size=16,num_workers=0):
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    image_path = image_dir
    train_path = os.path.join(image_path,image_type)
    test_path = os.path.join(image_path,"test_{}".format(image_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)
    # print(iter(train_dataset).next())
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    return train_loader, test_loader

def scale(x, feature_range=(-1,1)):
    min,max = feature_range
    x = x * (max-min) + min
    return x

