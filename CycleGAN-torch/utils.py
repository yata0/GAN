import matplotlib.pyplot as plt
import torchvision
from data import get_data_loader
import numpy as np
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))

def datashow(dataloader):
    dataiter = iter(dataloader)
    images,_ = dataiter.next()
    fig = plt.figure(figsize=(12,8))
    imshow(torchvision.utils.make_grid(images))
    plt.show()

def write_summary(loss_dict,iter_index, writer):
    for name in loss_dict:
        writer.add_scalar("loss/{}".format(name),loss_dict[name],iter_index)
    
if __name__ == "__main__":
    train_data_x ,test_data_x = get_data_loader("summer")
    train_data_y, test_data_y = get_data_loader("winter")
    datashow(train_data_x)
    datashow(train_data_y)