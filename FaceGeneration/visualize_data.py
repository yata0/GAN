import matplotlib.pyplot as plt
from FaceDataset import get_dataloader
import pickle as pkl
import numpy as np

def imshow(img,sample=False):
    
    if sample:
        img = (img + 1) /2
        npimg = img.detach().cpu().numpy()
    else:
        npimg = img.numpy()
    # print(npimg.max())
    plt.imshow(np.transpose(npimg,(1,2,0)))

def batch_show(batch_data,sample,target_file):
    fig = plt.figure(figsize=(20,4))
    plot_size = 20
    for idx in np.arange(plot_size):
        ax = fig.add_subplot(2, plot_size/2, idx+1,xticks=[],yticks=[])
        imshow(batch_data[idx],sample)
    # plt.show()
    # plt.gcf()
    # plt.draw()
    plt.savefig(target_file)

if __name__ == "__main__":
    dataloader = get_dataloader(30,32)
    dataset = iter(dataloader)
    batch_data, _ = dataset.next()
    batch_show(batch_data)