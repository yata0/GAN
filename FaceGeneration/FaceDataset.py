import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
def get_dataloader(batch_size, image_size, data_dir = r"E:\torch学习\data\processed_celeba_small"):
    # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    transform = transforms.Compose([
 
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    #顺序很重要
    image_datasets = datasets.ImageFolder(data_dir, transform)
    image_loader = DataLoader(image_datasets, batch_size=batch_size, shuffle=True)
    return image_loader

def scale(x, feature_range=(-1,1)):
    """
    the input x with the range is (0, 1)
    """
    x = x * 2 - 1
    return x

if __name__ == "__main__":
    dataloader = get_dataloader(64, 32)
    data_iter = iter(dataloader)
    images = data_iter.next()
    img = images[0]
    scaled_img = scale(img)
    print("Min: ", scaled_img.min())
    print("Max: ", scaled_img.max())
