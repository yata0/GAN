from large_model import generator, discriminator
from FaceDataset import get_dataloader,scale
from utils import mkdirs
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from loss import real_loss, fake_loss,WGAN_fake_loss,WGAN_real_loss,WGAN_gp
from utils import write_summary
from tensorboardX import SummaryWriter
import pickle as pkl
from visualize_data import batch_show
import os
import pdb

log_dir = r"E:\torch学习\logs\facegeneration\gp-large-model"
model_dir = r"E:\torch学习\model\facegeneration\gp-large-model"
samples_dir = r"E:\torch学习\samples\facegeneration\gp-large-model"

mkdirs([log_dir, model_dir, samples_dir])

def weights_init_norm(m,gain=0.02):
    classname = m.__class__.__name__
    if hasattr(classname, "weight") and (classname.find("Conv")!=-1 or classname.find("Linear")!=-1):
        init.normal_(m.weight.data, 0.0, gain)
        if hasattr(m,"bias") and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

def build_model():
    D = discriminator()
    G = generator()
    D.apply(weights_init_norm)
    G.apply(weights_init_norm)
    print(D)
    print(G)
    return D,G

def train(g_model,d_model,g_opimizer,d_optimizer,dataloader,epochs):
    writer= SummaryWriter(log_dir=log_dir)
    samples_random = np.random.uniform(-1,1,size=(20,100))
    fixed_z = torch.from_numpy(samples_random).float()
    if torch.cuda.is_available():
        fixed_z = fixed_z.cuda()
    samples = []
    for index in range(epochs):
        
        train_epoch(g_model,d_model,g_opimizer,d_optimizer,dataloader, writer,index,print_every=20)
        if index % 5==0:
            # 
            torch.save(g_model.state_dict(),os.path.join(model_dir,"epoch_{}.pkl".format(index)))
            G.eval()
            samples_z = G(fixed_z)
            samples.append(samples_z)
            G.train()
            batch_show(samples_z,True,os.path.join(samples_dir,"epoch_{}.png".format(index)))
    with open('train_gp_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)
    torch.save(g_model.state_dict(),os.path.join(model_dir,"epoch_{}_discriminator.pkl".format(index)))

def train_epoch(g_model,d_model,g_opimizer,d_optimizer,dataloader, writer,epoch_index,print_every=20):
    use_gpu = torch.cuda.is_available()
    num_batches = len(dataloader)
    # pdb.set_trace()
    for iter_index, train_data in enumerate(dataloader):
        iter_index += 1
        total_iter_index = iter_index + num_batches * epoch_index
        train_data,_ = train_data
        train_data = scale(train_data)

        samples_random = np.random.uniform(-1,1,size=(train_data.size(0),100))
        z = torch.from_numpy(samples_random).float()
        if use_gpu:
            z = z.cuda()
            train_data = train_data.cuda()
        d_optimizer.zero_grad()
        d_real = D(train_data)

        fake_sample = g_model(z)
        d_fake = d_model(fake_sample)

        d_real_loss = WGAN_real_loss(d_real)
        d_fake_loss = WGAN_fake_loss(d_fake)
        gp_loss = WGAN_gp(train_data,fake_sample,d_model)
        d_loss = d_real_loss + d_fake_loss + 10 * gp_loss

        d_loss.backward()
        d_optimizer.step()

        #####generator#######
        g_optimizer.zero_grad()
        samples_random = np.random.uniform(-1,1,size=(train_data.size(0),100))
        z = torch.from_numpy(samples_random).float()
        if use_gpu:
            z = z.cuda()
        fake_sample = g_model(z)
        g_fake = d_model(fake_sample)
        g_loss = WGAN_real_loss(g_fake)
        g_loss.backward()
        g_optimizer.step()
        write_summary(writer,{"loss/d_real_loss":d_real_loss,
                        "loss/d_fake_loss":d_fake_loss,
                        "loss/d_loss":d_loss,
                        "loss/gp_loss":gp_loss,
                        "loss/g_loss":g_loss},total_iter_index)
        
        if iter_index%print_every==0:
            print("epoch:{}\titerations:[{}/{}]\td_loss:{}\tg_loss:{}".format(epoch_index, iter_index, num_batches,d_loss.item(),g_loss.item()))


if __name__ == "__main__":
    dataloader = get_dataloader(batch_size=32, image_size=64)
    D,G = build_model()
    d_optimizer = optim.Adam(D.parameters(),lr=0.00001,betas=(0.5,0.999))
    g_optimizer = optim.Adam(G.parameters(),lr=0.00001,betas=(0.5,0.999))
    if torch.cuda.is_available():
        D =D.cuda()
        G =G.cuda()
    train(G,D,g_optimizer,d_optimizer,dataloader,200)