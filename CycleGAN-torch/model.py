import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torchvision.transforms as transforms

import numpy as np
import warnings
from data import get_data_loader, scale
from loss import real_mse_loss,fake_mse_loss,cycle_consistency_loss
import torch.optim as optim
from nn import create_model
from helper import checkpoint,save_samples
from tensorboardX import SummaryWriter
from utils import write_summary

        

def train_epoch(epoch_index, model_list, dataloader_list, optimizer_list,writer,print_every=10):
    G_XtoY,G_YtoX,D_X,D_Y = model_list
    # model_list = [G_XtoY,G_YtoX,D_X,D_Y]
    # opti_list = [g_optimizer,d_x_optimizer,d_y_optimizer]
    # data_loder_list = [train_X_loader,test_X_loader,train_Y_loader,test_Y_loader]
    dataloader_X, test_dataloader_X,dataloader_Y,test_dataloader_Y = dataloader_list
    g_opti,d_x_optimizer,d_y_optimizer = optimizer_list
    iter_index = 0
    epoch_batches = min(len(dataloader_X),len(dataloader_Y))
    print("X length:\t",len(dataloader_X))
    print("Y length:\t",len(dataloader_Y))
    d_x_epoch_loss = 0
    d_y_epoch_loss = 0
    g_epoch_loss = 0
    print_d_x = 0
    print_d_y = 0
    print_g = 0
    for batch_X, batch_Y in zip(dataloader_X, dataloader_Y):
        images_X,_ = batch_X
        images_X = scale(images_X)
        images_Y,_ = batch_Y
        images_Y = scale(images_Y)
        if torch.cuda.is_available():
            images_X = images_X.cuda()
            images_Y = images_Y.cuda()
        iter_index += 1
        iter_step = epoch_index * epoch_batches + iter_index
        loss_dict = train_step(model_list,opti_list,images_X,images_Y)
        print_d_x += loss_dict["d_x_loss"]
        print_d_y += loss_dict["d_y_loss"]
        print_g += loss_dict["g_loss"]
        d_x_epoch_loss += loss_dict["d_x_loss"]
        d_y_epoch_loss += loss_dict["d_y_loss"]
        g_epoch_loss += loss_dict["g_loss"]
        if iter_index%print_every==0:
            print("[epoch:{} iter {}/{}]\td_x_loss:{}\td_y_loss:{}\tg_loss:{}".format(epoch_index,
                                                                    iter_index,epoch_batches,
                                                                    print_d_x/print_every,
                                                                    print_d_y/print_every,
                                                                    print_g/print_every))
            write_summary(loss_dict,iter_step,writer)
            print_d_x = 0
            print_d_y = 0
            print_g = 0
    return d_x_epoch_loss/epoch_batches, d_y_epoch_loss/epoch_batches,g_epoch_loss/epoch_batches

def train_step(model_list,opti_list,train_X_batch,train_Y_batch):

    G_XtoY,G_YtoX,D_X,D_Y = model_list
    g_opti,d_x_optimizer,d_y_optimizer = opti_list

    # Discriminator
    d_x_optimizer.zero_grad()
    d_y_optimizer.zero_grad()
    fake_Y = G_XtoY(train_X_batch)
    fake_X = G_YtoX(train_Y_batch)
    real_X_output = D_X(train_X_batch)
    fake_X_output = D_X(fake_X)
    real_Y_output = D_Y(train_Y_batch)
    fake_Y_output = D_Y(fake_Y)
    real_X_loss = real_mse_loss(real_X_output)
    real_Y_loss = real_mse_loss(real_Y_output)
    fake_X_loss = fake_mse_loss(fake_X_output)
    fake_Y_loss = fake_mse_loss(fake_Y_output)
    d_x_loss = real_X_loss + fake_X_loss
    d_y_loss = real_Y_loss + fake_Y_loss
    d_x_loss.backward()
    d_x_optimizer.step()
    d_y_loss.backward()
    d_y_optimizer.step()

    #Generator
    g_opti.zero_grad()
    fake_X = G_YtoX(train_Y_batch)
    fake_Y = G_XtoY(train_X_batch)
    fake_X_output = D_X(fake_X)
    fake_Y_output = D_Y(fake_Y)
    x_ad_loss = real_mse_loss(fake_X_output)
    y_ad_loss = real_mse_loss(fake_Y_output)
    cycle_X = G_YtoX(fake_Y)
    cycle_Y = G_XtoY(fake_X)
    x_cycle_loss = cycle_consistency_loss(train_X_batch,cycle_X,10)
    y_cycle_loss = cycle_consistency_loss(train_Y_batch,cycle_Y,10)
    g_loss = x_ad_loss+y_ad_loss+x_cycle_loss+y_cycle_loss
    g_loss.backward()
    g_opti.step()
    loss_dict = {
        "d_x_real":real_X_loss,
        "d_y_real":real_Y_loss,
        "d_x_fake":fake_X_loss,
        "d_y_fake":fake_Y_loss,
        "d_x_loss":d_x_loss,
        "d_y_loss":d_y_loss,
        "g_x_ad_loss":x_ad_loss,
        "g_y_ad_loss":y_ad_loss,
        "g_x_cycle_loss":x_cycle_loss,
        "g_y_cycle_loss":y_cycle_loss,
        "g_loss":g_loss
    }
    return loss_dict

def train_loop(num_epochs,model_list,dataloader_list,optimizer_list,save_dir,log_dir,sample_dir,sample_every=10,print_every=10):
    
    G_XtoY,G_YtoX,D_X,D_Y = model_list
    test = torch.zeros(1,3,128,128)
    if torch.cuda.is_available():
        test = test.cuda()
    print("save dir:\t{}\tlogdir:{}".format(save_dir,log_dir))
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_graph(G_XtoY,test)
    writer.add_graph(G_YtoX,test)
    writer.add_graph(D_X,test)
    writer.add_graph(D_Y,test)
    for epoch_index in range(1,num_epochs+1):

        d_x_loss,d_y_loss,g_loss = train_epoch(epoch_index, model_list, dataloader_list, optimizer_list,writer,print_every)
        print("epoch:[{}/{}]\td_x_loss:{}\td_y_loss:{}\tg_loss:{}".format(epoch_index, 
                                                                            num_epochs,
                                                                            d_x_loss,
                                                                            d_y_loss,
                                                                            g_loss))
        
        if epoch_index%sample_every==0:
            checkpoint(epoch_index,*model_list,save_dir)
            G_XtoY.eval()
            G_YtoX.eval()
            iter_test_X,iter_test_Y = iter(dataloader_list[1]),iter(dataloader_list[3])
            save_samples(epoch_index,scale(iter_test_Y.next()[0]),scale(iter_test_X.next()[0]),G_YtoX,G_XtoY,16,sample_dir)
            # save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY)
            G_XtoY.train()
            G_YtoX.train()




if __name__ == "__main__":
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    G_XtoY,G_YtoX,D_X,D_Y = create_model()
    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())
    g_optimizer = optim.Adam(g_params,lr,[beta1, beta2])
    d_x_optimizer = optim.Adam(D_X.parameters(),lr,[beta1,beta2])
    d_y_optimizer = optim.Adam(D_Y.parameters(),lr,[beta1, beta2])
    train_X_loader,test_X_loader = get_data_loader("summer")
    train_Y_loader,test_Y_loader = get_data_loader("winter")
    model_list = [G_XtoY,G_YtoX,D_X,D_Y]
    opti_list = [g_optimizer,d_x_optimizer,d_y_optimizer]
    data_loder_list = [train_X_loader,test_X_loader,train_Y_loader,test_Y_loader]
    # train_epoch(0, train_X_loader,train_Y_loader,test_X_loader,test_Y_loader)
    train_loop(400,model_list,data_loder_list,opti_list,r"E:\torch学习\model\CycleGan",r"E:\torch学习\logs\CycleGan",r"E:\torch学习\samples\CycleGan")