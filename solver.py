import torch
from torch.nn import functional as F
from network import build_model
import numpy as np
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
import torch.nn as nn
import argparse
import os.path as osp
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy.misc
from  utils import  count_model_flops,count_model_params
from PIL import Image
import json
size_coarse = (10,10)
class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        #self.build_model()
        self.net = build_model(self.config.network, self.config.arch)
        #self.net.eval()
        if config.mode == 'test':
            print('Loading pre-trained model for testing from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model, map_location=torch.device('cpu')))
        if config.mode == 'train':
            if self.config.load == '':
                print("Loading pre-trained imagenet weights for fine tuning")
                '''self.net.FeatureExtractionModule.load_pretrained_model(self.config.pretrained_model
                                                        if isinstance(self.config.pretrained_model, str)
                                                        else self.config.pretrained_model[self.config.network])'''
                # load pretrained backbone
            else:
                print('Loading pretrained model to resume training')
                self.net.load_state_dict(torch.load(self.config.load))  # load pretrained model
        
        '''if self.config.cuda:
            self.net = self.net.cuda()'''
       
        

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        #self.print_network(self.net, 'General Network Structure')

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params_t = 0
        num_params=0
        for p in model.parameters():
            if p.requires_grad:
                num_params_t += p.numel()
            else:
                num_params += p.numel()
        print(name)
        #print(model)
        print("The number of trainable parameters: {}".format(num_params_t))
        print("The number of parameters: {}".format(num_params))
        print(f'Flops: {count_model_flops(model)}')
        print(f'Flops: {count_model_params(model)}')
      
    def generate_intermediate_map(self,input_map):
        input_map = ((torch.sum(input_map,1)**2)/input_map.shape[1]).unsqueeze(0)
        input_map = F.interpolate(input_map, tuple(im_size), mode='bilinear', align_corners=True)
        input_map = np.squeeze(torch.sigmoid(input_map)).cpu().data.numpy()
        input_map = (input_map - input_map.min()) / (input_map.max() - input_map.min() + 1e-8)
        multi_fuse_input_map = 255 * input_map
        #multi_fuse_input_map=cv2.applyColorMap((multi_fuse_input_map).astype(np.uint8), cv2.COLORMAP_RAINBOW)
        filename_input_map = os.path.join(self.config.test_folder, name[:-4] + '_input_map.png')
        cv2.imwrite(filename_input_map, multi_fuse_input_map)
      
    def test(self):
        print('Testing...')
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']), \
                                           data_batch['depth']
            with torch.no_grad():
                '''if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)'''
                #input = torch.cat((images, depth), dim=0)
                #torch.cuda.synchronize()
                tsince = int(round(time.time()*1000)) 
                preds,_ = self.net(images,depth)
                #torch.cuda.synchronize()
                ttime_elapsed = int(round(time.time()*1000)) - tsince
                print ('test time elapsed {}ms'.format(ttime_elapsed))
                #generate_intermediate_map(in)
                preds = F.interpolate(preds, (320,320), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_folder, name[:-4] + '_general.png')
                cv2.imwrite(filename, multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')
    
  
    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        
        loss_vals=  []
        
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            r_sal_loss_item=0
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_depth, sal_label, sal_edge = data_batch['sal_image'], data_batch['sal_depth'], data_batch[
                    'sal_label'], data_batch['sal_edge']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image, sal_depth, sal_label, sal_edge= sal_image.to(device), sal_depth.to(device), sal_label.to(device),sal_edge.to(device)

               
                self.optimizer.zero_grad()
                sal_label_coarse = F.interpolate(sal_label, size_coarse, mode='bilinear', align_corners=True)
                
                sal_final, coarse_sal_align = self.net(sal_image,sal_depth)
                
                #sal_loss_coarse_rgb =  F.binary_cross_entropy_with_logits(coarse_sal_rgb, sal_label_coarse, reduction='sum')
                sal_loss_coarse =  F.binary_cross_entropy_with_logits(coarse_sal_align, sal_label_coarse, reduction='sum')
                sal_final_loss =  F.binary_cross_entropy_with_logits(sal_final, sal_label, reduction='sum')
                #edge_loss_rgbd0=F.smooth_l1_loss(sal_edge_rgbd0,sal_edge)
                #edge_loss_rgbd1=F.smooth_l1_loss(sal_edge_rgbd1,sal_edge)
                #edge_loss_rgbd2=F.smooth_l1_loss(sal_edge_rgbd2,sal_edge)
                
                sal_loss_fuse = sal_final_loss + sal_loss_coarse
                sal_loss = sal_loss_fuse/ (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data
                r_sal_loss_item+=sal_loss.item() * sal_image.size(0)
                sal_loss.backward()
                self.optimizer.step()

                if (i + 1) % (self.show_every // self.config.batch_size) == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %0.4f  ||sal_final:%0.4f' % (
                        epoch, self.config.epoch, i + 1, iter_num, r_sal_loss,sal_final_loss ))
                    # print('Learning rate: ' + str(self.lr))
                    writer.add_scalar('training loss', r_sal_loss / (self.show_every / self.iter_size),
                                      epoch * len(self.train_loader.dataset) + i)
                    '''writer.add_scalar('sal_loss_coarse_rgb training loss', sal_loss_coarse_rgb.data,
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar('sal_loss_coarse_depth training loss', sal_loss_coarse_depth.data,
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar('sal_edge training loss', edge_loss_rgbd2.data,
                                      epoch * len(self.train_loader.dataset) + i)

                    r_sal_loss = 0
                    res = coarse_sal_depth[0].clone()
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    writer.add_image('coarse_sal_depth', torch.tensor(res), i, dataformats='HW')
                    grid_image = make_grid(sal_label_coarse[0].clone().cpu().data, 1, normalize=True)

                    res = coarse_sal_rgb[0].clone()
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    writer.add_image('coarse_sal_rgb', torch.tensor(res), i, dataformats='HW')
                    grid_image = make_grid(sal_label_coarse[0].clone().cpu().data, 1, normalize=True)'''
                    
                    fsal = sal_final[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_final', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)

  


            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
            train_loss=r_sal_loss_item/len(self.train_loader.dataset)
            loss_vals.append(train_loss)
            
            print('Epoch:[%2d/%2d] | Train Loss : %.3f' % (epoch, self.config.epoch,train_loss))
            
        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)
        
