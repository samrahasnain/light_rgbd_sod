import torch
import torch.nn as nn
import torch.nn.functional as F
from MobileNetV2 import mobilenet_v2
import numpy as np
class FeatureExtractionModule(nn.Module):
    def __init__(self, backbone):
        super(FeatureExtractionModule, self).__init__()
        self.backbone = backbone

    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        pretrained_dict = {ka: va for ka, va in pretrained_dict.items() if ka in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        
    def forward(self, x,y):
        conv1r, conv2r, conv3r, conv4r, conv5r = self.backbone(x)
        conv1d, conv2d, conv3d, conv4d, conv5d = self.backbone(y)        
        print("Backbone Features shape")
        print("RGB1: ",conv1r.shape,"    Depth1: ",conv1d.shape)
        print("RGB2: ",conv2r.shape,"    Depth2: ",conv2d.shape)
        print("RGB3: ",conv3r.shape,"    Depth3: ",conv3d.shape)
        print("RGB4: ",conv4r.shape,"    Depth4: ",conv4d.shape)
        print("RGB5: ",conv5r.shape,"    Depth5: ",conv5d.shape)
        

        return conv,tran,q,k,v,x_att # list of tensor that compress model output


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
    
    def forward(self, x,y):
        sal_final = x + y
        return sal_final


class General(nn.Module):
    def __init__(self,FeatureExtractionModule,Decoder):
        super(General, self).__init__()
        self.FeatureExtractionModule = FeatureExtractionModule
        self.decoder = Decoder
       
    def forward(self,rgb,depth):
        x,y = self.FeatureExtractionModule(rgb,depth)
        sal_final = self.decoder(x,y)
        return sal_final
      
def build_model(network='mobilenet', base_model_cfg='mobilenet'):
   
        backbone = mobilenet_v2()
        
        return General(FeatureExtractionModule(backbone), Decoder())
