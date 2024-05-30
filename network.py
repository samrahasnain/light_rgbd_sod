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
        '''print("Backbone Features shape")
        print("RGB1: ",conv1r.shape,"    Depth1: ",conv1d.shape)
        print("RGB2: ",conv2r.shape,"    Depth2: ",conv2d.shape)
        print("RGB3: ",conv3r.shape,"    Depth3: ",conv3d.shape)
        print("RGB4: ",conv4r.shape,"    Depth4: ",conv4d.shape)
        print("RGB5: ",conv5r.shape,"    Depth5: ",conv5d.shape)'''
        

        return conv1r, conv2r, conv3r, conv4r, conv5r, conv1d, conv2d, conv3d, conv4d, conv5d # list of tensor that compress model output
import torch
import torch.nn as nn

class ECABlock(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y).expand_as(x)

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
        
class SaliencyAlignment(nn.Module):
    def __init__(self):
        super(SaliencyAlignment, self).__init__()
        self.eca = ECABlock(320)
        self.dwsc = depthwise_separable_conv(320,320,3,1)
        self.relu = nn.ReLU()
        self.conv_last = depthwise_separable_conv(320,1,3,1)
    
    def forward(self, x,y):
        x = self.dwsc(x)
        y = self.eca(y)
        Fxy = x + y
        Fxy_conv = self.relu(self.conv_last(Fxy))
        #print('saliency alignmnet: ', Fxy_conv.shape)
        return Fxy_conv

class InceptionModuleModified(nn.Module):
    def __init__(self):
        super(InceptionModuleModified, self).__init__()
        in_channels = 16
        self.relu = nn.ReLU(inplace=True)
        self.branch1x1 = nn.Sequential(
            depthwise_separable_conv(in_channels, int(in_channels/4), kernel_size=1, padding=0),self.relu
        )    

        self.branch3x3 = nn.Sequential(
            depthwise_separable_conv(in_channels, int(in_channels/2), kernel_size=1, padding=0),self.relu,
            depthwise_separable_conv(int(in_channels/2), int(in_channels/4), kernel_size=3, padding=1),self.relu
        )

        self.branch5x5 = nn.Sequential(
            depthwise_separable_conv(in_channels, int(in_channels/4), kernel_size=1, padding=0),self.relu,
            depthwise_separable_conv(int(in_channels/4), int(in_channels/4), kernel_size=5, padding=2),self.relu
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),self.relu,
            depthwise_separable_conv(in_channels, int(in_channels/4), kernel_size=1, padding=0),self.relu
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)
        
class FeatureAlignmnetModule(nn.Module):
    def __init__(self,in_channels):
        super(FeatureAlignmnetModule, self).__init__()
        k=16
        self.conv = nn.Sequential(
            depthwise_separable_conv(in_channels, k , kernel_size=1, padding=0), nn.ReLU()
        )    
        
    def forward(self, x):
        aligned_x = self.conv(x)
        #print('aligned_x: ',aligned_x.shape)
        return aligned_x


        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        k=16
        self.upsample = nn.ConvTranspose2d(k,k, kernel_size=4, stride=2 , padding=1) # 10x10 to 20x20
        self.last_conv = nn.Conv2d(k,1,1,1)
    
    def forward(self, F_rgbd5, F_r5, F_r4, F_r3, F_r2, F_r1,F_d5, F_d4, F_d3, F_d2, F_d1):
        out5= (F_rgbd5 * F_r5) + (F_rgbd5 * F_d5)
        up_out5 = self.upsample(out5)
        out4 = (up_out5 * F_r4) + (up_out5 * F_d4)
        up_out4 = self.upsample(out4)
        out3 = (up_out4 * F_r3) + (up_out4 * F_d3)
        up_out3 = self.upsample(out3)
        out2 = (up_out3 * F_r2) + (up_out3 * F_d2)
        up_out2 = self.upsample(out2)
        out1 = (up_out2 * F_r1) + (up_out2 * F_d1)
        up_out1 = self.upsample(out1)
        sal_final = self.last_conv(up_out1)
        return sal_final


class General(nn.Module):
    def __init__(self,FeatureExtractionModule, InceptionModuleModified, SaliencyAlignment,  Decoder):
        super(General, self).__init__()
        self.FeatureExtractionModule = FeatureExtractionModule
        self.FAM1 = FeatureAlignmnetModule(16)
        self.FAM2 = FeatureAlignmnetModule(24)
        self.FAM3 = FeatureAlignmnetModule(32)
        self.FAM4 = FeatureAlignmnetModule(96)
        self.FAM5 = FeatureAlignmnetModule(320)
        self.inceptionmodule = InceptionModuleModified
        self.saliencyalignment = SaliencyAlignment
        self.decoder = Decoder
        
       
    def forward(self,rgb,depth):
        conv1r, conv2r, conv3r, conv4r, conv5r, conv1d, conv2d, conv3d, conv4d, conv5d = self.FeatureExtractionModule(rgb,depth)
        sal_align = self.saliencyalignment(conv5r, conv5d)
        F_r5 = self.inceptionmodule(self.FAM5(conv5r))
        F_r4 = self.inceptionmodule(self.FAM4(conv4r))
        F_r3 = self.inceptionmodule(self.FAM3(conv3r))
        F_r2 = self.inceptionmodule(self.FAM2(conv2r))
        F_r1 = self.inceptionmodule(self.FAM1(conv1r))
        F_d5 = self.FAM5(conv5d)
        F_d4 = self.FAM4(conv4d)
        F_d3 = self.FAM3(conv3d)
        F_d2 = self.FAM2(conv2d)
        F_d1 = self.FAM1(conv1d)
        sal_final = self.decoder(sal_align, F_r5, F_r4, F_r3, F_r2, F_r1,F_d5, F_d4, F_d3, F_d2, F_d1)
        
        return sal_final, sal_align
      
def build_model(network='mobilenet', base_model_cfg='mobilenet'):
   
        backbone = mobilenet_v2()
        
        return General(FeatureExtractionModule(backbone), InceptionModuleModified(), SaliencyAlignment(),Decoder())
