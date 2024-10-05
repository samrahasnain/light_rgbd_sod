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
        

        return conv1r, conv2r, conv3r, conv4r, conv5r, conv1d, conv2d, conv3d, conv4d, conv5d # list of tensor that compress model output


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1,dilation=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin,dilation=dilation)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.bn = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.relu(self.bn(self.pointwise(out)))
        return out
        
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        
        # Create the list of fully connected layers (input -> hidden -> output)
        layers = []
        in_features = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())  # Activation function after each layer
            in_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_features, output_size))
        
        # Define the model as a Sequential block of layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # Ensure kernel_size is odd for padding consistency
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        # Convolutional layer for the spatial attention map
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply average pooling and max pooling along the channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, height, width)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (batch, 1, height, width)
        
        # Concatenate the average and max pooled features along the channel dimension
        concat = torch.cat([avg_out, max_out], dim=1)  # (batch, 2, height, width)
        
        # Pass the concatenated result through a convolution and sigmoid to generate attention map
        attention = self.conv(concat)
        attention = self.sigmoid(attention)  # (batch, 1, height, width)
        
        # Multiply input with attention map to get refined output
        out = x * attention
        return out


# Custom module with multiple depthwise separable conv layers
class LKA(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super(LKA, self).__init__()

       
        self.dsconv =  depthwise_separable_conv(in_channels_list, out_channels_list, kernel_size=3, padding=1)
            
        self.dsconv_dilated = depthwise_separable_conv(out_channels_list, out_channels_list, kernel_size=3, padding=3, dilation=3)
            
        self.conv1x1 = nn.Conv2d(out_channels_list, out_channels_list, kernel_size=1)
           

    def forward(self, x_list):

            x = self.dsconv(x_list)
           
            x = self.dsconv_dilated(x)
           
            x = self.conv1x1(x)
            
            return x



class levelEnhancedModule(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super(levelEnhancedModule, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        # Define Spatial Attention for each level
        self.sa = nn.ModuleList([SpatialAttention(3) for _ in range(5)])
        
        # Define depthwise separable convolution for each level with varying in/out channels
        self.dsconv = nn.ModuleList([
            depthwise_separable_conv(in_channels_list[i], out_channels_list[i], 3, 1,1) 
            for i in range(5)
        ])

        # Define LKA for each level with varying in/out channels
        self.LKA= nn.ModuleList([
            LKA(in_channels_list[i], out_channels_list[i]) 
            for i in range(5)
        ])
        hidden_sizes = [in_channels * 4 for in_channels in in_channels_list]
        # Define MLP for each level with varying in/out channels
        self.MLP= nn.ModuleList([
            MLP(in_channels_list[i], hidden_sizes,out_channels_list[i]) 
            for i in range(5)
        ])
    

    def forward(self, F1r, F2r, F3r, F4r, F5r, F1d, F2d, F3d, F4d, F5d):
        F_r = [F1r, F2r, F3r, F4r, F5r]
        F_d = [F1d, F2d, F3d, F4d, F5d]

        F_Rme = []
        F_Dme = []
        F_F   = []
        F_Fd  = []
        for i in range(5):
            # Level i+1 RGB modality enhanced features FiRme
            F_rle = F_r[i]
            F_dle = F_d[i]
            
            # Apply spatial attention on F_rle
            F_rle_sa = self.sa[i](F_rle)* F_dle
            
            # Concatenate the original and spatially attended features
            F_rle_cat_sa = torch.cat((F_rle, F_rle_sa), dim=1)
            print(F_rle_cat_sa.shape, F_dle.shape)
            # Element-wise multiplication of concatenated RGB and depth modality enhanced features
            F_rdle = F_rle_cat_sa 
            
            # Apply depthwise separable convolution for the current level
            F_rme = self.dsconv[i](F_rdle)
            # ENHANCEMNET ON D MODALITY Apply spatial attention on F_dle
            F_dle_sa = self.sa[i](F_dle)* F_rle
            
            # Concatenate the original and spatially attended features
            F_dle_cat_sa = torch.cat((F_dle, F_dle_sa), dim=1)
            
            # Element-wise multiplication of concatenated RGB and depth modality enhanced features
            F_drle = F_dle_cat_sa 
            
            # Apply depthwise separable convolution for the current level
            F_dme = self.dsconv[i](F_drle)
            F_f = self.dsconv[i](F_rme+F_dme)
            F_F.append(F_f)
            # Append results for each level
            F_Rme.append(F_rme)
            F_Dme.append(F_dme)
        for i in range(5,-1,-1):
            # Level i+1 RGB modality enhanced features FiRme
            if i==5:
                F_Fi = F_F[i]
                VAB = self.MLP[i](self.LKA[i](F_Fi)*F_Fi)
                
            else:
                F_Fi = F_F[i]
                F_Fe = F_Fi + self.dsconv[i](torch.cat((F_Fi,self.upsample(F_Fd[i+1])),dim=1))
                VAB = self.MLP[i](self.LKA[i](self.dsconv[i](torch.cat((F_Fe,self.upsample(F_Fd[i+1])),dim=1))))

            F_Fd.insert(0,VAB)
                
        return F_Fd

class Decoder(nn.Module):
    def __init__(self,in_channels_list, out_channels_list):
        super(Decoder,self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1x1 = nn.ModuleList([
            nn.Conv2d(out_channels_list[i], out_channels_list[i], kernel_size=1)
            for i in range(len(out_channels_list))
        ])
        
    def forward(self,F_Fd, F1r, F2r, F3r, F4r, F5r, F1d, F2d, F3d, F4d, F5d):
        F_r = [F1r, F2r, F3r, F4r, F5r]
        F_d = [F1d, F2d, F3d, F4d, F5d]
        F_Rle = []
        F_Dle = []
        for i in range(4):
            F_rle = F_r[i] + self.conv1x1(torch.cat((F_r[i],self.upsample(F_Fd[i+1])),dim=1))
            F_dle = F_d[i] + self.conv1x1(torch.cat((F_d[i],self.upsample(F_Fd[i+1])),dim=1))
            F_Rle.append(F_rle)
            F_Dle.append(F_dle)
        F_Rle.append(F5r)
        F_Dle.append(F5d)   
        return F_Rle, F_Dle

        



class General(nn.Module):
    def __init__(self,FeatureExtractionModule,levelEnhancedModule,Decoder):
        super(General, self).__init__()
        self.FeatureExtractionModule = FeatureExtractionModule
        self.levelEnhancedModule = levelEnhancedModule
        self.decoder = Decoder
      
       
    def forward(self,rgb,depth):
        F1r, F2r, F3r, F4r, F5r, F1d, F2d, F3d, F4d, F5d = self.FeatureExtractionModule(rgb,depth)
        F_Fd = self.levelEnhancedModule(F1r, F2r, F3r, F4r, F5r, F1d, F2d, F3d, F4d, F5d)
        F_Rle , F_Dle = self.decoder(F_Fd, F1r, F2r, F3r, F4r, F5r, F1d, F2d, F3d, F4d, F5d)
        
        return F_Fd[1],F_Fd[2],F_Fd[3],F_Fd[4],F_Fd[5],F_Rle[2] , F_Dle[2]
      
def build_model(network='mobilenet', base_model_cfg='mobilenet'):
   
        backbone = mobilenet_v2()
        in_channels_list = [16,24,32,96,320]
        out_channels_list = [16,24,32,96,320]
        return General(FeatureExtractionModule(backbone),levelEnhancedModule(in_channels_list,out_channels_list),Decoder)
