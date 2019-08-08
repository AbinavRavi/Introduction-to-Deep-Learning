"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        model_pre=torchvision.models.vgg16_bn(pretrained=True)
        model_pre=model_pre.features
        self.relu = nn.ReLU(inplace=True)
        self.pretrained=model_pre
        
        self.deconv1=nn.ConvTranspose2d(512,512,kernel_size=3, stride=2,padding=0)#2
        self.bn1 = nn.BatchNorm2d(512)#2
        self.deconv2=nn.ConvTranspose2d(512, 256,kernel_size=3, stride=2,padding=1,output_padding=1)#2
        self.bn2 = nn.BatchNorm2d(256)#2
        self.deconv3=nn.ConvTranspose2d(256, 128,kernel_size=3, stride=2,padding=1,output_padding=1)#2
        self.bn3 = nn.BatchNorm2d(128)#2
        self.deconv4=nn.ConvTranspose2d(128, 64 ,kernel_size=3, stride=2,padding=1,output_padding=1)#2
        self.bn4 = nn.BatchNorm2d(64)#2
        self.deconv5=nn.ConvTranspose2d(64, 64 ,kernel_size=3, stride=2,padding=1,output_padding=1)#2
        self.bn5 = nn.BatchNorm2d(64)#2
        self.conv1=nn.Conv2d(64,num_classes,kernel_size=1)#2
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        for i in range(0,24):#2
            x=self.pretrained[i](x)#2
        x1=x#2
        for i in range(24,34):#2
            m=self.pretrained[i]#2
            x=m(x)#2
        x2=x#2
        for i in range(34,44):#2
            x=self.pretrained[i](x)#2
         
        #transpose
        x=self.deconv1(x)#2
        x=self.bn1(self.relu(x)+x2)#2
        x=self.deconv2(x)#2
        x=self.bn2(self.relu(x)+x1)#2
        x=self.deconv3(x)#2
        x=self.bn3(self.relu(x))#2
        x=self.deconv4(x)#2
        x=self.bn4(self.relu(x))#2
        x=self.deconv5(x)#2
        x=self.bn5(self.relu(x))#2
        x=self.conv1(x)#2
        torch.nn.Upsample((240,240), mode='nearest')#2
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
