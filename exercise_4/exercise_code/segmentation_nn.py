"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        self.num_classes = num_classes
        self.vgg_feat = models.vgg11(pretrained=True).features
        self.fcn = nn.Sequential(
                                nn.Conv2d(512, 1024, 7),
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                nn.Conv2d(1024, 2048, 1),
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                nn.Conv2d(2048, num_classes, 1)
                                )

        pass

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
        x_input = x
        x = self.vgg_feat(x)
        x = self.fcn(x)
        x = F.upsample(x, x_input.size()[2:], mode='bilinear', align_corners=True).contiguous()

        pass

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
