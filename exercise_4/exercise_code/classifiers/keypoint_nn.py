import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, bias=True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout_1 = nn.Dropout(p=0.1)
        
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, bias=True)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout_2 = nn.Dropout(p=0.2)
        
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, bias=True)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout_3 = nn.Dropout(p=0.3)
        
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, bias=True)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout_4 = nn.Dropout(p=0.4)
        
        self.dense_1 = nn.Linear(6400, 1000, bias=True)
        self.dropout_5 = nn.Dropout(p=0.5)
        
        self.dense_2 = nn.Linear(1000, 1000, bias=True)
        self.dropout_6 = nn.Dropout(p=0.6)
        self.dense_3 = nn.Linear(1000, 30)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        x = self.dropout_1(self.maxpool_1(F.relu(self.conv_1(x))))
        x = self.dropout_2(self.maxpool_2(F.relu(self.conv_2(x))))
        x = self.dropout_3(self.maxpool_3(F.relu(self.conv_3(x))))
        x = self.dropout_4(self.maxpool_4(F.relu(self.conv_4(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout_5(F.relu(self.dense_1(x)))
        x = self.dropout_6(F.relu(self.dense_2(x)))
        x = self.dense_3(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
