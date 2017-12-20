import types
import torch
import torch.nn as nn
from resnet_mod import resnet50, resnet18, resnet101
from util import show_memusage
import torch.nn.functional as F

class mvcnn(nn.Module):
    def __init__(self, num_classes=17, pretrained=True):
        """
        Constructs a multi-view CNN where views are each fed to the same CNN and the feature maps are merged with a LSTM.
        Incorporates a form of pyramid pooling, LSTM attention, and CNN attention. 
        Args:
            num_classes (int): Number of output nodes for the problem.
            pretrained (bool): Whether or not to use a pretrained model for cnn. Recommend True.
        """ 
        super(mvcnn, self).__init__()

        # Define initial pretrained CNN each view goes through
        self.cnn = resnet50(pretrained=pretrained)

        # 3 convolutional layers of differing kernel size for locating threats of different size.
        self.conv1 = nn.Sequential(
                        nn.Conv2d(2048, 512, kernel_size=(1,1), stride=1, padding=0, dilation=1, groups=1, bias=True),
                        nn.ReLU(),
                        nn.BatchNorm2d(512)
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(2048, 256, kernel_size=(3,3), stride=2, padding=1, dilation=1, groups=1, bias=True),
                        nn.ReLU(),
                        nn.BatchNorm2d(256)
                    )
        self.conv3 = nn.Sequential(
                        nn.Conv2d(2048, 128, kernel_size=(5,5), stride=3, padding=2, dilation=1, groups=1, bias=True),
                        nn.ReLU(),
                        nn.BatchNorm2d(128)
                    )

        self.avgpool1 = nn.AdaptiveAvgPool2d(1)

        # Feed each view to LSTM with attention.
        self.lstm = nn.LSTM(input_size=2048 + 128*4*3 + 256*5*4 + 512*10*8, hidden_size=768, num_layers=1,
                            bias=True, batch_first=False, dropout=0, bidirectional=False)

        # Attention weights for LSTM.
        self.attention = nn.Linear(768, 16)
        self.softmax = nn.Softmax()

        # Experimental CNN attention layer (accidentally left this in when I made my final submissions...
        # haven't been able to tell if it hurts or helps.)
        self.cnn_attention = nn.Linear(2048, 2048)

        # LSTM results are fed to a final linear layer.
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(768, num_classes)
        

    def forward(self, x):
        # Compute feature vector for each view
        outputs = []
        for i in range(x.size()[1]):
            view = x[:, i]
            features = self.cnn(view)

            # CNN attention
            avg_pool = self.avgpool1(features).view(features.size(0), -1)
            attention = self.cnn_attention(avg_pool).unsqueeze(2)

            features = torch.mul(features, attention.unsqueeze(3).expand_as(features))

            # Go through each threat detection layer.
            features = torch.cat((self.avgpool1(features).view(features.size(0), -1),
                                    self.conv1(features).view(features.size(0), -1), 
                                    self.conv2(features).view(features.size(0), -1),
                                    self.conv3(features).view(features.size(0), -1)), 1)
            outputs.append(features)

        # Feed results to LSTM.
        outputs = torch.stack(outputs, dim=0)
        outputs, _ = self.lstm(outputs)

        attn_weights = self.softmax(
            self.attention(outputs[-1])
        )

        # Apply attention to the outputs and combine into a single output.
        outputs = outputs.permute(1, 0, 2)
        attn_weights = torch.unsqueeze(attn_weights, 1)
        outputs = torch.bmm(attn_weights, outputs)

        # Feed to linear classifier.
        outputs = torch.squeeze(outputs, 1)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)

        return outputs