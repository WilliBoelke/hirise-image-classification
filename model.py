import torch.nn as nn
import torch.nn.functional as F

number_of_classes = 8
m_kernel_size = 5


class HiRiseModel(nn.Module):
    def __init__(self):
        super(HiRiseModel, self).__init__()
        # convolution layer
        self.convolution1 = nn.Conv2d(1, 5, kernel_size=m_kernel_size)
        self.convolution2 = nn.Conv2d(5, 10, kernel_size=m_kernel_size)
        self.conv_dropout = nn.Dropout2d()
        self.fully_con_linear_layer1 = nn.Linear(1440, 1440)
        self.fully_con_linear_layer2 = nn.Linear(1440, 1440)
        self.fully_con_linear_layer3 = nn.Linear(1440, 1440)
        self.fully_con_linear_layer4 = nn.Linear(1440, number_of_classes)

    def forward(self, x):
        x = self.convolution1(x)
        x = F.max_pool2d(x, 4)  # pooling layer
        x = F.relu(x)
        x = self.convolution2(x)
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 4)
        #print(x.shape)
       # exit()
        x = F.relu(x)
        x = x.view(-1, 1440)
        x = F.relu(self.fully_con_linear_layer1(x))
        x = F.relu(self.fully_con_linear_layer2(x))
        x = F.relu(self.fully_con_linear_layer3(x))
        x = self.fully_con_linear_layer4(x)
        return F.log_softmax(x, dim=1)
