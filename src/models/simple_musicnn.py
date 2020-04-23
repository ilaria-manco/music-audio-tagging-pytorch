import torch
import torch.nn as nn
import torch.nn.functional as F

"""First implementation used for training of model 2020-04-21-13_58_40. """

class SimpleMusicnn(nn.Module):
    def __init__(self, level, y_input_dim, filter_type, k_height_factor, k_width_factor, filter_factor):
        super(SimpleMusicnn, self).__init__()

        self.level = level

        self.front_end = SimpleFrontEnd(
            y_input_dim, filter_type, k_height_factor, k_width_factor, filter_factor)

        if self.level == "front_end":
            self.dense1 = nn.Linear(in_features=36924, out_features=200)
        elif self.level == "mid_end":
            input_dim = (204, 1, 181)
            self.midend = SimpleMidEnd(input_dim)
            self.dense1 = nn.Linear(in_features=34752, out_features=200)

        self.dense2 = nn.Linear(in_features=200, out_features=50)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.front_end(x)

        if self.level == "mid_end":
            x = self.midend(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = self.sigmoid(self.dense2(x))
        return x


class SimpleFrontEnd(nn.Module):
    def __init__(self, y_input_dim, filter_type, k_height_factor, k_width_factor, filter_factor):
        super(SimpleFrontEnd, self).__init__()
        self.y_input_dim = y_input_dim
        self.k_height = k_height_factor
        self.k_width = k_width_factor
        self.filter_type = filter_type
        self.filter_factor = filter_factor

        # 1 convolutional layer
        if self.filter_type == "timbral":
            self.conv = self.timbral_block()
        if self.filter_type == "temporal":
            self.conv = self.temporal_block()

        nn.init.xavier_uniform_(self.conv.weight)
        self.batch_norm = nn.BatchNorm2d(num_features=self.out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(30, 1), stride=(30, 1))

    def timbral_block(self):
        k_h = int(self.y_input_dim * self.k_height)
        self.out_channels = int(self.filter_factor * 128)
        return nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(k_h, 7))

    def temporal_block(self):
        k_w = int(self.y_input_dim * self.k_width)
        self.out_channels = int(self.filter_factor * 32)
        return nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(1, k_w))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.batch_norm(x)
        x = self.pool(x)

        return x


class SimpleMidEnd(nn.Module):
    def __init__(self, input_dim, num_of_filters=64):
        super(SimpleMidEnd, self).__init__()
        self.input_channels, self.input_w, self.input_h = input_dim
        self.num_of_filters = num_of_filters

        # LAYER 1
        self.conv1 = nn.Conv1d(self.input_channels,
                               self.num_of_filters, kernel_size=7, padding=3)
        self.batch_norm1 = nn.BatchNorm1d(num_features=self.num_of_filters)
        # LAYER 2
        # TODO add residual connection
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv1d(self.num_of_filters,
                               self.num_of_filters, kernel_size=7, padding=3)
        self.batch_norm2 = nn.BatchNorm1d(num_features=self.num_of_filters)
        # LAYER 3
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv1d(self.num_of_filters,
                               self.num_of_filters, kernel_size=7, padding=3)
        self.batch_norm3 = nn.BatchNorm1d(num_features=self.num_of_filters)

    def forward(self, x):
        # TODO: add padding before each convolution
        # x = F.pad(x, p1d, 'constant', 0)
        x = x.view(x.shape[0], x.shape[1], x.shape[3])
        x = F.relu(self.conv1(x))
        out_bn_conv1 = self.batch_norm1(x)

        x = F.relu(self.conv2(out_bn_conv1))
        out_bn_conv2 = self.batch_norm2(x)
        res_conv2 = out_bn_conv2 + out_bn_conv1

        # TODO double check with musiccnn-training repo
        # no batch normalisation there
        x = F.relu(self.conv3(x))
        out_bn_conv3 = self.batch_norm3(x)
        res_conv3 = res_conv2 + out_bn_conv3

        out = torch.stack((out_bn_conv1, res_conv2, res_conv3))
        out = out.view(out.shape[1], out.shape[0] * out.shape[2], out.shape[3])

        return out
