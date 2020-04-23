import torch
import torch.nn as nn
import torch.nn.functional as F


class Musicnn(nn.Module):
    """Base class for network modules in the musicnn model,
    a deep convolutional neural networks for music audio tagging.

    Model architecture and original Tensorflow implementation: Jordi Pons -
    https://github.com/jordipons/musicnn/
    """

    def __init__(self, y_input_dim, filter_type, k_height_factor, k_width_factor, filter_factor, pool_type):
        super(Musicnn, self).__init__()
        self.pool_type = pool_type
        self.front_end = FrontEnd(
            y_input_dim, filter_type, k_height_factor, k_width_factor, filter_factor)

        front_end_channels = self.front_end.out_channels
        self.midend = MidEnd(front_end_channels)

        self.back_end = BackEnd(50, self.pool_type)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.front_end(x)
        x = self.midend(x)
        x = self.back_end(x)
        x = self.sigmoid(x)
        return x


class FrontEnd(nn.Module):
    """Musically motivated CNN front-end single layer http://mtg.upf.edu/node/3508.

    Args:
    - x:
    - filter_type:
    - filter_factor: multiplicative factor that controls the number of filters
      (i.e. the number of output channels)

    """

    def __init__(self, y_input_dim, filter_type, k_height_factor, k_width_factor, filter_factor):
        super(FrontEnd, self).__init__()
        self.y_input_dim = y_input_dim
        self.filter_type = filter_type
        self.filter_factor = filter_factor
        self.k_h = int(self.y_input_dim * k_height_factor)
        self.k_w = int(self.y_input_dim * k_width_factor)

        # 1 convolutional layer
        if self.filter_type == "timbral":
            self.conv = self.timbral_block()
            max_pool_size = self.y_input_dim - self.k_h + 1
            self.pool = nn.MaxPool2d(kernel_size=(
                max_pool_size, 1), stride=(max_pool_size, 1))
        if self.filter_type == "temporal":
            self.conv = self.temporal_block()

        nn.init.xavier_uniform_(self.conv.weight)
        self.batch_norm = nn.BatchNorm2d(num_features=self.out_channels)

    def timbral_block(self):
        self.out_channels = int(self.filter_factor * 128)
        # return nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(self.k_h, 7))
        return nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(self.k_h, 7), padding=(0, 3))

    def temporal_block(self):
        self.out_channels = int(self.filter_factor * 32)
        # return nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(1, self.k_w))
        return nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(1, self.k_w), padding=(0, 3))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.batch_norm(x)
        x = self.pool(x)
        # TODO comment below for old model
        # x = torch.squeeze(x)

        return x


class MidEnd(nn.Module):
    """Dense layers for mid-end.

    Args:
    - filter_factor: multiplicative factor that controls the number of filters
      (i.e. the number of output channels)

    """

    def __init__(self, input_channels, num_of_filters=64):
        super(MidEnd, self).__init__()
        self.input_channels = input_channels
        self.num_of_filters = num_of_filters

        # LAYER 1
        self.conv1 = nn.Conv1d(self.input_channels,
                               self.num_of_filters, kernel_size=7, padding=3)
        self.batch_norm1 = nn.BatchNorm1d(num_features=self.num_of_filters)
        # LAYER 2
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
        # comment line below out
        x = x.view(x.shape[0], x.shape[1], x.shape[3])
        out_conv1 = F.relu(self.conv1(x))
        out_bn_conv1 = self.batch_norm1(out_conv1)

        out_conv2 = F.relu(self.conv2(out_bn_conv1))
        out_bn_conv2 = self.batch_norm2(out_conv2)
        # res_conv2 = out_bn_conv2 + out_bn_conv1
        res_conv2 = out_conv2 + out_bn_conv1

        # TODO double check with musiccnn-training repo:
        # why is bn computed but not used?
        out_conv3 = F.relu(self.conv3(out_bn_conv2))
        out_bn_conv3 = self.batch_norm3(out_conv3)
        res_conv3 = res_conv2 + out_conv3

        out = torch.cat((out_bn_conv1, res_conv2, res_conv3), dim=1)
        return out


class BackEnd(nn.Module):
    def __init__(self, output_units, pool_type):
        super(BackEnd, self).__init__()
        self.output_units = output_units
        self.pool_type = pool_type

        # TODO add pool_type as argument after implementing attention
        # temporal pooling
        if self.pool_type == "temporal":
            self.mean_pool = nn.AvgPool2d(kernel_size=(187, 1))
            self.max_pool = nn.MaxPool2d(kernel_size=(187, 1))
            self.batch_norm = nn.BatchNorm1d(374)
            self.dense = nn.Linear(in_features=374, out_features=200)
            self.bn_dense = nn.BatchNorm1d(200)
            self.dense2 = nn.Linear(in_features=200, out_features=50)
        # attention
        elif self.pool_type == "attention":
            context = 3
            self.attention = nn.Conv1d(in_channels=192,
                                       out_channels=192,
                                       kernel_size=context,
                                       padding=int(context / 3))
            self.softmax = nn.Softmax(dim=1)
            self.batch_norm = nn.BatchNorm1d(192)
            self.dense = nn.Linear(in_features=192, out_features=100)
            self.bn_dense = nn.BatchNorm1d(100)
            self.dense2 = nn.Linear(in_features=100, out_features=50)

        self.flat = Flatten()
        self.flat_pool_dropout = nn.Dropout()
        self.dense_dropout = nn.Dropout()

    def forward(self, x):
        if self.pool_type == "temporal":
            max_pool = self.max_pool(x)
            mean_pool = self.mean_pool(x)
            tmp_pool = torch.cat((max_pool, mean_pool), dim=2)
        elif self.pool_type == "attention":
            attention_weights = self.softmax(self.attention(x))
            tmp_pool = torch.mul(attention_weights, x)
            tmp_pool = torch.sum(tmp_pool, dim=2)

        flat_pool = self.flat(tmp_pool)
        flat_pool = self.batch_norm(flat_pool)
        flat_pool_dropout = self.flat_pool_dropout(flat_pool)

        dense = F.relu(self.dense(flat_pool_dropout))
        bn_dense = self.bn_dense(dense)
        dense_dropout = self.dense_dropout(bn_dense)
        out = self.dense2(dense_dropout)
        return out


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
