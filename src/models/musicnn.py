import torch
import torch.nn as nn
import torch.nn.functional as F


class Musicnn(nn.Module):
    """Base class for network modules in the musicnn model,
    a deep convolutional neural networks for music audio tagging.

    Model architecture and original Tensorflow implementation: Jordi Pons -
    https://github.com/jordipons/musicnn/
    """

    def __init__(self):
        super(Musicnn, self).__init__()

    def forward(self, *input):
        raise NotImplementedError


class FrontEnd(Musicnn):
    """Musically motivated CNN front-end single layer http://mtg.upf.edu/node/3508.

    Args:
    - x:
    - filter_type:
    - filter_factor: multiplicative factor that controls the number of filter
      (i.e. the number of output channels)

    """
    def __init__(self, yInput, filter_type, k_height, k_width, filter_factor=1.6):
        super(FrontEnd, self).__init__()
        self.yInput = yInput
        self.k_height = k_height
        self.filter_factor = filter_factor
        self.out_channels = 0
        # LAYER
        if filter_type == "timbral":
            self.conv = self.timbral_block()
        if filter_type == "temporal":
            self.conv = self.temporal_block()
        nn.init.xavier_uniform_(self.conv.weight)
        self.batch_norm = nn.BatchNorm2d(num_features=self.out_channels)
        # TODO: is below needed?
        self.pool = nn.MaxPool2d(kernel_size=(1, self.conv_bn.shape[2]), stride=(1, self.conv_bn.shape[2]))

    def timbral_block(self):
        k_h = int(self.yInput * self.k_height)
        self.out_channels = int(self.filter_factor * 128)
        return nn.Conv2d(in_channels=1, out_channels=int(self.filter_factor * 128), kernel_size=(7, k_h))

    def temporal_block(self):
        k_w = int(self.yInput * self.k_width)
        self.out_channels = int(self.filter_factor * 32)
        return nn.Conv2d(in_channels=1, out_channels=int(self.filter_factor * 32), kernel_size=(k_w, 1))
        
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.batch_norm(x)
        x = self.pool(x)

        return x


class MidEnd(Musicnn):
    """Dense layers for mid-end.

    Args:
    - filter_factor: multiplicative factor that controls the number of filters
      (i.e. the number of output channels)

    """
    def __init__(self, filter_factor):
        super(MidEnd, self).__init__()
        # TODO: check padding and input dimensions
        # front_end_pad = 
        self.filter_factor = int(filter_factor)
        # LAYER 1
        self.conv1 = nn.Conv1d(1, self.filter_factor, kernel_size=7)
        self.batch_norm1 = nn.BatchNorm1d(num_features=self.out_channels)
        # LAYER 2
        self.conv2 = nn.Conv1d(1, self.filter_factor, kernel_size=7)
        self.batch_norm2 = nn.BatchNorm1d(num_features=self.out_channels)
        # LAYER 3
        self.conv3 = nn.Conv1d(1, self.filter_factor, kernel_size=7)
        self.batch_norm3 = nn.BatchNorm1d(num_features=self.out_channels)

    def forward(self, x):
        # TODO: add padding before each convolution
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
    
        return [out_bn_conv1, res_conv2, res_conv3]


class BackEnd(Musicnn):
    """Back end.

    Args:
    - feature_map:
    - number_of_classes
    - output_units: 

    """
    def __init__(self, feature_map, number_of_classes, output_units):
        super(BackEnd, self).__init__()
        self.feature_map = feature_map
        self.output_units = output_units
        self.num_of_classes = number_of_classes

        # LAYERS
        self.flat = Flatten()
        self.batch_norm = nn.BatchNorm1d(num_features=self.feature_map.shape[0])
        self.flat_pool_dropout = nn.Dropout()
        self.dense = nn.Linear(in_features=self.feature_map.shape[0], out_features=self.output_units)
        self.bn_dense = nn.BatchNorm1d(num_features=self.output_units)
        self.dense_dropout = nn.Dropout()
        self.dense2 = nn.Linear(in_features=self.output_units, out_features=self.num_of_classes)

    def forward(self):
        # temporal pooling
        max_pool = torch.max(self.feature_map, axis=1)
        mean_pool = torch.mean(self.feature_map, axes=[1])
        tmp_pool = torch.cat([max_pool, mean_pool], 2)

        # penultimate dense layer
        flat_pool = self.flatten(tmp_pool)
        flat_pool = self.batch_norm(flat_pool)
        flat_pool_dropout = self.flat_pool_dropout(flat_pool)
        dense = F.relu(self.dense(flat_pool_dropout))
        bn_dense = self.bn_dense(dense)
        dense_dropout = self.dense_dropout(bn_dense)
        logits = self.dense2(dense_dropout)
        return logits, bn_dense, mean_pool, max_pool


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
