import torch
import torch.nn as nn
import torch.nn.functional as F


class Musicnn(nn.Module):
    """Base class for network modules in the musicnn model,
    a deep convolutional neural networks for music audio tagging.

    Model architecture and original Tensorflow implementation: Jordi Pons -
    https://github.com/jordipons/musicnn/

    Args:
    - y_input_dim (int): height of the input
    - timbral_k_height (float or list): timbral filter height as a factor of the input dim 
    - temporal_k_width (int or list): temporal filter width of the input dim 
    - filter_factor (float): factor that controls the number of filters (output channels)
    - pool_type: type of pooling in the backend (temporal or attention)

    """
    def __init__(self,
                 y_input_dim,
                 filter_factor,
                 pool_type,
                 timbral_k_height=None,
                 temporal_k_width=None):
        super(Musicnn, self).__init__()
        self.pool_type = pool_type
        self.front_end = FrontEnd(y_input_dim, timbral_k_height, temporal_k_width,
                                  filter_factor)

        # front_end_channels = self.front_end.out_channels
        frontend_output_height = 561
        self.midend = MidEnd(input_channels=frontend_output_height)

        self.midend_channels = (self.midend.num_of_filters*3)+frontend_output_height
        self.back_end = BackEnd(input_height=self.midend_channels, output_units=50, pool_type=self.pool_type)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.front_end(x)
        x = self.midend(x)
        x = self.back_end(x)
        x = self.sigmoid(x)

        return x


class FrontEnd(nn.Module):
    """Musically motivated CNN front-end single layer http://mtg.upf.edu/node/3508. """
    def __init__(self, y_input_dim, timbral_k_height, temporal_k_width,
                 filter_factor):
        super(FrontEnd, self).__init__()
        self.y_input_dim = y_input_dim
        self.filter_factor = filter_factor
        if not isinstance(timbral_k_height, list):
            timbral_k_height = [timbral_k_height]
        if not isinstance(temporal_k_width, list):
            temporal_k_width = [temporal_k_width]
        self.k_h = [int(self.y_input_dim * k) for k in timbral_k_height]
        self.k_w = temporal_k_width

        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        for k_h in self.k_h:
            conv_layer, batch_norm_layer, pool_layer = self.timbral_block(k_h)
            self.conv_layers.append(conv_layer)
            self.batch_norm_layers.append(batch_norm_layer)
            self.pool_layers.append(pool_layer)

        for k_w in self.k_w:
            conv_layer, batch_norm_layer, pool_layer = self.temporal_block(k_w)
            self.conv_layers.append(conv_layer)
            self.batch_norm_layers.append(batch_norm_layer)
            self.pool_layers.append(pool_layer)

        # nn.init.xavier_uniform_(self.conv.weight)

    def timbral_block(self, k_h):
        out_channels = int(self.filter_factor * 128)
        conv_layer = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(k_h, 8),
                               padding=(0, 3))
        batch_norm = nn.BatchNorm2d(num_features=out_channels)
        h_out = self.y_input_dim - k_h + 1
        pool_layer = nn.MaxPool2d(kernel_size=(h_out, 1),
                                  stride=(h_out, 1))

        return conv_layer, batch_norm, pool_layer

    def temporal_block(self, k_w):
        out_channels = int(self.filter_factor * 32)
        pad_w = (k_w - 1) // 2
        conv_layer = nn.Conv2d(in_channels=1,
                               out_channels=out_channels,
                               kernel_size=(1, k_w),
                               padding=(0, pad_w))
        batch_norm = nn.BatchNorm2d(num_features=out_channels)
        pool_layer = nn.MaxPool2d(kernel_size=(self.y_input_dim, 1),
                                  stride=(self.y_input_dim, 1))

        return conv_layer, batch_norm, pool_layer

    def forward(self, x):
        out = []
        for i in range(len(self.conv_layers)):
            conv = F.relu(self.conv_layers[i](x))
            bn = self.batch_norm_layers[i](conv)
            pool = self.pool_layers[i](bn)
            out.append(pool)
        out = torch.cat(out, dim=1)
        out = torch.squeeze(out)
        return out


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
                               self.num_of_filters,
                               kernel_size=7,
                               padding=3)
        self.batch_norm1 = nn.BatchNorm1d(num_features=self.num_of_filters)
        # LAYER 2
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv1d(self.num_of_filters,
                               self.num_of_filters,
                               kernel_size=7,
                               padding=3)
        self.batch_norm2 = nn.BatchNorm1d(num_features=self.num_of_filters)
        # LAYER 3
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv1d(self.num_of_filters,
                               self.num_of_filters,
                               kernel_size=7,
                               padding=3)
        self.batch_norm3 = nn.BatchNorm1d(num_features=self.num_of_filters)

    def forward(self, x):
        out_conv1 = F.relu(self.conv1(x))
        out_bn_conv1 = self.batch_norm1(out_conv1)

        out_conv2 = F.relu(self.conv2(out_bn_conv1))
        out_bn_conv2 = self.batch_norm2(out_conv2)
        res_conv2 = out_conv2 + out_bn_conv1

        # TODO why is bn computed but not used
        out_conv3 = F.relu(self.conv3(out_bn_conv2))
        out_bn_conv3 = self.batch_norm3(out_conv3)
        res_conv3 = res_conv2 + out_conv3

        out = torch.cat((x, out_bn_conv1, res_conv2, res_conv3), dim=1)

        return out


class BackEnd(nn.Module):
    def __init__(self, input_height, output_units, pool_type):
        super(BackEnd, self).__init__()
        self.output_units = output_units
        self.pool_type = pool_type
        self.input_height = input_height

        # temporal pooling
        if self.pool_type == "temporal":
            self.mean_pool = nn.AvgPool2d(kernel_size=(1, 186))
            self.max_pool = nn.MaxPool2d(kernel_size=(1, 186))
            self.batch_norm = nn.BatchNorm1d(self.input_height*2)
            self.dense = nn.Linear(in_features=self.input_height*2, out_features=200)
            self.bn_dense = nn.BatchNorm1d(200)
            self.dense2 = nn.Linear(in_features=200, out_features=50)
        # attention
        elif self.pool_type == "attention":
            context = 3
            self.attention = nn.Conv1d(in_channels=self.input_height,
                                       out_channels=self.input_height,
                                       kernel_size=context,
                                       padding=int(context / 3))
            self.softmax = nn.Softmax(dim=1)
            self.batch_norm = nn.BatchNorm1d(self.input_height)
            self.dense = nn.Linear(in_features=self.input_height, out_features=100)
            self.bn_dense = nn.BatchNorm1d(100)
            self.dense2 = nn.Linear(in_features=100, out_features=50)

        self.flat = Flatten()
        self.flat_pool_dropout = nn.Dropout()
        self.dense_dropout = nn.Dropout()

    def forward(self, x):
        if self.pool_type == "temporal":
            max_pool = self.max_pool(x)
            mean_pool = self.mean_pool(x)
            tmp_pool = torch.cat((max_pool, mean_pool), dim=1)
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
