import torch
import numpy as np
import config_file
import argparse
from model import FrontEnd, MidEnd, BackEnd
from dataset import SampleDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def load_data():
    sampling = confi


def train(*input):
    raise NotImplementedError


def build_model():
    inputs = np.zeros(1)
    timbral_layer_77 = FrontEnd(inputs, "timbral", 0.7, k_width, filter_factor=1.6)
    timbral_layer_74 = FrontEnd(inputs, "timbral", kernel_size)
    temporal_layer1 = FrontEnd(inputs, "temporal", )
    temporal_layer2 = FrontEnd(inputs, "temporal")
    temporal_layer3 = FrontEnd
    mid_end = MidEnd("something")
    back_end = BackEnd("something")
    # TODO: double check torch.cat does what I think it does
    frontend_features = torch.cat(timbral_layer_77, timbral_layer_74,
                                   temporal_layer1, temporal_layer2,
                                   temporal_layer3)
    midend_features = torch.cat(mid_end)
    backend_features = torch.cat(back_end)
    return frontend_features, midend_features, backend_features


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('configurationID', help='ID of the configuration dictionary')
    args = parser.parse_args()
    config = config_file.config_preprocess[args.configurationID]

    config = config_file.config_training
    sampling = config_file["train_sampling"]
    training_set = SampleDataset("", "", "")
    loader = DataLoader(training_set, batch_size=config["batch_size"])
