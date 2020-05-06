import pickle
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset


class MTTDataset(Dataset):
    def __init__(self,
                 data_root,
                 index_file,
                 gt_file,
                 x_input_dim,
                 random_sampling=True,
                 preprocess=False):
        self.preprocess = preprocess
        self.x_input_dim = x_input_dim
        self.random_sampling = random_sampling
        if self.preprocess:
            self.data = self.preprocess_data()
        # if tsv, need to specify sep="\t".
        self.data_root = data_root
        # Index file: [id, file_path]
        self.index_file = pd.read_csv(data_root + index_file, sep="\t")
        # Ground truth file: [id, one-hot vectors]
        self.ground_truth = pd.read_csv(data_root + gt_file, sep="\t")
        self.file_ids, self.file_names, self.labels = self.get_labels()

    def __getitem__(self, index):
        # format the file path and load the file
        path = self.data_root + "mtt/mtt_mels/" + self.file_names[index]
        mel_spec = pickle.load(open(path, 'rb'))
        mel_spec = np.log10(10000 * mel_spec + 1)

        x_dim, y_dim, _ = mel_spec.shape
        last_frame = int(x_dim - int(self.x_input_dim)) + 1
        if self.random_sampling:
            time_stamp = random.randint(0, last_frame - 1)
            mel_spec = mel_spec[time_stamp:time_stamp + self.x_input_dim, :]

        return mel_spec.T, self.labels[index], self.file_ids[index]

    def __len__(self):
        return len(self.file_names)

    def get_labels(self):
        # initialize lists to hold file names, labels, and folder numbers
        file_ids = []
        file_names = []
        labels = []
        for i in range(0, len(self.ground_truth)):
            file_id = self.ground_truth.iloc[i, 0]
            file_ids.append(file_id)
            list_of_labels = [
                float(label)
                for label in self.ground_truth.iloc[i,
                                                    1].strip("[.]").split(",")
            ]
            labels.append(list_of_labels)
            # Map id to file path
            path_to_audio = list(
                self.index_file[self.index_file.iloc[:,
                                                     0] == file_id].iloc[:,
                                                                         1])[0]
            path_to_mel = path_to_audio[:path_to_audio.rfind(".")] + ".pk"
            file_names.append(path_to_mel)
        return file_ids, file_names, np.array(labels)

    def preprocess_data(*input_data):
        raise NotImplementedError
