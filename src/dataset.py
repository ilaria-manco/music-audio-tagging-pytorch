import torch
import pandas as pd
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    # wrapper for the dataset
    # Argument List
    #  path to the tsv file
    #  path to the audio files / mel files?

    def __init__(self, data_root, index_file, gt_file, preprocess=False):
        self.preprocess = preprocess
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
        path = self.data_root + "/" + self.file_names[index]
        mel_spec = torch.load(path)
        # TODO: add log compression

        return mel_spec, self.labels[index]

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
            labels.append(self.ground_truth.iloc[i, 1])
            # Map id to file path
            path_to_audio = list(self.index_file[self.index_file.iloc[:, 0] == file_id].iloc[:, 1])[0]
            path_to_mel = path_to_audio[:path_to_audio.rfind(".")] + ".pk"
            file_names.append(path_to_mel)
        return file_ids, file_names, labels

    def preprocess_data(*input_data):
        raise NotImplementedError