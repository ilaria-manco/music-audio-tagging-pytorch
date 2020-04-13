DATA_PATH = "/homes/im311/dl4am/data/"

config_preprocess = {
    'mtt_spec': {
        # first 3 keys are dataset dependent
        "identifier": "mtt",
        "index_file": "index_mtt.tsv",
        "audio_path": "/import/c4dm-datasets/MagnaTagATune/audio/",     # path to the audio files
        "hop_length": 256,                                              # hop length for the FFT (number of samples between successive frames)
        "n_ftt": 512,                                                   # number of frequency bins of the FFT (frame size = length of FFT window)
        "n_mels": 96,                                                   # number of mel bands
        "resample_sr": 16000
    },
    'musicradar_spec': {
        # first 3 keys are dataset dependent
        "identifier": "musicradar",
        "index_file": "index_musicradar.tsv",
        "audio_path": "",                   # path to the audio files
        "hop_length": 256,                  # hop length for the FFT (number of samples between successive frames)
        "n_ftt": 512,                       # number of frequency bins of the FFT (frame size = length of FFT window)
        "n_mels": 96,                       # number of mel bands
        "resample_sr": 16000
    }
}

DATASET = 'mtt'  # 'mtt' or 'musicradar'

config_training = {
    "spec": {
        "name_run": "",
        # data
        "melspectrograms_path": DATA_PATH + DATASET + "/_mels",
        "gt_train": DATASET + "/train_gt_" + DATASET + ".tsv",
        "gt_val": DATASET + "/val_gt_" + DATASET + ".tsv",
        # input setup
        "n_frames" = 187,
        "pre-processing": "log_compression",
        "train_sampling": "random",
        # "training
        # training parameters
        "load_model": None,
        "epochs": 600,
        "batch_size": 32,
        "weight_decay": 1e-5,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "patience": 75,
        # experiment settings
        "num_classes_dataset": 50,
        "val_batch_size": 32
    }
}
