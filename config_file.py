DATA_PATH = "/homes/im311/repos/music-audio-tagging-pytorch/data/"
NO_POOLING_MODEL = DATA_PATH + "experiments/2020-04-21-13_58_40/best_model.pth.tar"
# TEMP_POOLING_MODEL is complete.
# Trained with patience = 75, initial lr=0.001
TEMP_POOLING_MODEL = DATA_PATH + "experiments/2020-04-22-13_42_14/best_model.pth.tar"
ATTENTION_MODEL = DATA_PATH + "experiments/2020-04-23-18_15_55/best_model.pth.tar"

config_preprocess = {
    'mtt_spec': {
        # first 3 keys are dataset dependent
        "identifier": "mtt",
        "index_file": "index_mtt.tsv",
        "audio_path":
        "/import/c4dm-datasets/MagnaTagATune/audio/",  # path to the audio files
        "hop_length":
        256,  # hop length for the FFT (number of samples between successive frames)
        "n_ftt":
        512,  # number of frequency bins of the FFT (frame size = length of FFT window)
        "n_mels": 96,  # number of mel bands
        "resample_sr": 16000
    }
}

DATASET = 'mtt'

config_training = {
    "name_run": "",

    # data
    "melspectrograms_path": "_mels/",
    "gt_train": DATASET + "/train_gt_" + DATASET + ".tsv",
    "gt_val": DATASET + "/val_gt_" + DATASET + ".tsv",
    "gt_test": DATASET + "/test_gt_" + DATASET + ".tsv",
    "index_file": DATASET + "/index_" + DATASET + ".tsv",

    # input setup
    "n_frames": 187,
    "pre-processing": "log_compression",
    "train_sampling": "random",

    # learning parameters
    "model_name": "77timbral_temporal",
    "load_model": None,
    "epochs": 600,
    "batch_size": 32,
    "weight_decay": 1e-5,
    "learning_rate": 0.01,
    "optimizer": "Adam",
    "patience": 40,

    # experiment settings
    "num_classes_dataset": 50,
    "val_batch_size": 32
}
