import argparse
import config_file
import os
import json
from preprocess_data import preprocess

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('configurationID',
                        help='ID of the configuration dictionary')
    args = parser.parse_args()
    config = config_file.config_preprocess[args.configurationID]

    # 1. Make new directory for the mel spectrograms
    config["melspectrogram_path"] = config['identifier'] + \
        "/%s_mels/" % (config['identifier'])
    # set audio representations folder
    if not os.path.exists(config_file.DATA_PATH +
                          config['melspectrogram_path']):
        os.makedirs(config_file.DATA_PATH + config['melspectrogram_path'])

    # 2. Find audio files to preprocess
    files_to_preprocess = []
    f = open(config_file.DATA_PATH + config["index_file"])
    for line in f.readlines():
        file_id, audio = line.strip().split("\t")
        melspectrogram = audio[:audio.rfind(".")] + ".pk"  # .npy or .pk
        # (id, path to audio file, path to mel spectrogram)
        files_to_preprocess.append(
            (file_id, config["audio_path"] + audio, config_file.DATA_PATH +
             config["melspectrogram_path"] + melspectrogram))

    # 3. Compute mel spectrograms
    preprocess(files_to_preprocess, config)
    # 4. Save the parameters in a json
    json.dump(
        config,
        open(
            config_file.DATA_PATH + config['melspectrogram_path'] +
            "config.json", "w"))
