""" Code based on musicnn-training repo by Jordi Pons 
(https: // github.com / jordipons / musicnn - training / ) """

import torchaudio
import config_file
import pickle
import os
from pathlib import Path


def compute_melspectrograms(audio_file, melspec_file, config):
    audio, sr = torchaudio.load(audio_file)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        hop_length=config["hop_length"],
        n_fft=config["n_ftt"],
        n_mels=config["n_mels"])(audio).T
    print(mel_spectrogram.shape)
    length = mel_spectrogram.shape[0]

    with open(melspec_file, "wb") as f:
        pickle.dump(mel_spectrogram, f)  # mel_spectrogram shape: NxM
    return length


def preprocess(files, config):
    for file_index, file_to_process in enumerate(files):
        try:
            [file_id, audio_file, melspec_file] = files[file_index]
            if not os.path.exists(melspec_file[:melspec_file.rfind('/') + 1]):
                path = Path(melspec_file[:melspec_file.rfind('/') + 1])
                path.mkdir(parents=True, exist_ok=True)
                # TODO move block below out of if statement after first rerun
            length = compute_melspectrograms(audio_file, melspec_file, config)
            # index.tsv writing
            fw = open(
                config_file.DATA_PATH + config['melspectrogram_path'] + ".tsv",
                "a")
            fw.write("%s\t%s\t%s\n" %
                     (file_id, melspec_file[len(config_file.DATA_PATH):],
                      audio_file[len(config_file.DATA_PATH):]))
            fw.close()
            print(
                str(file_index) + '/' + str(len(files)) +
                ' Computed: %s' % audio_file)

        except Exception as e:
            ferrors = open(
                config_file.DATA_PATH + config['melspectrogram_path'] +
                "errors" + ".txt", "a")
            ferrors.write(audio_file + "\n")
            ferrors.write(str(e))
            ferrors.close()
            print('Error computing audio representation: ', audio_file)
            print(str(e))
