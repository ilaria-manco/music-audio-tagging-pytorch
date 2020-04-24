import config_file
import torch
import argparse
import numpy as np
import pickle
import os
from math import log10, floor
import librosa
from models.musicnn import Musicnn
from models.simple_musicnn import SimpleMusicnn


def obtain_audio_rep(audio_path, mel_path):
    if not os.path.exists(mel_path):
    # compute_melspectrograms(audio_path, mel_path, config_file.config_preprocess['mtt_spec'])
        audio, sr = librosa.load(audio_path, sr=16000)
        mel_spectrogram = librosa.feature.melspectrogram(audio, n_fft=512, hop_length=256, n_mels=96)
        with open(mel_path, "wb") as f:
            pickle.dump(mel_spectrogram, f)  # mel_spectrogram shape: NxM
    mel_spec = pickle.load(open(mel_path, 'rb')).T
    mel_spec = np.log10(10000 * mel_spec + 1)
    return mel_spec


def split_spectrogram(mel_spec):
    patch_length = config_file.config_training["n_frames"]
    num_patches = int(mel_spec.shape[0] / patch_length)
    mel_spec_chunks = np.zeros((num_patches, patch_length, config_file.config_preprocess['mtt_spec']['n_mels']))
    for i in range(0, num_patches):
        mel_spec_chunks[i] = mel_spec[i:i + patch_length, :]
    return mel_spec_chunks


def get_top_tags(model_output, n_tags, patch_boundaries=None):
    mtt_tags = ['guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'indian', 'opera', 'male', 'singing', 'vocals', 'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age', 'dance', 'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird', 'country', 'metal', 'female voice', 'choral']
    if patch_boundaries:
        patch_start = patch_boundaries[0] * 187
        patch_end = patch_boundaries[1] * 187
        mean = np.mean(model_output[patch_start:patch_end], axis=0)
    else:
        mean = np.mean(model_output, axis=0)
    top_n_tags = {}
    for tag_index in mean.argsort()[-n_tags:][::-1]:
        top_n_tags.update({mtt_tags[tag_index]: round(mean[tag_index], -int(floor(log10(abs(mean[tag_index])))))})
    return top_n_tags


def extract_model_features(mel_spec, model_number):
    inputs = split_spectrogram(mel_spec)

    if model_number == "1":
        path_to_model = config_file.NO_POOLING_MODEL
        model = SimpleMusicnn(level="mid_end", y_input_dim=96, filter_type="timbral", k_height_factor=0.7, k_width_factor=1., filter_factor=1.6)
    elif model_number == "2":
        path_to_model = config_file.TEMP_POOLING_MODEL
        model = Musicnn(y_input_dim=96, filter_type="timbral", k_height_factor=0.7, k_width_factor=1., filter_factor=1.6, pool_type="temporal")
    elif model_number == "3":
        path_to_model = config_file.ATTENTION_MODEL
        model = Musicnn(y_input_dim=96, filter_type="timbral", k_height_factor=0.7, k_width_factor=1., filter_factor=1.6, pool_type="attention")
    else:
        raise ValueError("Invalid model number selected")
    print(path_to_model)
    model = model.double()
    model.eval() 
    checkpoint = torch.load(path_to_model, map_location="cuda:0")
    model.load_state_dict(checkpoint['state_dict'])
    inputs = torch.tensor(inputs)
    
    outputs = np.zeros((len(inputs), 50))
    with torch.no_grad():
        for index, i in enumerate(inputs):
            i = i.T.view(1, 1, 96, 187)
            output = torch.squeeze(model(i))
            outputs[index] = output
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute mel spectrogram of input audio")

    parser.add_argument("input_audio", type=str, help="path to input audio")
    parser.add_argument("output_path", type=str, help="path to save mel spectrogram to")
    parser.add_argument("model_number", type=str, help="number of pretrained model to load - 2 is best")
    args = parser.parse_args()

    input_audio_path = args.input_audio
    output_path = args.output_path
    model_number = args.model_number
    mel_spec = obtain_audio_rep(input_audio_path, output_path)
    extract_model_features(mel_spec, model_number)
