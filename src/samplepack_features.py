import config_file
import os
import pickle
import argparse
import numpy as np
from extract_features import obtain_audio_rep, split_spectrogram, extract_model_features


def get_spectrograms_of_samples(path_to_samplepack):
    mel_num = 0
    for root, dirs, files in os.walk(path_to_samplepack):
        for name in files:
            if name[-4:] == '.wav':
                audio_path = os.path.join(root, name)
                mel_spec_path = audio_path.replace("wav", "pk")
                if not os.path.exists(mel_spec_path):
                    try:
                        obtain_audio_rep(audio_path, mel_spec_path)
                        mel_num += 1
                        print("mel no. " + str(mel_num) + " computed for ", audio_path)
                    except Exception:
                        print('Error computing audio representation: ', audio_path)
                                

def save_model_features(path_to_samplepack, model_number):
    i = 0
    for root, dirs, files in os.walk(path_to_samplepack):
        for name in files:
            if name[-3:] == '.pk':
                mel_spec_path = os.path.join(root, name)
                path_to_feature = mel_spec_path.replace("pk", "npy")
                # if not os.path.exists(path_to_feature):
                mel_spec = pickle.load(open(mel_spec_path, 'rb'))
                try:
                    output = extract_model_features(mel_spec, model_number)
                    np.save(path_to_feature, output)
                    i += 1
                    print(str(i), "features computed", str(path_to_feature))
                except Exception:
                    print("Error computing feature:", path_to_feature)


def get_closest_feature(input_feature, path_to_samplepack, top_n=3):
    features = {}
    for root, dirs, files in os.walk(path_to_samplepack):
        for name in files:
            if name[-4:] == '.npy':
                feature_path = os.path.join(root, name)
                output_feature = np.load(feature_path)
                dist = np.linalg.norm(np.mean(output_feature, axis=0) - np.mean(input_feature, axis=0))
                if not np.isnan(dist):
                    features.update({feature_path: dist})
    sorted_features = sorted(features.items(), key=lambda item: item[1], reverse=True)[:3]
    return sorted_features
                     

if __name__ == '__main__':
    # get_spectrograms_of_samples(path_to_samplepack)
    # save_model_features(path_to_samplepack, model_number="2")
    parser = argparse.ArgumentParser(description="Compute mel spectrogram of input audio")
    parser.add_argument("input_audio", type=str, help="path to input audio")
    parser.add_argument("path_to_samplepack", type=str, help="path to sample pack")
    parser.add_argument("num_samples", type=str, help="number of samples to return")
    args = parser.parse_args()

    path_to_samplepack = "/homes/im311/datasets/musicradar_samples/samples/"

    # sample_track = "Saccharine.wav"
    # sample_track_path = config_file.DATA_PATH + "sample_tracks/" + sample_track
    sample_track_path = args.input_audio
    top_n = args.num_samples

    mel_spec_path = sample_track_path.replace("wav", "pk")
    path_to_feature = sample_track_path.replace("wav", "npy")
    mel_spec = pickle.load(open(mel_spec_path, 'rb')).T
    output = extract_model_features(mel_spec, model_number="2")
    # np.save(path_to_feature, output)
    # feature = np.load(path_to_feature)

    close_features = get_closest_feature(output, path_to_samplepack, top_n=top_n)
    print(close_features)
