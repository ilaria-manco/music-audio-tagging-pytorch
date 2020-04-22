
"""
Ilaria Manco 2020
ECS7013P - Deep Learning for Audio and Music 
Coursework
File: run_training.py
Description: 
"""

import config_file
import json
import os
import torch
import time
from train import train, build_model
from dataset import MTTDataset

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    config = config_file.config_training

    # 1. Load config parameters used in 'preprocess_data.py'
    config_json = config_file.DATA_PATH + config_file.DATASET + "/" + \
        config_file.DATASET + config['melspectrograms_path'] + 'config.json'
    with open(config_json, "r") as f:
        params = json.load(f)
    config['audio_rep'] = params
    config['x_input_dim'] = config['n_frames']
    config['y_input_dim'] = config['audio_rep']['n_mels']

    # 2. Load data
    sampling = config["train_sampling"]
    if sampling == "random":
        random_sampling = True
        print(random_sampling)
    # Load training and validation data
    training_set = MTTDataset(
        config_file.DATA_PATH, config["index_file"], config["gt_train"], config['x_input_dim'], random_sampling)
    validation_set = MTTDataset(
        config_file.DATA_PATH, config["index_file"], config["gt_val"], config['x_input_dim'])

    config['classes_vector'] = list(range(config['num_classes_dataset']))

    print('# Train:', len(training_set.file_ids))
    print('# Val:', len(validation_set.file_ids))
    print('# Classes:', config['classes_vector'])

    # 3. Save experimental settings
    experiment_id = str(time.strftime('%Y-%m-%d-%H_%M_%S', time.gmtime()))
    model_folder = config_file.DATA_PATH + \
        'experiments/' + str(experiment_id) + '/'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    json.dump(config, open(model_folder + 'config.json', 'w'))
    print('\nConfig file saved: ' + str(config))

    print('\nEXPERIMENT: ', str(experiment_id))
    print('-----------------------------------')

    # 4. Define model, optimizer & loss
    if config['load_model'] is not None:
        model = torch.load(config['load_model'])
        print('Pre-trained model loaded!')
    else:
        model = build_model(config['model_name'], config['y_input_dim'])

    # 5. Write headers of the train_log.tsv
    log_file = open(model_folder + 'train_log.tsv', 'a')
    log_file.write(
        'Time_stamp\tepoch\ttrain_loss\tval_loss\tepoch_time\tlearing_rate\n')
    log_file.close()

    # 6. Start training
    epochs = config['epochs']
    batch_size = config["batch_size"]
    learning_rate = config['learning_rate']
    patience = config['patience']
    weight_decay = config['weight_decay']

    print('Training started..')

    train(training_set, validation_set, model, learning_rate,
          weight_decay, epochs, batch_size, patience, model_folder)
