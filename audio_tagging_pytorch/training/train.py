from torch import nn
import torch
import time
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from audio_tagging_pytorch.models.musicnn import Musicnn
from audio_tagging_pytorch.training.training_utils import update_training_log, save_checkpoint


def train(training_set, validation_set, model, learning_rate, weight_decay,
          epochs, batch_size, patience, model_folder):
    cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(training_set, batch_size=batch_size)
    val_loader = DataLoader(validation_set, batch_size=batch_size)

    criterion = nn.BCELoss()

    optimizer = Adam(model.parameters(),
                     lr=learning_rate,
                     weight_decay=weight_decay)
    # Adaptive learning rate
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               mode='min',
                                               factor=0.5,
                                               patience=patience,
                                               verbose=True)

    model.to(cuda_device)

    k_patience = 0
    best_val = np.Inf

    for epoch in range(epochs):
        # Training iteration
        epoch_start_time = time.time()
        train_loss = train_epoch(model,
                                 criterion,
                                 optimizer,
                                 train_loader,
                                 cuda_device,
                                 is_training=True)
        val_loss = train_epoch(model,
                               criterion,
                               optimizer,
                               val_loader,
                               cuda_device,
                               is_training=False)
        # Decrease the learning rate after not improving in the validation set
        scheduler.step(val_loss)

        # check if val loss has been improving during patience period. If not, stop
        is_val_improving = scheduler.is_better(val_loss, best_val)
        if not is_val_improving:
            k_patience += 1
        else:
            k_patience = 0
        if k_patience > patience * 3:
            # Increase patience since adaptive learning rate is changed AFTER this step
            print("Early Stopping")
            break

        best_val = scheduler.best

        time_stamp = str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))
        epoch_time = time.time() - epoch_start_time
        lr = optimizer.param_groups[0]['lr']
        print(
            'Epoch %d, train loss %g, val loss %g, epoch-time %gs, lr %g, time-stamp %s'
            % (epoch + 1, train_loss, val_loss, epoch_time, lr, time_stamp))
        update_training_log(model_folder, epoch + 1, train_loss, val_loss,
                            epoch_time, lr, time_stamp)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        # save checkpoint in appropriate path (new or best)
        save_checkpoint(checkpoint, is_val_improving, model_folder)


def train_epoch(model, criterion, optimizer, data_loader, cuda_device,
                is_training):
    running_loss = 0.0
    model.train()
    n_batches = 0

    for i, batch in enumerate(data_loader):
        features, labels, file_ids = batch
        labels = labels.float()
        X = features.to(device=cuda_device)
        y = labels.to(device=cuda_device)

        _y = model(X)
        loss = criterion(_y, y)

        # Backward and optimize
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        running_loss += loss.item()

        n_batches += 1

    return running_loss / n_batches


def build_model(model_name, y_input_dim):
    if model_name == "timbral_temporal":
        model = Musicnn(y_input_dim=y_input_dim,
                        timbral_k_height=[0.4, 0.7],
                        temporal_k_width=[32, 64, 128],
                        filter_factor=1.6,
                        pool_type="temporal")
    elif model_name == "timbral_attention":
        model = Musicnn(y_input_dim=y_input_dim,
                        timbral_k_height=[0.4, 0.7],
                        temporal_k_width=[32, 64, 128],
                        filter_factor=1.6,
                        pool_type="attention")

    return model
