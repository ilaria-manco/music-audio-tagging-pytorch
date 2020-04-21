from torch import nn
import torch
import time
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import training_utils
from musicnn import FrontEnd, MidEnd, BackEnd, Musicnn
import torch.nn.functional as F


def train(training_set, validation_set, model, learning_rate, weight_decay, epochs, batch_size, patience, model_folder):
    cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(training_set, batch_size=batch_size)
    val_loader = DataLoader(validation_set, batch_size=batch_size)

    criterion = nn.BCELoss()

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Adaptive learning rate
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)

    model.to(cuda_device)

    k_patience = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()

        train_loss = train_epoch(model, criterion, optimizer, train_loader, cuda_device, is_training=True)
        val_loss = train_epoch(model, criterion, optimizer, val_loader, cuda_device, is_training=False)
        # Decrease the learning rate after not improving in the validation set
        scheduler.step(val_loss)
        
        time_stamp = str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))
        epoch_time = time.time() - epoch_start_time
        print('Epoch %d, train loss %g, val loss %g, epoch-time %gs, lr %g, time-stamp %s' %
              (epoch + 1, train_loss, val_loss, epoch_time, learning_rate, time_stamp))
        training_utils.update_training_log(model_folder, epoch + 1, train_loss, val_loss, epoch_time, learning_rate, time_stamp)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        training_utils.save_checkpoint(checkpoint, scheduler.is_better, model_folder)

        # TODO: early stopping
        # Early stopping: keep the best model in validation set
        if not scheduler.is_better:
            k_patience += 1
        
        if k_patience > patience:
            print("Early Stopping")
            break


def train_epoch(model, criterion, optimizer, data_loader, cuda_device, is_training):
    running_loss = 0.0
    model.train()
    n_batches = 0

    for i, batch in enumerate(data_loader):
        features, labels = batch
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
    if model_name == "77timbral_midend":
        model = Musicnn(level="mid_end", y_input_dim=y_input_dim, filter_type="timbral", k_height_factor=0.7, k_width_factor=1., filter_factor=1.6)
        # frontend = FrontEnd(level="front_end", y_input_dim=y_input_dim, filter_type="timbral", k_height_factor=0.7, k_width_factor=1., filter_factor=1.6)
    else:
        model = Musicnn(level="front_end", y_input_dim=y_input_dim, filter_type="timbral", k_height_factor=0.7, k_width_factor=1., filter_factor=1.6)
    
    return model
