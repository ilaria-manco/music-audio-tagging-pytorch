from datetime import datetime
import torch


def get_epoch_time():
    return int((datetime.now() - datetime(1970, 1, 1)).total_seconds())


def update_training_log(model_folder, epoch, train_loss, val_loss, epoch_time,
                        learning_rate, time_stamp):
    # log_file = open(model_folder + 'train_log.tsv', 'a')
    log_file = open(model_folder + 'train_log.tsv', 'a')
    log_file.write(
        '%d\t%g\t%g\t%gs\t%g\t%s\n' %
        (epoch, train_loss, val_loss, epoch_time, learning_rate, time_stamp))
    log_file.close()


def save_checkpoint(state, is_best, checkpoint_dir):
    if is_best:
        checkpoint_path = checkpoint_dir + 'best_model.pth.tar'
    else:
        checkpoint_path = checkpoint_dir + 'checkpoint.pth.tar'
    torch.save(state, checkpoint_path)
