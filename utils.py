import os
import torch


def load_checkpoint(model, optimizer, scheduler, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        loss = checkpoint['loss']
        val_loss = checkpoint['val_loss']
        acc = checkpoint['acc']
        val_acc = checkpoint['val_acc']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, scheduler, start_epoch, loss, val_loss, acc, val_acc


def rename_file(filename):
    if len(filename.split('_0.1lr_200Tmax'))>1:
        print('file already renamed:',filename)
    else:
        print('rename')
        print(filename)
        [part0, part1] = filename.split('_0.1lr')
        new_name = part0+'_0.1lr_200Tmax'+part1
        print(new_name)
        os.rename(filename, new_name)
    return