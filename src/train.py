import json
import os

import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from tqdm import tqdm

from alphapose.models import builder
from alphapose.opt import cfg, logger, opt
from alphapose.utils.metrics import DataLogger, calc_accuracy, calc_integral_accuracy, evaluate_mAP
from alphapose.utils.transforms import get_func_heatmap_to_coord


num_gpu = torch.cuda.device_count()

valid_batch = 1 * num_gpu
if opt.sync:
    norm_layer = nn.SyncBatchNorm
else:
    norm_layer = nn.BatchNorm2d

def train(opt, train_loader, model, criterion, optimizer):
    # TODO: write logs to wandb
    
    model.train()
    norm_type = cfg.LOSS.get('NORM_TYPE', None)  # in FastPose there is no NORM_TYPE property

    train_loader = tqdm(train_loader, dynamic_ncols=True)  # fancy progress bar

    epoch_losses = []
    epoch_accs = []

    # main train loop
    for i, (imgs, labels, label_masks, _, bboxes) in enumerate(train_loader):
        if isinstance(imgs, list):
            imgs = [img.cuda().requires_grad_() for img in imgs]
        else:
            imgs = imgs.cuda().requires_grad_()
        labels = labels.cuda()
        label_masks = label_masks.cuda()  # TODO: what is this?

        output = model(imgs)

        if cfg.LOSS.get('TYPE') == 'MSELoss':  # FastPose use MSELoss by default
            loss = 0.5 * criterion(output.mul(label_masks), labels.mul(label_masks))
            acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))
        else:
            loss = criterion(output, labels, label_masks)
            acc = calc_integral_accuracy(output, labels, label_masks, output_3d=False, norm_type=norm_type)

        if isinstance(imgs, list):
            batch_size = imgs[0].size(0)
        else:
            batch_size = imgs.size(0)

        epoch_losses.append(loss.item())  # loss
        epoch_accs.append(acc)  # accuracy
        avg_loss = np.array(epoch_losses).mean()
        avg_acc = np.array(epoch_accs).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loader.set_description(
            'loss: {loss:.8f} | acc: {acc:.4f}'.format(
                loss=avg_loss,
                acc=avg_acc)
        )

    epoch_loss = np.array(epoch_losses).sum()
    epoch_acc = np.array(epoch_accs).mean()

    train_loader.close()

    return epoch_loss, epoch_acc


def validate(opt, val_loader, model, criterion):
    model.eval()
    norm_type = cfg.LOSS.get('NORM_TYPE', None)  # in FastPose there is no NORM_TYPE property

    val_loader = tqdm(val_loader, dynamic_ncols=True)  # fancy progress bar

    epoch_losses = []
    epoch_accs = []

    # main train loop
    for i, (imgs, labels, label_masks, _, bboxes) in enumerate(val_loader):
        if isinstance(imgs, list):
            imgs = [img.cuda().requires_grad_() for img in imgs]
        else:
            imgs = imgs.cuda().requires_grad_()
        labels = labels.cuda()
        label_masks = label_masks.cuda()  # TODO: what is this?

        output = model(imgs)

        if cfg.LOSS.get('TYPE') == 'MSELoss':  # FastPose use MSELoss by default
            loss = 0.5 * criterion(output.mul(label_masks), labels.mul(label_masks))
            acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))
        else:
            loss = criterion(output, labels, label_masks)
            acc = calc_integral_accuracy(output, labels, label_masks, output_3d=False, norm_type=norm_type)

        if isinstance(imgs, list):
            batch_size = imgs[0].size(0)
        else:
            batch_size = imgs.size(0)

        epoch_losses.append(loss.item())  # loss
        epoch_accs.append(acc)  # accuracy
        avg_loss = np.array(epoch_losses).mean()
        avg_acc = np.array(epoch_accs).mean()

        val_loader.set_description(
            'validation loss: {loss:.8f} | validation acc: {acc:.4f}'.format(
                loss=avg_loss,
                acc=avg_acc)
        )

    epoch_loss = np.array(epoch_losses).sum()
    epoch_acc = np.array(epoch_accs).mean()

    val_loader.close()

    return epoch_loss, epoch_acc



def main():
    logger.info('\n******************************\n* OPT:')
    logger.info(opt)
    logger.info('\n******************************\n* CFG:')
    logger.info(cfg)
    logger.info('\n******************************\n')

    wandb.init(project="test-fastpose", name=opt.exp_id)

    model = preset_model(cfg)
    model = nn.DataParallel(model).cuda()

    wandb.watch(model)

    criterion = builder.build_loss(cfg.LOSS).cuda()  # MSELoss by default in FastPose

    if cfg.TRAIN.OPTIMIZER == 'adam':  # adam by default
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.TRAIN.LR)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)

    all_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True)
    
    train_len = int(len(all_dataset) * 0.8)
    val_len = len(all_dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_len, val_len])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=True, num_workers=opt.nThreads, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=False, num_workers=opt.nThreads, drop_last=True)

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    opt.trainIters = 0

    logger.info(f"\nUsing loss: {cfg.LOSS.get('TYPE')}\n")
    
    wandb.config.epochs = cfg.TRAIN.END_EPOCH
    wandb.config.loss = cfg.LOSS.get('TYPE')
    wandb.config.optimizer = cfg.TRAIN.OPTIMIZER
    wandb.config.lr = cfg.TRAIN.LR
    wandb.config.lr_factor = cfg.TRAIN.LR_FACTOR

    best_val_loss = 999

    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i + 1
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'############# Starting Epoch {opt.epoch} | LR: {current_lr} #############')

        # Training
        loss, acc = train(opt, train_loader, model, criterion, optimizer)
        loss /= len(train_loader)

        # Logs
        logger.epochInfo('Train', opt.epoch, loss, acc)
        wandb.log({"train_MSELoss": loss, "train_Acc": acc}, step=opt.epoch)
    
        lr_scheduler.step()

        if (i + 1) % opt.snapshot == 0:  # snapshot = 2 by default
            # Save checkpoint
            torch.save(model.module.state_dict(), './exp/{}-{}/model_last.pth'.format(opt.exp_id, cfg.FILE_NAME))
            # Prediction Test
            with torch.no_grad(): # TODO
                val_loss, val_acc = validate(opt, val_loader, model, criterion)
                val_loss /= len(val_loader)
                logger.epochInfo('Validation', opt.epoch, val_loss, val_acc)
                wandb.log({"val_MSELoss": val_loss, "train_Acc": val_acc}, step=opt.epoch)
                if val_loss.item() <= best_val_loss:
                    logger.info(f"\nValidation loss decreased, saving model...\n")
                    best_val_loss = val_loss.item()
                    torch.save(model.module.state_dict(), './exp/{}-{}/model_best.pth'.format(opt.exp_id, cfg.FILE_NAME))

        # Time to add DPG
        if (i + 1) == cfg.TRAIN.DPG_MILESTONE:
            logger.info(f"\nDPG milestone reached - enabling DPG\n")
            torch.save(model.module.state_dict(), './exp/{}-{}/model_{}_final.pth'.format(opt.exp_id, cfg.FILE_NAME, opt.epoch))
            # Adjust learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = cfg.TRAIN.LR
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.DPG_STEP, gamma=0.1)
            # Reset dataset
            all_dataset = builder.build_dataset(cfg.DATASET.TRAIN, preset_cfg=cfg.DATA_PRESET, train=True, dpg=True)
            train_dataset, val_dataset = torch.utils.data.random_split(all_dataset, [train_len, val_len])
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=True, num_workers=opt.nThreads, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu, shuffle=False, num_workers=opt.nThreads, drop_last=False)

    torch.save(model.module.state_dict(), './exp/{}-{}/model_{}_final_DPG.pth'.format(opt.exp_id, cfg.FILE_NAME, opt.epoch))


def preset_model(cfg):
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    if cfg.MODEL.PRETRAINED:
        logger.info(f'\nLoading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'\nLoading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items()
                            if k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('\nCreate new model')
        logger.info('=> init weights\n')
        model._initialize()

    return model


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
