import os
import json
import argparse
import torch
import dataloaders
import models
import math
from utils import Logger
from trainer import Trainer
import torch.nn.functional as F
from utils.losses import abCE_loss, CE_loss, consistency_weight, dice_loss

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    torch.manual_seed(42)
    train_logger = Logger()

    # DATA LOADERS
    config['train_supervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']
    supervised_loader = dataloaders.VOC(config['train_supervised'])
    unsupervised_loader = dataloaders.VOC(config['train_unsupervised'])
    val_loader = dataloaders.VOC(config['val_loader'])
    iter_per_epoch = len(unsupervised_loader)

    # SUPERVISED LOSS
    if config['model']['sup_loss'] == 'CE':
        sup_loss = CE_loss
    elif config['model']['sup_loss'] == 'multi':
        diceloss = dice_loss().cuda()
        sup_loss = [CE_loss, diceloss]
    elif config['model']['sup_loss'] == 'dice':
        diceloss = dice_loss().cuda()
        sup_loss = diceloss
    else:
        sup_loss = abCE_loss(iters_per_epoch=iter_per_epoch, epochs=config['trainer']['epochs'],
                                num_classes=val_loader.dataset.num_classes)

    # MODEL
    rampup_ends = int(config['ramp_up'] * config['trainer']['epochs'])
    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=len(unsupervised_loader),
                                        rampup_ends=rampup_ends)
    cons_w_flip = consistency_weight(final_w=config['flip_loss_w'], iters_per_epoch=len(unsupervised_loader),
                                        rampup_ends=rampup_ends)

    model = models.CCT(num_classes=val_loader.dataset.num_classes, conf=config['model'],
    						sup_loss=sup_loss, cons_w_unsup=cons_w_unsup,
    						weakly_loss_w=config['weakly_loss_w'], cons_w_flip=cons_w_flip,
                            use_weak_lables=config['use_weak_lables'])

    # pretrained_dict = torch.load('./competition/saved_supervise_0728/CCT/checkpoint_7.pth')
    # checkpoint = torch.load(args.model)
    # model.load_state_dict(pretrained_dict['state_dict'], strict=True)

    print(f'\n{model}\n')

    # TRAINING
    trainer = Trainer(
        model=model,
        resume=resume,
        config=config,
        supervised_loader=supervised_loader,
        unsupervised_loader=unsupervised_loader,
        val_loader=val_loader,
        iter_per_epoch=iter_per_epoch,
        train_logger=train_logger)

    trainer.train()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config.json',type=str,
                        help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)
    args = parser.parse_args()

    config = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True
    main(config, args.resume)
