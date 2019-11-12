
import argparse

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
from tqdm import tqdm
import os.path as osp
# from networks.ccnet import Res_Deeplab
# from networks.deeplabv3 import Res_Deeplab
# from networks.pspnet import Res_Deeplab
# from networks.danet import Res_Deeplab
# from networks.v1 import Res_Deeplab
# from networks.v2 import Res_Deeplab
# from networks.v3 import Res_Deeplab
# from networks.v4 import Res_Deeplab
from networks.danet_color import Res_Deeplab

from dataset.datasets import ModaDataset
import random
import timeit
import logging
from tensorboardX import SummaryWriter
from utils.utils import decode_labels, inv_preprocess, decode_predictions
from utils.criterion import CriterionCrossEntropy, CriterionOhemCrossEntropy, CriterionDSN, CriterionOhemDSN
from utils.encoding import DataParallelModel, DataParallelCriterion


CUDA_LAUNCH_BLOCKING = 1

start = timeit.default_timer()

IMG_MEAN = np.array((121.20981688, 132.55833042, 141.29624324), dtype=np.float32)

BATCH_SIZE = 8
DATA_DIRECTORY = './data/'
LIST_PATH = './data/train.lst'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 14
NUM_STEPS = 150000
# NUM_STEPS = 1000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './dataset/resnet101-imagenet.pth'
# RESTORE_FROM = 'snapshots/ccnet-da//more/CS_scenes_30000.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10000
SNAPSHOT_DIR = 'snapshots/color-new-da/'
WEIGHT_DECAY = 1e-4

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet NetWork")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the ModaNet dataset.")
    parser.add_argument("--list-path", type=str, default=LIST_PATH,
                        help="List path to the directory containing the ModaNet dataset.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-rotate", action="store_true",
                        help="Whether to randomly rotate the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default='1,0,2,3',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=2,
                        help="choose the number of recurrence.")
    parser.add_argument("--ft", type=bool, default=False,
                        help="fine-tune the model with large input size.")
    return parser.parse_args()

args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003

def main():
    writer = SummaryWriter(args.snapshot_dir)

    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    cudnn.enabled = True

    deeplab = Res_Deeplab(num_classes=args.num_classes)
    print(deeplab)

    saved_state_dict = torch.load(args.restore_from)
    new_params = deeplab.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

    deeplab.load_state_dict(new_params)

    model = DataParallelModel(deeplab)
    model.train()
    model.float()
    # model.apply(set_bn_momentum)
    model.cuda()

    criterion = CriterionDSN()  # CriterionCrossEntropy()
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(ModaDataset(args.data_dir, args.list_path, max_iters=args.num_steps*args.batch_size,
                    # mirror=args.random_mirror,
                                                mirror=True, rotate=True,
                                              mean=IMG_MEAN),
                    batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, deeplab.parameters()), 'lr': args.learning_rate}],
        lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    print('start training!')
    for i_iter, batch in enumerate(trainloader):
        i_iter += args.start_iters
        images, labels, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter)
        # preds = model(images, args.recurrence)
        preds = model(images)

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        if i_iter % 100 == 0:
            writer.add_scalar('learning_rate', lr, i_iter)
            writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)

        # if i_iter % 5000 == 0:
        #     images_inv = inv_preprocess(images, args.save_num_images, IMG_MEAN)
        #     labels_colors = decode_labels(labels, args.save_num_images, args.num_classes)
        #     if isinstance(preds, list):
        #         preds = preds[0]
        #     preds_colors = decode_predictions(preds, args.save_num_images, args.num_classes)
        #     for index, (img, lab) in enumerate(zip(images_inv, labels_colors)):
        #         writer.add_image('Images/'+str(index), img, i_iter)
        #         writer.add_image('Labels/'+str(index), lab, i_iter)
        #         writer.add_image('preds/'+str(index), preds_colors[index], i_iter)

        print('iter = {} of {} completed, loss = {}'.format(i_iter, args.num_steps, loss.data.cpu().numpy()))

        if i_iter >= args.num_steps - 1:
            print('save model ...')
            torch.save(deeplab.state_dict(), osp.join(args.snapshot_dir, 'CS_scenes_' + str(args.num_steps) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0:
            print('taking snapshot ...')
            torch.save(deeplab.state_dict(), osp.join(args.snapshot_dir, 'CS_scenes_' + str(i_iter) + '.pth'))

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()