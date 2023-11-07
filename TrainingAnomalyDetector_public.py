import argparse
import logging
import os
from os import path
import torch
import torch.backends.cudnn as cudnn

import metric
from anomaly_detector_model import AnomalyDetector, custom_objective, RegularizedLoss
from features_loader import FeaturesLoader
from lr_scheduler import MultiFactorScheduler
from model import model

parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
# debug
parser.add_argument('--debug-mode', type=bool, default=True,
                    help="print all setting for debugging.")
# io
parser.add_argument('--features_path', default='out_SHTech_VGG_RGB',
                    help="path to features")
parser.add_argument('--annotation_path', default="/home/zhang/AnomalyDetectionCVPR2018-Pytorch-master/list/TrainList_ShangHaiTech.txt",
                    help="path to train annotation")
parser.add_argument('--annotation_path_test', default="/home/zhang/AnomalyDetectionCVPR2018-Pytorch-master/list/TestList_ShangHaiTech.txt",
                    help="path to test annotation")
parser.add_argument('--clip-length', type=int, default=16,
                    help="define the length of each input sample.")
parser.add_argument('--train-frame-interval', type=int, default=2,
                    help="define the sampling interval between frames.")
parser.add_argument('--val-frame-interval', type=int, default=2,
                    help="define the sampling interval between frames.")
parser.add_argument('--task-name', type=str, default='',
                    help="name of current task, leave it empty for using folder name")
parser.add_argument('--model-dir', type=str, default="./exps_ShanghaiTech_VGG_RGB_5w/model",
                    help="set logging file.")
parser.add_argument('--log-file', type=str, default="log_ShanghaiTech_VGG_RGB_5w.log",
                    help="set logging file.")

# device
parser.add_argument('--gpus', type=str, default="0",
                    help="define gpu id")
parser.add_argument('--resume-epoch', type=int, default=-1,
                    help="resume train")
# optimization
parser.add_argument('--fine-tune', type=bool, default=True, help="apply different learning rate for different layers")
parser.add_argument('--batch-size', type=int, default=16,
                    help="batch size")
parser.add_argument('--lr-base', type=float, default=0.01,
                    help="learning rate")
parser.add_argument('--lr_mult_old_layers', type=float, default=0.2,
                    help="learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help="weight_decay")
parser.add_argument('--lr-steps', type=list, default=[int(1e4 * x) for x in [5, 10, 15]],
                    help="number of samples to pass before changing learning rate")  # 1e6 million

parser.add_argument('--lr-factor', type=float, default=1,
                    help="reduce the learning with factor")
parser.add_argument('--save-frequency', type=float, default=1000,
                    help="save once after N epochs")
parser.add_argument('--end-epoch', type=int, default=50000,
                    help="maxmium number of training epoch")
parser.add_argument('--random-seed', type=int, default=1,
                    help='random seed (default: 1)')

if not path.exists('models32'):
    os.mkdir('models32')


def set_logger(log_file='', debug_mode=False):
    if log_file:
        if not os.path.exists("./" + os.path.dirname(log_file)):
            os.makedirs("./" + os.path.dirname(log_file))
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]

    """ add '%(filename)s:%(lineno)d %(levelname)s:' to format show source file """
    logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers)


def adjust_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        if 'lr_mult' in param_group:
            lr_mult = param_group['lr_mult']
        else:
            lr_mult = 1.0
        param_group['lr'] = lr * lr_mult


if __name__ == "__main__":
    args = parser.parse_args()
    set_logger(log_file=args.log_file, debug_mode=args.debug_mode)

    torch.manual_seed(args.random_seed)

    train_loader = FeaturesLoader(features_path=args.features_path,
                                  annotation_path=args.annotation_path)
    """
    val_loader = FeaturesLoader(features_path=args.features_path,
                                annotation_path=args.annotation_path_test)
    """
    train_iter = torch.utils.data.DataLoader(train_loader,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=8,  # 4, # change this part accordingly
                                             pin_memory=True)
    """
    eval_iter = torch.utils.data.DataLoader(val_loader,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=1,  # 4, # change this part accordingly
                                            pin_memory=True)
    """
    iter_seed = torch.initial_seed() + 100

    network = AnomalyDetector()
    net = model(net=network,
                criterion=RegularizedLoss(network, custom_objective).cuda(),
                model_prefix=args.model_dir,
                step_callback_freq=5,
                save_checkpoint_freq=args.save_frequency,
                opt_batch_size=args.batch_size,  # optional, 60 in the paper
                )

    if torch.cuda.is_available():
        net.net.cuda()
        torch.cuda.manual_seed(args.random_seed)
        net.net = torch.nn.DataParallel(net.net).cuda()
    """
        In the original paper:
        lr = 0.01
        epsilon = 1e-8
    """
    optimizer = torch.optim.Adadelta(net.net.parameters(),
                                     lr=args.lr_base,
                                     eps=1e-8)

    if args.resume_epoch < 0:
        epoch_start = 0
        step_counter = 0
    else:
        # net.load_checkpoint(epoch=args.resume_epoch, optimizer=optimizer)
        epoch_start = args.resume_epoch
        step_counter = epoch_start * train_iter.__len__()

    # set learning rate scheduler
    num_worker = 1
    lr_scheduler = MultiFactorScheduler(base_lr=args.lr_base,
                                        steps=[int(x / (args.batch_size * num_worker)) for x in args.lr_steps],
                                        factor=args.lr_factor,
                                        step_counter=step_counter)
    # define evaluation metric
    metrics = metric.MetricList(metric.Loss(name="loss-ce"),)
    # enable cudnn tune
    cudnn.benchmark = True

    net.fit(train_iter=train_iter,
            eval_iter=None,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            epoch_start=epoch_start,
            epoch_end=args.end_epoch)
