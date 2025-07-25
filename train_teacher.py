"""
Training a single model (student or teacher)
"""

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
# import tensorboard_logger as tb_logger
# 変更後
from torch.utils.tensorboard import SummaryWriter

from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders
from dataset.imagenet import get_imagenet_dataloader
from dataset.cinic10 import get_cinic10_dataloaders
# from dataset.imagenet_dali import get_dali_data_loader
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate
from helper.loops import train_vanilla as train, validate_vanilla

import signal
import sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def save_checkpoint_on_interrupt(signal, frame):
    print("\nInterrupt received. Saving checkpoint...")
    state = {
        'epoch': epoch,
        'best_acc': best_acc,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, 'checkpoint_interrupt.pth')
    torch.save(state, save_file)
    print(f"Checkpoint saved to {save_file}")
    sys.exit(0)

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # baisc
    parser.add_argument('--print_freq', type=int, default=200, help='print frequency') # トレーニング中にログ（進捗状況やメトリクス）を出力する頻度（バッチ）
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size') 
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers') # ワーカープロセスの数
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES') # GPUのID（0:最初のGPUを使用）
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet32x4')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar100', 'imagenet', 'cifar10', 'cinic10'], help='dataset')
    parser.add_argument('-t', '--trial', type=str, default='0', help='the experiment id')
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)

    # multiprocessing
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                    help='url used to set up distributed training')
    
    opt = parser.parse_args()

    # set different learning rates for these MobileNet/ShuffleNet models
    if opt.model in ['MobileNetV2', 'MobileNetV2_1_0', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_1_5']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard 
    opt.model_path = './save/teachers/models'
    opt.tb_path = './save/teachers/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # set the model name
    # opt.model_name = '{}_vanilla_{}_trial_{}'.format(opt.model, opt.dataset, opt.trial)
    # if opt.dali is not None:
    #     opt.model_name += '_dali:' + opt.dali

    # set the model name
    opt.model_name = '{}_vanilla_{}_trial_{}_epochs_{}_bs_{}'.format(
        opt.model, opt.dataset, opt.trial, opt.epochs, opt.batch_size
    )
    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

best_acc = 0
total_time = time.time()
def main():
    opt = parse_option()

    # signal.signal(signal.SIGINT, save_checkpoint_on_interrupt)  # Ctrl+Cで中断したときにチェックポイントを保存
    signal.signal(signal.SIGINT, save_checkpoint_on_interrupt)
    
    # ASSIGN CUDA_ID
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    if torch.cuda.is_available() and opt.gpu_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    else:
        print("No valid GPU ID provided or GPU not available. Using CPU.")

    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        if torch.cuda.is_available():
            # GPUが利用可能な場合のみ分散学習を有効化
            world_size = 1
            opt.world_size = ngpus_per_node * world_size
            # 分散プロセスを起動
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
        else:
            # GPUがない場合はエラーメッセージを出力して終了
            print("Error: Multiprocessing distributed training requires GPUs.")
            exit(1)
    else:
        # 単一プロセスで実行（CPUまたは単一GPU）
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)

def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    total_time = time.time()
    opt.gpu = int(gpu) if gpu and gpu.isdigit() else None
    opt.gpu_id = int(gpu) if gpu and gpu.isdigit() else None    
    
    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))
    else:
        print("No GPU detected. Using CPU for training.")


    if opt.multiprocessing_distributed:
        if opt.gpu is not None:
            opt.rank = int(gpu)
            dist_backend = 'nccl'
            dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                    world_size=opt.world_size, rank=opt.rank)
        else:
            print("Multiprocessing distributed training is not supported on CPU.")
            return
        
    print("Initializing model...")
    # model
    n_cls = {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000,
        'cinic10': 10,
    }.get(opt.dataset, None)
    
    try:
        model = model_dict[opt.model](num_classes=n_cls)
    except KeyError:
        print("This model is not supported.")

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available() and opt.gpu is not None:
        # GPUが利用可能な場合
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)  
                model = model.cuda(opt.gpu)
                criterion = criterion.cuda(opt.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                opt.batch_size = int(opt.batch_size / ngpus_per_node)
                opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion = criterion.cuda()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model).cuda()
            else:
                model = model.cuda()
    else:
        # CPU環境の場合
        print("Using CPU for training.")
        model = model  # CPU上でそのまま使用

    cudnn.benchmark = True if torch.cuda.is_available() else False

    # dataloader
    print(f"Loading dataset: {opt.dataset}...")
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.dataset == 'cifar10':
        train_loader, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.dataset == 'imagenet':
        if opt.dali is None:
            train_loader, val_loader, train_sampler = get_imagenet_dataloader(
                        dataset = opt.dataset,
                        batch_size=opt.batch_size, num_workers=opt.num_workers, 
                        multiprocessing_distributed=opt.multiprocessing_distributed)
        # else:
        #     train_loader, val_loader = get_dali_data_loader(opt)
    elif opt.dataset == 'cinic10':
        train_loader, val_loader = get_cinic10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)

    else:
        raise NotImplementedError(opt.dataset)
    print("Dataset loaded successfully!")

    # tensorboard
    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
                # 変更前
        # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
        
        # 変更後
        writer = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    # Early Stoppingのパラメータ
    patience = 10  # 精度が改善しない場合に停止するまでのエポック数
    no_improve_count = 0  # 改善が見られないエポック数をカウント
    train_acc_history = []  # train_accを記録するリスト

    # routine
    for epoch in range(1, opt.epochs + 1):
        print(f"Starting epoch {epoch}/{opt.epochs}...")
        if opt.multiprocessing_distributed:
            if opt.dali is None:
                train_sampler.set_epoch(epoch)
            # No test_sampler because epoch is random seed, not needed in sequential testing.

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_acc_top5, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' * Epoch {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}'.format(epoch, train_acc, train_acc_top5, time2 - time1))

            # logger.log_value('train_acc', train_acc, epoch)
            # logger.log_value('train_loss', train_loss, epoch)
            writer.add_scalar('train/acc', train_acc, epoch)
            writer.add_scalar('train/loss', train_loss, epoch)


        # train_accを記録
        train_acc_history.append(train_acc)

        test_acc, test_acc_top5, test_loss = validate_vanilla(val_loader, model, criterion, opt)

        if opt.dali is not None:
            train_loader.reset()
            val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))
            
            # logger.log_value('test_acc', test_acc, epoch)
            # logger.log_value('test_acc_top5', test_acc_top5, epoch)
            # logger.log_value('test_loss', test_loss, epoch)
            writer.add_scalar('test/acc', test_acc, epoch)
            writer.add_scalar('test/acc_top5', test_acc_top5, epoch)
            writer.add_scalar('test/loss', test_loss, epoch)

            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                no_improve_count = 0  # 改善が見られたのでリセット

                state = {
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'model': model.module.state_dict() if opt.multiprocessing_distributed else model.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
                
                test_merics = { 'test_loss': float('%.2f' % test_loss),
                                'test_acc': float('%.2f' % test_acc),
                                'test_acc_top5': float('%.2f' % test_acc_top5),
                                'epoch': epoch}
                
                save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))
                
                print('saving the best model!')
                torch.save(state, save_file)
            else:
                no_improve_count += 1  # 改善が見られない場合カウントを増やす



            
        print(f"Epoch {epoch} completed. Train Acc: {train_acc:.2f}, Loss: {train_loss:.2f}")

    # train_accの履歴を保存
    train_acc_file = os.path.join(opt.save_folder, "train_acc_history.json")
    save_dict_to_json({"train_acc": train_acc_history}, train_acc_file)
    print(f"Train accuracy history saved to {train_acc_file}")

    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        # This best accuracy is only for printing purpose.
        print('best accuracy:', best_acc)

        # save parameters
        state = {k: v for k, v in opt._get_kwargs()}

        # No. parameters(M)
        num_params = (sum(p.numel() for p in model.parameters())/1000000.0)
        state['Total params'] = num_params
        state['Total time'] =  float('%.2f' % ((time.time() - total_time) / 3600.0))
        params_json_path = os.path.join(opt.save_folder, "parameters.json") 
        save_dict_to_json(state, params_json_path)

if __name__ == '__main__':
    main()