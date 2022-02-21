# coding=utf-8
from __future__ import absolute_import, division, print_function

import sys
import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
from apex import amp
# from apex.parallel import DistributedDataParallel as DDP

from models.config import get_config

from models.ResNet.resnet_PBG import resnet18 as ResNet18_PBG, PiggybackConv

from models.ViT.modeling_RKRPBG import VisionTransformer as VisionTransformer_RKRPBG
from models.ViT.modeling_PBG import VisionTransformer as VisionTransformer_PBG, PiggybackFC

from models.Swin.swin_PBG import SwinTransformer as SwinTransformer_PBG

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader, get_loader_splitCifar100, get_loader_splitImagenet, get_VD_loader
from utils.dist_util import get_world_size

# DDP
from argparse import ArgumentParser
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, task):
    save_folder = os.path.join(args.output_dir, args.name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(save_folder, "%s_task%d.bin" % (args.name, task))
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def model_initialization(args, model):
    if 'ResNet' in args.model_type:
        # 事前学習済モデルのロード
        pre_model_dict = torch.load('/host/space0/takeda-m/jupyter/notebook/RKR/model/resnet34-b627a593.pth')
        pre_model_keys = [k for k, v in pre_model_dict.items()]
        new_model_dict = {}
        for k, v in model.state_dict().items():
            if k in pre_model_keys or 'unc_filt' in k:
                pre_k = k.replace('.unc_filt', '.weight')
                new_model_dict[k] = pre_model_dict[pre_k]
            else:
                new_model_dict[k] = v
        model.load_state_dict(new_model_dict)

    elif 'ViT' in args.model_type:
        model.load_from(np.load(args.pretrained_dir))

    elif 'Swin' in args.model_type:
        # 事前学習済モデルのロード
        pre_model_dict = torch.load(args.pretrained_dir)['model']
        del pre_model_dict['head.weight'], pre_model_dict['head.bias']
        pre_model_keys = [k for k, v in pre_model_dict.items()]

        new_model_dict = {}
        for k, v in  model.state_dict().items():
            if k in pre_model_keys or 'unc_filt' in k:
                pre_k = k.replace('.unc_filt', '.weight')
                if pre_model_dict[pre_k].size() != model.state_dict()[k].size():
                    print('load_pretrained: %s from %s to %s' % (k, pre_model_dict[pre_k].size(), model.state_dict()[k].size()))
                    new_model_dict[k] = nn.functional.interpolate(
                        pre_model_dict[pre_k].unsqueeze(0).unsqueeze(0).float(), 
                        size=model.state_dict()[k].size()).squeeze(0).squeeze(0).long()
                else:
                    new_model_dict[k] = pre_model_dict[pre_k]
            else:
                new_model_dict[k] = v
        model.load_state_dict(new_model_dict)

    return model

def setup(args, task, rank, all_task_specific_param):
    # Prepare model
    config = get_config(args.model_type, args.dataset)

    if args.lamb != None:
        config.lamb = args.lamb

    if args.rkr_scale != None:
        config.rkr_scale = args.rkr_scale

    if args.dataset == "cifar100":
        num_classes = [10] * 10
    elif args.dataset == "imagenet":
        num_classes = [100] * 10
    elif args.dataset == "VD":
        # num_classes = [1000, 100, 100, 2, 47, 43, 1623, 10, 101, 102]
        num_classes = [47, 43, 10, 101, 102]

    config.task_num = len(num_classes)

    if 'ResNet' in args.model_type:
        if args.model_type == 'ResNet18_PBG':
            model = ResNet18_PBG(pretrained=False, num_classes=num_classes[task], config=config, task=task)

    elif 'ViT' in args.model_type:
        if args.model_type == 'ViT-B_16_RKRPBG':
            model = VisionTransformer_RKRPBG(config, args.img_size, zero_head=True, num_classes=num_classes, task=task)
            model.load_from(np.load(args.pretrained_dir))
        elif args.model_type == 'ViT-B_16_PBG':
            model = VisionTransformer_PBG(config, args.img_size, zero_head=True, num_classes=num_classes[task], task=task)

    elif 'Swin' in args.model_type:
        if args.model_type == 'Swin_PBG':
            model = SwinTransformer_PBG(config, args.img_size, num_classes=num_classes[task], task=task)

    if 'RKRPBG' in args.model_type or task == 0:
        model = model_initialization(args, model)

    # Distributed training
    if args.local_rank != -1:
        model.to(rank)
        # DDP
        model = DDP(
                model,
                device_ids=[rank],
                find_unused_parameters=True,
            )
    else:
        model.to(args.device)

    num_params = count_parameters(model)
    global_param, task_specific_param = count_task_parameters(model)
    all_task_specific_param += task_specific_param

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    # logger.info("Total Parameter: \t%2.1fM" % num_params)
    logger.info("Global Parameter: \t%d" % global_param)
    logger.info("Task Specific Parameter: \t%d" % task_specific_param)
    logger.info("All Task Specific Parameter: \t%d" % all_task_specific_param)
    logger.info("Total Parameter: \t%d" % (global_param + all_task_specific_param))
    return args, model, all_task_specific_param, config


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params

def count_task_parameters(model):
    task_specific_params = []
    global_params = []
    for name, param in model.named_parameters():
        if 'SFG_F' in name:
            logger.info("Task Specific : %s" % name)
            task_specific_params.append(param.numel())
        elif 'head' in name:
            logger.info("Task Specific : %s" % name)
            task_specific_params.append(param.numel())
        elif 'weights_mat' in name:
            logger.info("Task Specific : %s" % name)
            task_specific_params.append(param.numel())
        elif 'unc_filt' in name and name not in ['concat_unc_filter', 'SFG_concat_unc_filter']:
            logger.info("Task Specific : %s" % name)
            task_specific_params.append(param.numel())
        else:
            logger.info("Gloabl : %s" % name)
            global_params.append(param.numel())
    task_specific_param = sum(task_specific_params)
    global_param = sum(global_params)
    return global_param, task_specific_param


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def valid(args, model, test_loader, global_step, task, rank):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("\n***** Running Validation *****")
    
    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                        #   disable=args.local_rank not in [-1, 0]
                          )
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        if args.local_rank != -1:
            batch = tuple(t.to(rank) for t in batch)
            # batch = tuple(t.to(args.device) for t in batch)
        else:
            batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x, task=task)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    # 他のノードから集める
    if args.local_rank != -1:
        dist.all_reduce(torch.tensor(accuracy).to(rank), op=dist.ReduceOp.SUM)
        # dist.all_reduce(torch.tensor(accuracy).to(dist.get_rank()), op=dist.ReduceOp.SUM)
        dist.barrier()
    print("\n")
    logger.info("Task%d Validation Results" % task)
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    # writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, rank=None):
    """ Train the model """
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    # train_loader, test_loader = get_loader(args)
    if args.dataset == 'cifar100':
        train_loader_list, test_loader_list, classes_list  = get_loader_splitCifar100(args, split_num=10, rank=rank)
    elif args.dataset == 'imagenet':
        train_loader_list, test_loader_list, classes_list  = get_loader_splitImagenet(args, split_num=10)
    elif args.dataset == 'VD':
        train_loader_list, test_loader_list, classes_list  = get_VD_loader(args)

    all_task_specific_param = 0
    for task in range(args.start_task, 10):

        if task == 1:
            pre_model_dict = model.state_dict()

        args, model, all_task_specific_param, config = setup(
            args, 
            task, 
            (None if args.local_rank == -1 else rank), 
            all_task_specific_param)

        if task != 0:
            pre_model_keys = [k for k, v in pre_model_dict.items()]
            new_model_dict = {}
            for k, v in model.state_dict().items():
                if k in pre_model_keys and 'unc_filt' not in k and 'weights_mat' not in k and 'head' not in k:
                    new_model_dict[k] = pre_model_dict[k]
                else:
                    print(k)
                    new_model_dict[k] = v
            model.load_state_dict(new_model_dict)

        train_loader = train_loader_list[task]
        test_loader = test_loader_list[task]

        print('----------------------------')
        # for name, param in model.named_parameters():
        #     if 'concat_unc_filt' in name:
        #         print(name)

        if task != 0 and args.lamb != 1.:
            
            idx = 0

            for name, layer in model.named_modules():
                if isinstance(layer, PiggybackConv):
                    # print(name)
                    layer.concat_unc_filter = torch.cat(unc_filt_list[idx], dim=0)
                    idx += 1

            # if 'ResNet' in args.model_type:
            #     for name, layer in model.named_modules():
            #         if isinstance(layer, PiggybackConv):
            #             # print(name)
            #             layer.concat_unc_filter = torch.cat(unc_filt_list[idx], dim=0)
            #             idx += 1

            # else:
            #     # 後でresnetのように簡単に書き換える
            #     layer_list = list(model.transformer.encoder.layer)
            #     # print(layer_list)
            #     for layer in layer_list:
            #         for mdls in layer.modules():
            #             for mdl in mdls.modules():
            #                 if isinstance(mdl, PiggybackFC):
            #                     mdl.concat_unc_filter = torch.cat(unc_filt_list[idx], dim=0)
            #                     idx += 1

        print('----------------------------')

        if args.model_type == 'ViT-B_16_RKRPBG':
            if task == 0: # 試しに
                for name, param in model.named_parameters():
                    param.requires_grad = True
            else:
                for name, param in model.named_parameters():
                    param.requires_grad = False
                    if 'SFG_F' in name:
                        param.requires_grad = True
                    if 'head' in name:
                        param.requires_grad = True
                    if 'weights_mat' in name:
                        param.requires_grad = True
                    if 'unc_filt' in name:
                        param.requires_grad = True
        elif args.model_type in ['ResNet18_PBG', 'ViT-B_16_PBG', 'Swin_PBG']:
            if task == 0:
                for name, param in model.named_parameters():
                    param.requires_grad = True
            else:
                for name, param in model.named_parameters():
                    param.requires_grad = False
                    if 'head' in name:
                        param.requires_grad = True
                    if 'weights_mat' in name:
                        param.requires_grad = True
                    if 'unc_filt' in name and 'concat_unc_filter' not in name:
                        param.requires_grad = True

        for name, param in model.named_parameters():
            if param.requires_grad == True:
                logger.info(name)

        # Prepare optimizer and scheduler
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
        
        t_total = args.num_steps
        if args.decay_type == "cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        else:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

        # Train!
        logger.info("\n***** Running training *****")
        logger.info("  Total optimization steps = %d", args.num_steps)
        logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (
                        torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

        logger.info("  Eval steps = %d", len(test_loader))
        logger.info("  Eval Batch size = %d", args.eval_batch_size)

        optimizer.zero_grad()
        model.zero_grad()
        set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
        losses = AverageMeter()
        global_step, best_acc = 0, 0
        while True:
            if args.local_rank != -1:
                dist.barrier()
            model.train()
            epoch_iterator = tqdm(train_loader,
                                desc="Training (X / X Steps) (loss=X.X)",
                                bar_format="{l_bar}{r_bar}",
                                dynamic_ncols=True,
                                # disable=args.local_rank not in [-1, 0]
                                )
            for step, batch in enumerate(epoch_iterator):
                if args.local_rank != -1:
                    batch = tuple(t.to(rank) for t in batch)
                else:
                    batch = tuple(t.to(args.device) for t in batch)
                x, y = batch
                loss = model(x, y, task)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    losses.update(loss.item()*args.gradient_accumulation_steps)
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    epoch_iterator.set_description(
                        "Training%d (%d / %d Steps) (loss=%2.5f)" % (task, global_step, t_total, losses.val)
                    )
                    # if args.local_rank in [-1, 0]:
                    #     writer.add_scalar("train/loss{}".format(task), scalar_value=losses.val, global_step=global_step)
                    #     writer.add_scalar("train/lr{}".format(task), scalar_value=scheduler.get_lr()[0], global_step=global_step)
                    if global_step % args.eval_every == 0:
                    # if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                        accuracy = valid(args, model, test_loader, global_step, task, rank)
                        if best_acc < accuracy:
                            # save_model(args, model, task)
                            best_acc = accuracy
                        model.train()

                    if global_step % t_total == 0:
                        break

            losses.reset()
            if global_step % t_total == 0:
                break

        # DDP
        if args.local_rank != -1:
            dist.all_reduce(torch.tensor(best_acc).to(rank), op=dist.ReduceOp.SUM)

        logger.info("Task%d Best Accuracy: \t%f" % (task, best_acc))
        logger.info("End Training!")

        if args.lamb != 1.:
            if args.model_type == 'ViT-B_16_RKRPBG':
                # 非制約フィルタの中身を取り出す
                unc_filt_LM_dic = {}
                unc_filt_RM_dic = {}
                filter_LM_idx = 0
                filter_RM_idx = 0
                for name, param in model.named_parameters():
                    param.requires_grad = False
                    if 'unc_filt_LM' in name:
                        # print(name, '/', filter_LM_idx, param.shape)
                        unc_filt_LM_dic[filter_LM_idx] = param.detach()
                        filter_LM_idx += 1
                    elif 'unc_filt_RM' in name:
                        # print(name, '/', filter_RM_idx, param.shape)
                        unc_filt_RM_dic[filter_RM_idx] = param.detach()
                        filter_RM_idx += 1
                filter_idx = filter_LM_idx

                unc_filter_task_dic = {i: torch.matmul(unc_filt_LM_dic[i], unc_filt_RM_dic[i]) for i in range(filter_idx)}
            
            # elif args.model_type == 'ViT-B_16_PBG':
            #     # 非制約フィルタの中身を取り出す
            #     layer_list = list(model.transformer.encoder.layer)
            #     task_unc_filt_list = []
            #     for layer in layer_list:
            #         for mdls in layer.modules():
            #             for mdl in mdls.modules():
            #                 if isinstance(mdl, PiggybackFC):
            #                     task_unc_filt_list.append(mdl.unc_filt.detach())

            # elif args.model_type == 'ResNet18_PBG':
            #     task_unc_filt_list = []
            #     for name, layer in model.named_modules():
            #         if isinstance(layer, PiggybackConv):
            #             task_unc_filt_list.append(layer.unc_filt.detach())
            else:
                task_unc_filt_list = []
                for name, layer in model.named_modules():
                    if isinstance(layer, PiggybackConv):
                        task_unc_filt_list.append(layer.unc_filt.detach())

            # 非制約フィルタの値をフィルタリストに入れる
            if task == 0:
                unc_filt_list = []
                for task_unc_filt in task_unc_filt_list:
                        unc_filt_list.append([task_unc_filt])
            else:
                for i, task_unc_filt in enumerate(task_unc_filt_list):
                    unc_filt_list[i].append(task_unc_filt)

    # if args.local_rank in [-1, 0]:
    #     writer.close()


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet", 'VD'], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=[
        "ResNet18_PBG",
        "ViT-B_16_PBG", "ViT-B_16_RKRPBG",
        "Swin_PBG", 
        ],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--start_task", default=0, type=int, help="")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--lamb", default=None, type=float, help="")
    parser.add_argument("--rkr_scale", default=None, type=float, help="")

    parser.add_argument('--gpu_id', type=str, default=None, help='gpu id: e.g. 0 1. use -1 for CPU')

    parser.add_argument("--local_rank", type=int, default=0,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=100, # 42
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Set seed  
    set_seed(args)
    
    # parallelしない時
    if args.gpu_id != None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DDP
    args.world_size = args.n_gpu = torch.cuda.device_count()
    args.is_master = args.local_rank == 0

    if args.local_rank != -1:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            print(f"RANK and WORLD_SIZE in environ: {rank}/{args.world_size}")
        else:
            print('The environment variable "RANK" or "WORLD_SIZE" does not exist.')
            sys.exit(1)
        
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)


    # ファイル出力するhandlerを設定
    if args.local_rank == -1 or (args.local_rank != -1 and rank == 0):
        if 'ResNet' in args.model_type:
            dir_model = 'ResNet'
        elif 'ViT' in args.model_type:
            dir_model = 'ViT'
        elif 'Swin' in args.model_type:
            dir_model = 'Swin'

        handler = logging.FileHandler(filename="./logfile/{}/{}.log".format(dir_model, args.name))
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
        logger.addHandler(handler)

    # Setup logging
    if args.local_rank != -1:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO if rank == 0 else logging.WARN)
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Training
    train(args, rank=(None if args.local_rank == -1 else rank))

    if args.local_rank != -1:
        # destrory all processes
        dist.destroy_process_group()

if __name__ == "__main__":
    main()