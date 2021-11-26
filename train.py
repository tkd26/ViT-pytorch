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
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
# from apex.parallel import DistributedDataParallel as DDP

# from models.modeling import VisionTransformer, CONFIGS
from models.modeling_RKR import VisionTransformer
from models.modeling_RKR3_2 import VisionTransformer as VisionTransformer_RKR3_2, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader, get_loader_splitCifar100, get_loader_splitImagenet, get_VD_loader
from utils.dist_util import get_world_size

# DDP
from argparse import ArgumentParser
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

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


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type] # modeling_RKR.pyのCONFIGS

    if args.lamb != None:
        config.lamb = args.lamb

    if args.rkr_scale != None:
        config.rkr_scale = args.rkr_scale

    num_tasks = 10
    if args.dataset == "cifar100":
        num_classes = [10] * 10
    elif args.dataset == "imagenet":
        num_classes = [100] * 10
    elif args.dataset == "VD":
        num_classes = [1000, 100, 100, 2, 47, 43, 1623, 10, 101, 102]

    model = VisionTransformer_RKR3_2(config, args.img_size, zero_head=True, num_classes=num_classes)

    # print(model)
    # print(list(np.load(args.pretrained_dir)))
    # for name, param in model.named_parameters():
    #     print(name)

    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    # Distributed training
    if args.local_rank != -1:
        # DDP
        model = DDP(
                model,
                find_unused_parameters = True,
                device_ids=[args.local_rank]
            )

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def valid(args, model, test_loader, global_step, task):
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
    dist.all_reduce(torch.tensor(accuracy).to(args.device), op=dist.ReduceOp.SUM)
    print("\n")
    logger.info("Task%d Validation Results" % task)
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    # writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, model):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     os.makedirs(args.output_dir, exist_ok=True)
    #     writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    # train_loader, test_loader = get_loader(args)
    if args.dataset == 'cifar100':
        train_loader_list, test_loader_list, classes_list  = get_loader_splitCifar100(args, split_num = 10)
    elif args.dataset == 'imagenet':
        train_loader_list, test_loader_list, classes_list  = get_loader_splitImagenet(args, split_num = 10)
    elif args.dataset == 'VD':
        train_loader_list, test_loader_list, classes_list  = get_VD_loader(args)

    TASK_NUM = 10
    for task in range(args.start_task, TASK_NUM):

        train_loader = train_loader_list[task]
        test_loader = test_loader_list[task]

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

        # if args.fp16:
        #     model, optimizer = amp.initialize(models=model,
        #                                     optimizers=optimizer,
        #                                     opt_level=args.fp16_opt_level)
        #     amp._amp_state.loss_scalers[0]._loss_scale = 2**20
        

        for name, param in model.named_parameters():
            param.requires_grad = False
            if 'F_list' in name or 'RM_list' in name or 'LM_list' in name:
                if name.split('.')[-1] == str(task):
                    param.requires_grad = True
            if 'head' in name:
                if name.split('.')[-2] == str(task):
                    param.requires_grad = True

        # for name, param in model.named_parameters():
        #     if param.requires_grad == True:
        #         logger.info(name)

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

        model.zero_grad()
        set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
        losses = AverageMeter()
        global_step, best_acc = 0, 0
        while True:
            dist.barrier()
            model.train()
            epoch_iterator = tqdm(train_loader,
                                desc="Training (X / X Steps) (loss=X.X)",
                                bar_format="{l_bar}{r_bar}",
                                dynamic_ncols=True,
                                # disable=args.local_rank not in [-1, 0]
                                )
            for step, batch in enumerate(epoch_iterator):
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
                        accuracy = valid(args, model, test_loader, global_step, task)
                        if best_acc < accuracy:
                            # save_model(args, model, task)
                            best_acc = accuracy
                        model.train()

                    dist.barrier()
                    if global_step % t_total == 0:
                        break

            losses.reset()
            if global_step % t_total == 0:
                break

        # DDP
        dist.all_reduce(torch.tensor(best_acc).to(args.device), op=dist.ReduceOp.SUM)

        logger.info("Task%d Best Accuracy: \t%f" % (task, best_acc))
        logger.info("End Training!")

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
        "ViT-B_16", "ViT-B_16_FC", 
        "ViT-B_16_RKR", "ViT-B_16_RKRnoRG", "ViT-B_16_RKRnoSFG",
        "ViT-B_16_RKR3_2",
        "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
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

    parser.add_argument("--lamb", default=None, type=float, help="use only RKR3_2")
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

    if args.gpu_id != None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # DDP
    torch.cuda.set_device(args.local_rank)  
    dist.init_process_group(backend='nccl', init_method='env://')
    args.world_size = args.n_gpu = torch.cuda.device_count()
    args.is_master = args.local_rank == 0


    #handler2を作成
    handler = logging.FileHandler(filename="./logfile/{}.log".format(args.name))  #handler2はファイル出力
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))

    #loggerにハンドラを設定
    logger.addHandler(handler)


    # Setup CUDA, GPU & distributed training
    # if args.local_rank == -1:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     args.n_gpu = torch.cuda.device_count()
    # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend='nccl',
    #                                          timeout=timedelta(minutes=60))
    #     args.n_gpu = torch.cuda.device_count()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
    # destrory all processes
    dist.destroy_process_group()