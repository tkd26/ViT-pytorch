import logging
import sys
from tqdm import tqdm

import torch
from torch.functional import split
import torchvision
from torchvision import transforms, datasets
import pickle

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

logger = logging.getLogger(__name__)



def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        torch.distributed.barrier()

    # train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset, rank=args.local_rank, num_replicas=args.world_size)
    # test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset, rank=args.local_rank, shuffle=False, num_replicas=args.world_size)
    train_sampler = DistributedSampler(trainset, rank=args.local_rank, num_replicas=args.world_size)
    test_sampler = DistributedSampler(testset, rank=args.local_rank, shuffle=False, num_replicas=args.world_size)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader

class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, origin, transform, class_id):
        self.transform = transform
        self.dataset = [[data[0], data[1]] for data in origin if data[1] in class_id]
        for i in range(len(self.dataset)):
            before = self.dataset[i][1]
            self.dataset[i][1] = int((class_id == self.dataset[i][1]).nonzero().squeeze())
        self.datanum = len(self.dataset)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data, out_label = self.dataset[idx][0], self.dataset[idx][1]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label

def get_loader_splitCifar100(args, split_num):

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                     
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    class_id_list = []
    classes_list = []
    trainset_list = []
    trainloader_list = []
    testset_list = []
    testloader_list = []
    
    classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]

    trainset_origin = torchvision.datasets.CIFAR100(root = './data', train = True, download = False)
    testset_origin = torchvision.datasets.CIFAR100(root = './data', train = False, download = False)

    class_id_list = torch.chunk(torch.randperm(100), split_num)

    for i in range(split_num):
        # if i > 1:
        #     continue
        classes_list.append([classes[int(id)] for id in class_id_list[i]])

        trainset = Mydatasets(origin=trainset_origin, transform=train_transforms, class_id=class_id_list[i])
        testset = Mydatasets(origin=testset_origin, transform=test_transforms, class_id=class_id_list[i])
        
        trainset_list.append(trainset)
        testset_list.append(testset)

        # train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset, rank=args.local_rank)
        # test_sampler = SequentialSampler(testset)
        train_sampler = DistributedSampler(trainset, rank=args.local_rank, num_replicas=args.world_size)
        test_sampler = DistributedSampler(testset, rank=args.local_rank, shuffle=False, num_replicas=args.world_size)

        trainloader_list.append(
            DataLoader(
                trainset, 
                batch_size=args.train_batch_size, 
                # shuffle=True, 
                sampler=train_sampler,
                num_workers=4, 
                pin_memory=True))

        testloader_list.append(
            DataLoader(
                testset, 
                batch_size=args.eval_batch_size, 
                # shuffle=False, 
                sampler=test_sampler,
                num_workers=4, 
                pin_memory=True))
        print('task{} dataset loaded'.format(i))

    return trainloader_list, testloader_list, classes_list


def get_loader_splitImagenet(args, split_num):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        normalize,
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    class_id_list = []
    classes_list = []
    trainset_list = []
    trainloader_list = []
    testset_list = []
    testloader_list = []
    
    classes = [None] * 1000

    train_path = '/home/yanai-lab/takeda-m/space0/dataset/decathlon-1.0/data/imagenet12/space0/split_train/'
    test_path = '/home/yanai-lab/takeda-m/space0/dataset/decathlon-1.0/data/imagenet12/space0/split_val/'

    # print('loading imagenet...')
    # trainset = datasets.ImageFolder(train_path + '/train')
    # print('loading imagenet...')
    # testset = datasets.ImageFolder(test_path + '/val')
    # print('loaded imagenet')

    # trainset = imagefolder_to_datasets(trainset)
    # testset = imagefolder_to_datasets(testset)

    class_id_list = torch.chunk(torch.Tensor([i for i in range(1000)]), split_num)

    for i in range(split_num):
        classes_list.append([classes[int(id)] for id in class_id_list[i]])

        train_root = train_path + str(i)
        trainset_list.append(datasets.ImageFolder(root=train_root, transform=train_transforms))
        trainloader_list.append(DataLoader(trainset_list[i], batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True))
        
        test_root = test_path + str(i)
        testset_list.append(datasets.ImageFolder(root=test_root, transform=test_transforms))
        testloader_list.append(DataLoader(testset_list[i], batch_size=args.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True))
        print('task{} dataset loaded'.format(i))

    return trainloader_list, testloader_list, classes_list


def data_transform(args, data_path, name, train=True):
    with open(data_path + 'decathlon_mean_std.pickle', 'rb') as handle:
        dict_mean_std = pickle._Unpickler(handle)
        dict_mean_std.encoding = 'latin1'
        dict_mean_std = dict_mean_std.load()

    means = dict_mean_std[name + 'mean']
    stds = dict_mean_std[name + 'std']

    if name in ['gtsrb', 'omniglot', 'svhn']:  # no horz flip
        transform_train = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    if name in ['gtsrb', 'omniglot', 'svhn']:  # no horz flip
        transform_test = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])
    if train:
        return transform_train
    else:
        return transform_test

def get_VD_loader(args):
    data_path = '/home/yanai-lab/takeda-m/space0/dataset/decathlon-1.0/data/'
    trainloader_list = []
    testloader_list = []
    classes_list = []

    do_task_list = ['imagenet12', 'aircraft', 'cifar100', 'daimlerpedcls', 'dtd', 'gtsrb', 'omniglot', 'svhn', 'ucf101', 'vgg-flowers']
    for i, task_name in enumerate(do_task_list):
        if i < args.start_task:
            trainloader = []
            testloader = []
        else:
            trainloader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(data_path + task_name + '/train',
                                                        transform=data_transform(args, data_path, task_name)),
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        num_workers=4, pin_memory=True)
            testloader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(data_path + task_name + '/val',
                                                        transform=data_transform(args, data_path, task_name, train=False)),
                                                        batch_size=args.eval_batch_size,
                                                        shuffle=False,
                                                    num_workers=4, pin_memory=True)

        trainloader_list += [trainloader]
        testloader_list += [testloader]
        print('{} dataset loaded'.format(task_name))

    return trainloader_list, testloader_list, classes_list