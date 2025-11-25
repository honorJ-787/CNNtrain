import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from wideresnet import WideResNet


def get_args():
    parser = argparse.ArgumentParser("WideResNet CIFAR trainer (clean version)")

    parser.add_argument('--dataset', default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--data', default='./data',
                        help='dataset root directory')

    parser.add_argument('--layers', default=28, type=int,
                        help='WRN depth, e.g. 28')
    parser.add_argument('--widen-factor', default=10, type=int,
                        help='WRN widen factor, e.g. 10')
    parser.add_argument('--droprate', default=0.0, type=float,
                        help='dropout rate')

    # fine-tune: 默认训练 100 轮、lr 0.01
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)

    parser.add_argument('--num-workers', default=0, type=int,
                        help='dataloader workers (0 avoids Windows spawn+lambda 问题)')

    parser.add_argument('--log-dir', default='runs', help='TensorBoard log dir')
    parser.add_argument('--name', default='wrn_cifar', help='exp name')

    # ⭐ 新增：从已有 .pth 继续训练的路径
    parser.add_argument('--resume', type=str, default='',
                        help='path to pretrained .pth to finetune from')

    args = parser.parse_args()
    return args



def get_loaders(args):
    # 官方 CIFAR 均值 / 方差
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    # NEW: 加强数据增强
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.2),
        ratio=(0.3, 3.3),
    ),
])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset == 'cifar10':
        trainset = datasets.CIFAR10(root=args.data, train=True,
                                    download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=args.data, train=False,
                                   download=True, transform=transform_test)
        num_classes = 10
    else:
        trainset = datasets.CIFAR100(root=args.data, train=True,
                                     download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=args.data, train=False,
                                    download=True, transform=transform_test)
        num_classes = 100

    train_loader = data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, num_classes


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer=None):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    start = time.time()
    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)
        running_total += targets.size(0)
        running_correct += preds.eq(targets).sum().item()
        running_loss += loss.item() * targets.size(0)

        if (i + 1) % 50 == 0:
            print(f"[Train] Epoch {epoch} Step {i+1}/{len(loader)} "
                  f"Loss {loss.item():.4f}")

    epoch_loss = running_loss / running_total
    epoch_acc = 100.0 * running_correct / running_total
    elapsed = time.time() - start

    print(f"[Train] Epoch {epoch} Loss {epoch_loss:.4f} Acc {epoch_acc:.2f}% "
          f"Time {elapsed:.1f}s")

    if writer is not None:
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/acc', epoch_acc, epoch)

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, epoch, writer=None):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            _, preds = outputs.max(1)
            running_total += targets.size(0)
            running_correct += preds.eq(targets).sum().item()
            running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = 100.0 * running_correct / running_total
    print(f"[Test ] Epoch {epoch} Loss {epoch_loss:.4f} Acc {epoch_acc:.2f}%")

    if writer is not None:
        writer.add_scalar('val/loss', epoch_loss, epoch)
        writer.add_scalar('val/acc', epoch_acc, epoch)

    return epoch_loss, epoch_acc


def main():
    args = get_args()
    os.makedirs(args.data, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_loader, test_loader, num_classes = get_loaders(args)

    print(f"Creating WideResNet(depth={args.layers}, widen_factor={args.widen_factor}, "
          f"num_classes={num_classes})")
    model = WideResNet(args.layers, num_classes,
                       widen_factor=args.widen_factor,
                       dropRate=args.droprate)
    model = model.to(device)

    # ==== 这里真正加载预训练权重 ====
    pretrained_path = "wrn_cifar_best.pth"   # 文件和 tl.py 在同一文件夹
    if os.path.isfile(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=device)

        # 大概率 ckpt 本身就是 state_dict；如果你以后改成 checkpoint，就走下面这个分支
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict, strict=True)
        print("Loaded pretrained weights.")
    else:
        print(f"*** WARNING: {pretrained_path} not found, training from scratch. ***")
    # ==== 加载结束 ====


    if device.type == 'cuda':
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2
    )

    # TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(args.log_dir, args.name)
        writer = SummaryWriter(log_dir=log_dir)
        print("TensorBoard logs ->", log_dir)
    except Exception as e:
        print("TensorBoard not available, skip logging. (", e, ")")

    best_acc = 0.0
    ckpt_dir = 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print("=" * 60)
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        _, val_acc = evaluate(model, test_loader, criterion, device, epoch, writer)

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(ckpt_dir, f"{args.name}_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"** Saved best model to {ckpt_path}, acc={best_acc:.2f}%")

    print("Finished. Best test acc: {:.2f}%".format(best_acc))

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
