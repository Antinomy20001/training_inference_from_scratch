import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_datasets_ddp(data_dir: str, rank: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    if rank == 0:
        train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)
        dist.barrier()
    else:
        dist.barrier()
        train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=False)
        test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=False)
    return train_dataset, test_dataset


def setup_dist():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return backend, rank, world_size, local_rank, device


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device, epoch: int):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate_global(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct_local = 0
    total_local = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct_local += (preds == targets).sum().item()
        total_local += targets.size(0)
    correct_tensor = torch.tensor(correct_local, dtype=torch.float64, device=device)
    total_tensor = torch.tensor(total_local, dtype=torch.float64, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    acc = (correct_tensor / total_tensor).item() if total_tensor.item() > 0 else 0.0
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="/tmp/data")
    parser.add_argument("--save-path", type=str, default="/tmp/resnet18_mnist_ddp_multi_node.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    backend, rank, world_size, local_rank, device = setup_dist()
    pin_memory = device.type == "cuda"

    train_dataset, test_dataset = get_datasets_ddp(args.data_dir, rank)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.num_workers, pin_memory=pin_memory)

    model = build_model().to(device)
    ddp_model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, output_device=local_rank if device.type == "cuda" else None)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_one_epoch(ddp_model, train_loader, criterion, optimizer, device, epoch)
        test_acc = evaluate_global(ddp_model, test_loader, device)
        if rank == 0:
            print(f"Epoch {epoch}/{args.epochs} | loss {train_loss:.4f} | train_acc {train_acc:.4f} | test_acc {test_acc:.4f}")

    if rank == 0:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save(ddp_model.module.state_dict(), args.save_path)
        print(f"Saved checkpoint to: {args.save_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
