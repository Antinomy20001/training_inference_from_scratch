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
import torch.profiler as profiler
from torch.autograd.profiler import record_function

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
    # ----------  缺变量时自动补单机单卡 ----------
    os.environ.setdefault("RANK",        "0")
    os.environ.setdefault("WORLD_SIZE",  "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12355")
    # ---------------------------------------------
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    rank        = dist.get_rank()
    world_size  = dist.get_world_size()
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return backend, rank, world_size, local_rank, device

def warmup_model(model, loader, criterion, optimizer, device):
    (images, targets) = next(loader)
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    outputs = model(images)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    loader = iter(loader)
    warmup_model(model, loader, criterion, optimizer, device)

    prof = profiler.profile(
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    )

    with prof:
        for step, (images, targets) in enumerate(loader):
            with record_function('data host to device'):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                torch.cuda.synchronize()

            with record_function('optimizer.zero_grad'):
                optimizer.zero_grad()
                torch.cuda.synchronize()

            outputs = model(images)
            loss = criterion(outputs, targets)
            torch.cuda.synchronize()
            
            with record_function('bwd + optimizer.step'):
                loss.backward()
                dist.barrier()
                
                optimizer.step()
                dist.barrier()
                
                torch.cuda.synchronize()
            
            if step > 2:
                break

    prof.export_chrome_trace(f"/tmp/trace_{dist.get_rank()}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default="/tmp/data")
    args, _ = parser.parse_known_args()

    set_seed(args.seed)

    torch.backends.cudnn.benchmark = True

    _, rank, world_size, local_rank, device = setup_dist()
    pin_memory = device.type == "cuda"

    train_dataset, _ = get_datasets_ddp(args.data_dir, rank)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=pin_memory)

    model = build_model().to(device)
    ddp_model = DDP(
        model, 
        device_ids=[local_rank] if device.type == "cuda" else None, 
        output_device=local_rank if device.type == "cuda" else None,
        bucket_cap_mb=1024 * 1024 * 114514,  # no bucket
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)

    train_one_epoch(
        ddp_model, 
        train_loader, 
        criterion, 
        optimizer,
        device
    )
        
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
