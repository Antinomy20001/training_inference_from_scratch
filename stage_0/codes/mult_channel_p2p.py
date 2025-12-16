# multi_backend_alltoall.py
import torch
import torch.distributed as dist
import os

def main():
    # 初始化全局进程组（NCCL 作为主后端）
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')
    
    print(f"[Rank {rank}] Initialized on {device}, world_size={world_size}")
    
    # ========== 创建多后端进程组 ==========
    # 组1: GPU 通信组（全连接，NCCL）
    gpu_ranks = list(range(world_size))
    gpu_group = dist.new_group(ranks=gpu_ranks, backend='nccl')
    
    # 组2: CPU 控制组（仅偶数 rank，Gloo）
    cpu_ranks = [0, 2]
    cpu_group = dist.new_group(ranks=cpu_ranks, backend='gloo')
    
    # ⚠️ 关键约束：所有进程必须参与所有 new_group() 调用，否则会死锁

    # ========== GPU 组 All-to-All (NCCL) ==========
    # 所有进程参与，每个发送 world_size 个元素
    gpu_send = torch.arange(world_size, device=device, dtype=torch.float32) + rank * 10
    gpu_recv = torch.zeros(world_size, device=device, dtype=torch.float32)
    
    print(f"\n[Rank {rank}] GPU all-to-all SEND: {gpu_send.tolist()}")
    dist.all_to_all_single(gpu_recv, gpu_send, group=gpu_group)
    print(f"[Rank {rank}] GPU all-to-all RECV: {gpu_recv.tolist()}")
    
    # 等待 GPU 操作完成，确保数据已就绪
    torch.cuda.synchronize(device)
    
    # ========== CPU 组 All-to-All (Gloo) ==========
    # 仅 cpu_ranks 中的进程参与
    if rank in cpu_ranks:
        cpu_group_size = len(cpu_ranks)
        # 使用 CPU 张量（Gloo 不支持 GPU 张量）
        cpu_send = torch.arange(cpu_group_size, dtype=torch.float32) + rank * 100
        cpu_recv = torch.zeros(cpu_group_size, dtype=torch.float32)
        
        print(f"\n[Rank {rank}] CPU all-to-all SEND: {cpu_send.tolist()}")
        dist.all_to_all_single(cpu_recv, cpu_send, group=cpu_group)
        print(f"[Rank {rank}] CPU all-to-all RECV: {cpu_recv.tolist()}")
    else:
        print(f"\n[Rank {rank}] Not in CPU group, skipping CPU all-to-all")
    
    # ========== 同步与清理 ==========
    # 使用 GPU 组作为全局同步点（所有进程都在其中）
    dist.barrier(group=gpu_group)
    print(f"\n[Rank {rank}] ✅ All operations completed successfully!")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()