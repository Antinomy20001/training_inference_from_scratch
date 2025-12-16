import os
import platform
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup_process(rank, world_size, backend):
    """
    初始化分布式环境

    参数:
        rank: 当前进程的rank
        world_size: 总进程数
        backend: 指定后端 ('gloo', 'nccl')

    返回:
        rank, world_size, pid
    """
    # 设置通信地址
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    # 跨平台网络接口设置 (仅Gloo需要)
    if backend == 'gloo':
        system = platform.system()
        if system == "Darwin":  # macOS
            os.environ['GLOO_SOCKET_IFNAME'] = 'lo0'
        elif system == "Linux":
            os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
        # Windows不需要设置

    # 初始化进程组
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )

    # 如果使用NCCL，设置GPU设备
    if backend == 'nccl':
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requested but CUDA is not available")
        # 单GPU情况下，所有rank共享同一个GPU
        # 多GPU情况下，每个rank使用不同的GPU
        if torch.cuda.device_count() >= world_size:
            torch.cuda.set_device(rank % torch.cuda.device_count())
        else:
            torch.cuda.set_device(0)

    return rank, world_size, os.getpid(), backend


def cleanup():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(backend):
    """根据后端获取设备类型"""
    return torch.device("cuda" if backend == 'nccl' else "cpu")


def run_broadcast(rank, world_size, backend):
    """
    Broadcast操作：rank0的数据广播到所有进程

    预期输出（3个rank）：
    Rank 0 (pid: 12345) of 3 (backend=gloo): Before broadcast: tensor([[10., 10., 10.], [10., 10., 10.], [10., 10., 10.]])
    Rank 1 (pid: 12346) of 3 (backend=gloo): Before broadcast: tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    Rank 2 (pid: 12347) of 3 (backend=gloo): Before broadcast: tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    Rank 0 (pid: 12345) of 3 (backend=gloo): After broadcast: tensor([[10., 10., 10.], [10., 10., 10.], [10., 10., 10.]])
    Rank 1 (pid: 12346) of 3 (backend=gloo): After broadcast: tensor([[10., 10., 10.], [10., 10., 10.], [10., 10., 10.]])
    Rank 2 (pid: 12347) of 3 (backend=gloo): After broadcast: tensor([[10., 10., 10.], [10., 10., 10.], [10., 10., 10.]])
    """
    rank, world_size, pid, backend = setup_process(rank, world_size, backend)
    device = get_device(backend)

    # 只有rank 0有初始数据
    if rank == 0:
        tensor = torch.ones(3, 3, device=device) * 10
    else:
        tensor = torch.zeros(3, 3, device=device)

    print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): Before broadcast: {tensor}")

    # 从rank 0广播数据
    dist.broadcast(tensor, src=0)

    print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): After broadcast: {tensor}")

    cleanup()


def run_scatter(rank, world_size, backend):
    """
    Scatter操作：rank0的数据分散到所有进程

    预期输出（3个rank）：
    Rank 0 (pid: 12345) of 3 (backend=gloo): Before scatter: tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    Rank 1 (pid: 12346) of 3 (backend=gloo): Before scatter: tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    Rank 2 (pid: 12347) of 3 (backend=gloo): Before scatter: tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    Rank 0 (pid: 12345) of 3 (backend=gloo): After scatter: tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    Rank 1 (pid: 12346) of 3 (backend=gloo): After scatter: tensor([[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]])
    Rank 2 (pid: 12347) of 3 (backend=gloo): After scatter: tensor([[3., 3., 3.], [3., 3., 3.], [3., 3., 3.]])
    """
    rank, world_size, pid, backend = setup_process(rank, world_size, backend)
    device = get_device(backend)

    # 准备接收张量
    recv_tensor = torch.zeros(3, 3, device=device)

    # 只有rank 0准备发送数据
    send_tensors = None
    if rank == 0:
        send_tensors = [torch.ones(3, 3, device=device) * (i + 1) for i in range(world_size)]

    print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): Before scatter: {recv_tensor}")

    # 执行scatter
    dist.scatter(recv_tensor, send_tensors, src=0)

    print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): After scatter: {recv_tensor}")

    cleanup()


def run_gather(rank, world_size, backend):
    """
    Gather操作：所有rank的数据收集到rank0

    预期输出（3个rank）：
    Rank 0 (pid: 12345) of 3 (backend=gloo): Before gather: tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    Rank 1 (pid: 12346) of 3 (backend=gloo): Before gather: tensor([[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]])
    Rank 2 (pid: 12347) of 3 (backend=gloo): Before gather: tensor([[3., 3., 3.], [3., 3., 3.], [3., 3., 3.]])
    Rank 0 (pid: 12345) of 3 (backend=gloo): After gather - Received tensors:
        Rank 0  tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
        Rank 1  tensor([[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]])
        Rank 2  tensor([[3., 3., 3.], [3., 3., 3.], [3., 3., 3.]])
    Rank 1 (pid: 12346) of 3 (backend=gloo): After gather: tensor([[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]])
    Rank 2 (pid: 12347) of 3 (backend=gloo): After gather: tensor([[3., 3., 3.], [3., 3., 3.], [3., 3., 3.]])
    """
    rank, world_size, pid, backend = setup_process(rank, world_size, backend)
    device = get_device(backend)

    # 每个rank准备自己的数据
    send_tensor = torch.ones(3, 3, device=device) * (rank + 1)

    # 只有rank 0准备接收缓冲区
    recv_tensors = None
    if rank == 0:
        recv_tensors = [torch.zeros(3, 3, device=device) for _ in range(world_size)]

    print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): Before gather: {send_tensor}")

    # 执行gather
    dist.gather(send_tensor, recv_tensors, dst=0)

    if rank == 0:
        print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): After gather - Received tensors:")
        for i, tensor in enumerate(recv_tensors):
            print(f"    Rank {i} data: {tensor}")
    else:
        print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): After gather: {send_tensor}")

    cleanup()


def run_reduce(rank, world_size, backend):
    """
    Reduce操作：所有rank的数据归约到rank0

    预期输出（3个rank）：
    Rank 0 (pid: 12345) of 3 (backend=gloo): Before reduce: tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    Rank 1 (pid: 12346) of 3 (backend=gloo): Before reduce: tensor([[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]])
    Rank 2 (pid: 12347) of 3 (backend=gloo): Before reduce: tensor([[3., 3., 3.], [3., 3., 3.], [3., 3., 3.]])
    Rank 0 (pid: 12345) of 3 (backend=gloo): After reduce (sum result): tensor([[6., 6., 6.], [6., 6., 6.], [6., 6., 6.]])
    Rank 1 (pid: 12346) of 3 (backend=gloo): After reduce: (non-dst rank, tensor content undefined)
    Rank 2 (pid: 12347) of 3 (backend=gloo): After reduce: (non-dst rank, tensor content undefined)

    注意：仅 dst rank (0) 的结果有定义
    """
    rank, world_size, pid, backend = setup_process(rank, world_size, backend)
    device = get_device(backend)

    # 每个rank准备自己的数据
    tensor = torch.ones(3, 3, device=device) * (rank + 1)

    print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): Before reduce: {tensor}")

    # 执行reduce，将结果保存到rank 0
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): After reduce (sum result): {tensor}")
    else:
        print(
            f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): After reduce: (non-dst rank, tensor content undefined)")

    cleanup()


def run_all_gather(rank, world_size, backend):
    """
    All Gather操作：所有rank的数据收集到所有rank

    预期输出（3个rank）：
    Rank 0 (pid: 12345) of 3 (backend=gloo): Before all_gather: tensor([1., 1.])
    Rank 1 (pid: 12346) of 3 (backend=gloo): Before all_gather: tensor([2., 2.])
    Rank 2 (pid: 12347) of 3 (backend=gloo): Before all_gather: tensor([3., 3.])
    Rank 0 (pid: 12345) of 3 (backend=gloo): After all_gather - Received all tensors:
        Rank 0  tensor([1., 1.])
        Rank 1  tensor([2., 2.])
        Rank 2  tensor([3., 3.])
    Rank 1 (pid: 12346) of 3 (backend=gloo): After all_gather - Received all tensors:
        Rank 0  tensor([1., 1.])
        Rank 1  tensor([2., 2.])
        Rank 2  tensor([3., 3.])
    Rank 2 (pid: 12347) of 3 (backend=gloo): After all_gather - Received all tensors:
        Rank 0  tensor([1., 1.])
        Rank 1  tensor([2., 2.])
        Rank 2  tensor([3., 3.])
    """
    rank, world_size, pid, backend = setup_process(rank, world_size, backend)
    device = get_device(backend)

    # 每个rank准备自己的数据
    send_tensor = torch.ones(2, device=device) * (rank + 1)

    # 准备接收缓冲区
    recv_tensors = [torch.zeros(2, device=device) for _ in range(world_size)]

    print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): Before all_gather: {send_tensor}")

    # 执行all_gather
    dist.all_gather(recv_tensors, send_tensor)

    print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): After all_gather - Received all tensors:")
    for i, tensor in enumerate(recv_tensors):
        print(f"    Rank {i}  {tensor}")

    cleanup()


def run_all_to_all(rank, world_size, backend):
    """
    All to All操作：每个rank将数据分块并交换给其他rank

    注意：此操作仅在 NCCL (GPU) 后端支持。

    预期输出（3个rank，NCCL后端）：
    Rank 0 (pid: 12345) of 3 (backend=nccl): Before all_to_all - Send  [tensor([0., 0.], device='cuda:0'), tensor([1., 1.], device='cuda:0'), tensor([2., 2.], device='cuda:0')]
    Rank 1 (pid: 12346) of 3 (backend=nccl): Before all_to_all - Send  [tensor([3., 3.], device='cuda:0'), tensor([4., 4.], device='cuda:0'), tensor([5., 5.], device='cuda:0')]
    Rank 2 (pid: 12347) of 3 (backend=nccl): Before all_to_all - Send  [tensor([6., 6.], device='cuda:0'), tensor([7., 7.], device='cuda:0'), tensor([8., 8.], device='cuda:0')]
    Rank 0 (pid: 12345) of 3 (backend=nccl): After all_to_all - Received data: [tensor([0., 0.], device='cuda:0'), tensor([3., 3.], device='cuda:0'), tensor([6., 6.], device='cuda:0')]
    Rank 1 (pid: 12346) of 3 (backend=nccl): After all_to_all - Received  [tensor([1., 1.], device='cuda:0'), tensor([4., 4.], device='cuda:0'), tensor([7., 7.], device='cuda:0')]
    Rank 2 (pid: 12347) of 3 (backend=nccl): After all_to_all - Received  [tensor([2., 2.], device='cuda:0'), tensor([5., 5.], device='cuda:0'), tensor([8., 8.], device='cuda:0')]

    Gloo 后端不支持此操作，将直接抛出异常。
    """
    rank, world_size, pid, backend = setup_process(rank, world_size, backend)
    device = get_device(backend)

    # 每个rank准备world_size个分块的数据
    send_tensors = [torch.ones(2, device=device) * (rank * world_size + i) for i in range(world_size)]
    recv_tensors = [torch.zeros(2, device=device) for _ in range(world_size)]

    print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): Before all_to_all - Send  {send_tensors}")

    # 检查后端支持
    if backend == 'gloo':
        cleanup()  # 清理已初始化的进程组
        raise RuntimeError("Gloo backend does not support all_to_all operation. Use NCCL backend with GPU instead.")

    # NCCL后端支持内置all_to_all
    dist.all_to_all(recv_tensors, send_tensors)
    print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): After all_to_all - Received  {recv_tensors}")

    cleanup()


def run_reduce_scatter(rank, world_size, backend):
    """
    ReduceScatter操作：先归约再分散，每个rank只得到部分结果

    预期输出（3个rank）：
    Rank 0 (pid: 12345) of 3 (backend=gloo): Before reduce_scatter - Input: [tensor([1., 1.]), tensor([2., 2.]), tensor([3., 3.])]
    Rank 1 (pid: 12346) of 3 (backend=gloo): Before reduce_scatter - Input: [tensor([2., 2.]), tensor([3., 3.]), tensor([4., 4.])]
    Rank 2 (pid: 12347) of 3 (backend=gloo): Before reduce_scatter - Input: [tensor([3., 3.]), tensor([4., 4.]), tensor([5., 5.])]
    Rank 0 (pid: 12345) of 3 (backend=gloo): After reduce_scatter - Output: tensor([6., 6.])  # (1+2+3)
    Rank 1 (pid: 12346) of 3 (backend=gloo): After reduce_scatter - Output: tensor([9., 9.])  # (2+3+4)
    Rank 2 (pid: 12347) of 3 (backend=gloo): After reduce_scatter - Output: tensor([12., 12.]) # (3+4+5)
    """
    rank, world_size, pid, backend = setup_process(rank, world_size, backend)
    device = get_device(backend)

    # 每个rank准备完整的数据
    input_tensors = [torch.ones(2, device=device) * (rank + 1 + i) for i in range(world_size)]
    output_tensor = torch.zeros(2, device=device)

    print(
        f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): Before reduce_scatter - Input: {input_tensors}")

    # 执行reduce_scatter
    dist.reduce_scatter(output_tensor, input_tensors, op=dist.ReduceOp.SUM)

    print(
        f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): After reduce_scatter - Output: {output_tensor}")

    cleanup()


def run_all_reduce(rank, world_size, backend):
    """
    All Reduce操作：所有rank的数据归约后广播给所有rank

    预期输出（3个rank）：
    Rank 0 (pid: 12345) of 3 (backend=gloo): Before all_reduce: tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    Rank 1 (pid: 12346) of 3 (backend=gloo): Before all_reduce: tensor([[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]])
    Rank 2 (pid: 12347) of 3 (backend=gloo): Before all_reduce: tensor([[3., 3., 3.], [3., 3., 3.], [3., 3., 3.]])
    Rank 0 (pid: 12345) of 3 (backend=gloo): After all_reduce (sum): tensor([[6., 6., 6.], [6., 6., 6.], [6., 6., 6.]])
    Rank 1 (pid: 12346) of 3 (backend=gloo): After all_reduce (sum): tensor([[6., 6., 6.], [6., 6., 6.], [6., 6., 6.]])
    Rank 2 (pid: 12347) of 3 (backend=gloo): After all_reduce (sum): tensor([[6., 6., 6.], [6., 6., 6.], [6., 6., 6.]])
    """
    rank, world_size, pid, backend = setup_process(rank, world_size, backend)
    device = get_device(backend)

    # 每个rank准备自己的数据
    tensor = torch.ones(3, 3, device=device) * (rank + 1)

    print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): Before all_reduce: {tensor}")

    # 执行all_reduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"Rank {rank} (pid: {pid}) of {world_size} (backend={backend}): After all_reduce (sum): {tensor}")

    cleanup()


def check_environment():
    """检查并报告运行环境信息"""
    print("=" * 60)
    print("DISTRIBUTED TRAINING ENVIRONMENT INFORMATION")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"System platform: {platform.system()} {platform.release()}")
    print(f"Python version: {platform.python_version()}")

    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 检查后端支持
    print("\nAvailable backends:")
    print(f"  Gloo available: {dist.is_gloo_available()}")

    nccl_available = False
    try:
        nccl_available = dist.is_nccl_available()
    except Exception as e:
        print(f"  NCCL availability check failed: {str(e)}")

    print(f"  NCCL available: {nccl_available}")

    # 提供使用建议
    print("\nRECOMMENDATIONS:")
    if cuda_available and nccl_available:
        print("  - For GPU operations (including all_to_all): use 'nccl' backend")
    print("  - For CPU-only operations on this system: use 'gloo' backend")

    print("=" * 60)


if __name__ == "__main__":
    # 检查环境
    check_environment()


    backend = 'gloo'  # 或 'nccl'，用户需要根据环境手动设置
    print(f"\nSelected backend: {backend}")

    world_size = 3
    processes = []

    # 配置要运行的集合通信操作
    operations = [
        # run_broadcast,
        # run_scatter,
        # run_gather,
        # run_reduce,
        # run_all_gather,
        # run_all_to_all, # 只能nccv
        # run_reduce_scatter,
        run_all_reduce
    ]

    # 依次运行选定的操作
    for operation in operations:
        print(f"\n\n{'#' * 70}")
        print(f"## RUNNING OPERATION: {operation.__name__} (backend={backend})")
        print(f"{'#' * 70}\n")

        try:
            # 启动多个进程
            for rank in range(world_size):
                p = mp.Process(target=operation, args=(rank, world_size, backend))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
            processes = []
        except Exception as e:
            print(f"\n❌ ERROR during execution: {str(e)}")
            # 清理可能残余的进程
            for p in processes:
                if p.is_alive():
                    p.terminate()
            processes = []

    print("\n✅ Operation execution completed!")