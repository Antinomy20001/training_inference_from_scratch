# 训练与推理的优化

# 在做什么？

<del>就是优化训练和推理</del>

对AI模型训练和推理过程中，在给定模型的背景下，在给定的硬件资源下，充分榨干硬件，实现某种吞吐或者latency的最大化，并且保证可靠性和数值稳定性；

- 什么是模型？
    - 模型是`模型权重`+`计算方法`
- 什么是训练？
    - 训练是根据给定数据，通过训练算法**更新模型权重**
- 什么是推理？
    - 在已有`模型权重`下，输入数据，执行`计算方法`
    - 所以训练和推理的一大根本区别在于：**`模型权重`有没有得到更新**
    - 由于`模型权重`无需更新，所以用来更新`模型权重`的组件也不存在


# 需要知道什么？

- 模型的计算过程（可以不知道算法或数学意义的过程）
- 给定硬件条件的性能边界（以及什么叫“性能”）
- 已有的加速技术
- 排查性能瓶颈的能力
- 良好的coding功底（做模拟题，把想法快速实现出来，一般不涉及到复杂的语言特性）
- Python基础

# 一些时代背景

1. 现在是5202年
    - 除了LLM（及其衍生mllm、vlm、扩散模型、dit、wm、vla、lrm等）基本可以开除ai籍
    - 除了自回归（token）和diffusion外，其他决策方法可以开除learning籍
    - 不再有所谓静态图和动态图的争论
    - Torch和huggingface已经完成大一统，并且基本作为RFC存在
    - 各硬件层次的通信带宽逐渐成为所有性能问题的根因，越发明显
    - Pythonic思想贯彻大部分场景开发思想
        - 从顶层到底层，**每一层的代码都是对下一层的调度**，每一层都尽量不涉及实际密集运算，只有逻辑控制
        - **因此**，调度代码需要尽量动态灵活，从而快速排查、实现、定位、分析性能和迭代，而涉及具体底层过程使用dlopen的形式引入
        - **因此**，DSL（Domain-Specific Language, 领域特定语言）成为潮流，可以认为torch已经是一种DSL而非框架；simd和cuda等高性能计算编程也逐渐DSL化
        - **因此**，从pythonic的最上层往底层来看，每一层都像是一个compiler，需要通过compiler来降低overhead，以及做IR优化
        - "层"不等于"编程语言层次"
    
2. 分布式并行计算成为标配
    - 除了端侧，几乎不再存在不需要分布式并行推理/训练的场景
    - <del>（甚至端侧也有明显脱离单node趋势）</del>
    - 如何充分利用多硬件节点成为优化的主要问题
    - 由于node的增多，已经不能无视单node出现各种问题的几率
        - 例如，单个node故障率哪怕只有0.01%，在233个node持续以榨干性能的姿态运行长时间的情况下，故障率可以接近100%
    - 分布式共识反而不会成为问题，CAP不可能三角中可以最大化可用性和容忍性，几乎可以不考虑一致性，因为分布式训推的并行策略已经可以事先对数据进行schedule

# Roadmap

## Common

训推都需要知道的知识

### torch的基本使用和单卡最小训练过程

学会torch的官方教程，https://docs.pytorch.org/tutorials/beginner/basics/intro.html

用最新版本pytorch学会跑左边的“Learn the Basics”

别怕，不需要了解具体算法和数学原理，比如：
- 为什么模型要这么设置
- 为什么要有梯度下降
- 梯度是什么
- 反向传播为什么是这样计算，为什么要反向传播
- loss为什么这么算以及loss计算的数学意义

等等，也不需要知道为什么训练模型或者推理模型是这样的过程

学完后，你应该掌握了：
1. torch tensor的基本使用
2. torch的常见操作
3. （单卡）训练需要的组件和概念
    - `dataloader`、模型、`optimizer`、`loss`、`forwoard`、`backward`、`optimizer.step`、模型权重（checkpoint）
4. 所谓的模型是一个有向无环图，这个图叫计算图，每个节点是一个计算过程（`Function`或者说`Layer`或者是`Module`）和输入输出与权重（`Tensor`）和梯度（`grad`），节点之间边的方向是fwd时的运算顺序
    - `数据tensor`为`src`，`loss`为`tgt`
    - 所谓的`forward`(之后简称fwd)是从`src`开始执行到`tgt`，不断执行每个节点的fwd，每个节点的fwd都是前驱节点的fwd结果
    - 所谓的`backward`(之后简称bwd)是从`tgt`执行到`src`，不断执行每个节点的bwd方法，每个bwd的输入是后继节点的bwd结果，后继节点的bwd结果存在当前节点fwd输出tensor（=后继节点fwd输入）的`.grad`中
5. torch看起来是以`torch.nn.Module`作为某种基本组成单元的
6. 如果fwd过程都使用torch有的function，似乎我们没有编写bwd过程，这说明torch帮我们实现了，这也是torch最原始的定位：自动求导框架（autograd）


那么我们可以得到一个最小的训练过程：

![](https://littlenyima-1319014516.cos.ap-beijing.myqcloud.com/blog/2025/08/31/process-of-model-training.jpg)：

```python
dataloader = Dataloader(...) # 负责从某存储读取数据并打包为训练数据，我们先不管他
model = Module(...) # 模型定义，我们先不考虑模型具体形态
loss_fn = Loss(...) # loss函数的定义
optimizer = torch.optim.Adam(model.parameters(), ...) # 注意model.parameters()是要更新的模型权重的集合

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # 从X到loss，这一步叫fwd
        pred = model(X)
        loss = loss_fn(pred, y)

        # torch的autograd机制，从loss在计算图上反向遍历更新每个tensor的.grad
        # 这一步叫bwd
        loss.backward()
        # bwd结束后，计算图的每一个tenso都有了.grad梯度

        # 执行optimizer，根据待更新权重的grad和原数值，计算新数值并写回权重tensor
        optimizer.step()

        # 权重更新完了，清空所有tensor的grad，准备下一轮
        optimizer.zero_grad()

        # 我们把这个训练循环里的单次迭代，叫做“一个训练step”

```

此时，我们完全不需要知道model的算法细节，只需要知道他是一个model


### 拓展到单机多卡和多机多卡

我们已经了解了单卡训练的最小方式，那么我们可以问问伟大的gpt老师，怎样把这个过程拓展到单机多卡（ddp）和多机多卡（ddp），代码可以是一致的

[参考这个训练代码](./codes/ddp.py)，使用方式在代码中

我们可以先看看和单卡最小训练相比，多了什么东西呢？

- 启动方式变了，变成了[torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)，并且输入的时候有一些参数
- 变成多进程，看起来一个进程会使用一张卡
- 除了单卡有的之外，还多了[torch distributed](https://docs.pytorch.org/docs/stable/distributed.html)的init等处理
- Dataloader多了[DistributedSampler](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)
- 代码里面多了一堆读取的环境变量，多了`rank`的概念
- 代码里面model包了一层[DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)
    - 这是啥呢

好，我们先到这里，在此之前我们先想办法看看，单卡和多卡的训练step分别发生了什么吧

### 使用profiler工具了解每个训练steo过程发生了什么

#### 跑一个profiler文件

你已经学会跑`ddp.py`了，所以可以把[ddp_with_profiler.py](./codes/ddp_with_profiler.py)也跑了，自己看懂参数跑一下

跑完后会在`/tmp/`下生成若干个`trace_*.json`，然后运行[merge_profiler.py](./code/merge_profiler.py)(ai写的)，会得到`/tmp/trace_merged.json`, 我们把这个merge json丢到[perfetto.dev](https://ui.perfetto.dev/)中打开

我们在timeline的最左边，会发现有stream和thread两种timeline，分别对应gpu和cpu的活动。

cpu活动中会有我们这份代码中ddp的详细堆栈，我们可以放大，在尾巴里找到有`cudaLaunchKernel`的cpu活动，会发现他有一条曲线连在gpu活动，表示对应关系。

然后我们在单卡训练也这样做一个profiler，放到新的perfetto窗口打开，perfetto按`A/D`是左右平移，按`W/S`是按中心大小缩放，更详细的用法以及sql查询请问gpt老师，我们大部分情况下最多只需要按名字搜索某个活动（在perfetto里面叫event）

在`assets`里面分别准备了在H20上的[单卡](../assets/stage_0/perfetto/trace_single.json.gz)和[2卡ddp](../assets/stage_0/perfetto/ddp_2ranks.json.gz)的profiler，我们以这两个为例，放到perfetto后分别形如



### GPU的基本概念

TODO

### 集合通信

TODO

### 3D并行

TODO

### huggingface

一般习惯性把`diffusers`、`transformers`称为“hf”，可以认为是LLM时代的RFC，这两个库有常见LLM的模型实现（modeling）和基本训练过程封装，性能很感人，但是大家的各种优化都会把hf作为对齐效果的目标

我们先以`transformers`为例

TODO
