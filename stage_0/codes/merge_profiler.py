import os
import json

def merge_profiles(json_file_list: list, output_file: str):
    """
    将一个包含多个 PyTorch Profiler JSON 文件路径的列表合并成一个文件。

    该函数会为每个输入文件（代表一个 Rank）创建一个元数据事件，
    以便在 Chrome Tracing 或 Perfetto UI 中能够清晰地看到 "Rank 0", "Rank 1" 等进程名称。

    参数:
        json_file_list (list): 包含所有待合并 JSON 文件路径的字符串列表。
        output_file (str): 合并后输出的 JSON 文件路径。
    """
    if not json_file_list:
        print("错误：输入的 JSON 文件列表为空。")
        return

    print(f"开始合并 {len(json_file_list)} 个 profile 文件...")

    all_trace_events = []

    # 使用 enumerate 来为每个文件自动分配一个 Rank 编号
    for i, file_path in enumerate(json_file_list):
        if not os.path.exists(file_path):
            print(f"警告：文件不存在，已跳过: {file_path}")
            continue

        print(f"  - 正在处理 Rank {i}: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"警告：JSON 解析失败，已跳过文件: {file_path}")
            continue
        except Exception as e:
            print(f"警告：读取文件时发生未知错误，已跳过: {file_path}. 错误: {e}")
            continue

        # 获取当前文件的 trace events 列表
        trace_events = data.get('traceEvents', [])

        # 如果文件没有 trace events，则跳过
        if not trace_events:
            print(f"警告：文件中未找到 'traceEvents' 或列表为空，已跳过: {file_path}")
            continue

        # 找到该 Rank 的主进程 ID (pid)。通常，一个文件内的 pid 是相同的。
        # 我们从第一个事件中获取它。
        try:
            pid = trace_events[0].get('pid')
            if pid is None:
                print(f"警告：在 {file_path} 的第一个事件中未找到 'pid'，无法为其命名。")
                all_trace_events.extend(trace_events) # 仍然添加事件，只是没有命名
                continue
        except IndexError:
            # 这种情况对应于 trace_events 列表为空，上面已经处理过，但为了代码健壮性保留
            continue

        # 创建一个元数据事件 (Metadata Event)，用于命名该 Rank 的进程。
        # 在 UI 中，所有 pid 为 `pid` 的事件都会被归类到 "Rank i" 这个进程下。
        # ph: "M" 表示元数据 (Metadata)
        # name: "process_name" 是元数据类型，表示我们要设置进程名
        # args: {"name": ...} 是具体的进程名
        process_name = f"Rank {i}"
        name_event = {
            "name": "process_name",
            "ph": "M",
            "pid": pid,
            "args": {"name": process_name}
        }

        # 将命名事件插入到当前所有事件的最前面，然后将它们全部添加到总列表中
        all_trace_events.append(name_event)
        all_trace_events.extend(trace_events)

    # 检查是否收集到了任何事件
    if not all_trace_events:
        print("错误：所有文件均为空或无法处理，未生成任何合并数据。")
        return

    # 创建最终的 JSON 结构并写入文件
    merged_data = {'traceEvents': all_trace_events}

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f)
        print(f"\n成功！已将 {len(json_file_list)} 个文件合并到: '{output_file}'")
    except Exception as e:
        print(f"\n错误：写入最终文件失败: {output_file}. 错误: {e}")
    
if __name__ == '__main__':
    import glob

    merge_profiles(
        glob.glob('/tmp/trace_*.json'),
        '/tmp/trace_merged.json',
    )