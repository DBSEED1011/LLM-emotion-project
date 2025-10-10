import sys
import subprocess
import os
import multiprocessing
import time  # 可选，用于演示或节流

# 定义文件路径列表
file_paths = [
    fr'C:/Users/Thinkpadx13/Desktop/agent-trust-multiple/agent-trust-raw/agent_trust/generate_character_prompt.py',
    fr'C:/Users/Thinkpadx13/Desktop/agent-trust-multiple/agent-trust-raw/agent_trust/generate_game_setting_prompt.py',
    fr'C:/Users/Thinkpadx13/Desktop/agent-trust-multiple/agent-trust-raw/agent_trust/multi_round_person_noemotion_gpt3.5.py'
]

# 定义新的subjnum值范围
#subjnum_list = range(300,1017)
subjnum_list = [609, 68, 687, 688, 689, 691, 694, 696, 699, 708, 709, 711]

# 定义一个函数来修改文件中的subjnum值并生成新文件
def modify_and_generate_new_file(file_path, new_value):
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 修改subjnum的值
        modified = False
        for i, line in enumerate(lines):
            if line.strip().startswith('subjnum ='):
                lines[i] = f'subjnum = {new_value}\n'
                modified = True
                break

        if not modified:
            print(f"未找到subjnum赋值行，文件未修改：{file_path}")
            return None

        # 生成新文件路径
        new_file_path = f"agent-trust-raw/agent_trust/prompt/{new_value}_{os.path.basename(file_path)}"
        # 将修改后的内容写入新文件
        with open(new_file_path, 'w', encoding='utf-8') as new_file:
            new_file.writelines(lines)

        print(f"已成功生成新文件：{new_file_path}")
        return new_file_path
    except FileNotFoundError:
        print(f"文件未找到，请检查路径：{file_path}")
    except Exception as e:
        print(f"处理文件时出错：{file_path}，错误信息：{e}")
    return None

# 定义一个函数来执行文件
def execute_file(file_path):
    try:
        print(f"正在执行文件：{file_path}")
        result = subprocess.run([sys.executable, file_path], check=True, capture_output=True, text=True)
        if result.stderr:
            print(f"执行错误：\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"执行文件时出错：{file_path}，错误信息：{e}")

# 定义一个函数来处理每个subjnum值
def process_subjnum(new_subjnum):
    print(f"开始处理subjnum = {new_subjnum}")
    
    # 生成前两个新文件
    new_file_paths = []
    for file_path in file_paths[:2]:  # 只处理前两个文件
        new_file_path = modify_and_generate_new_file(file_path, new_subjnum)
        if new_file_path:
            new_file_paths.append(new_file_path)
    for new_file_path in new_file_paths:
        execute_file(new_file_path)
    
    # 生成第三个新文件
    third_new_file_path = modify_and_generate_new_file(file_paths[2], new_subjnum)
    if third_new_file_path:
        # 执行第三个新文件
        execute_file(third_new_file_path)

    print(f"完成处理subjnum = {new_subjnum}")

# 主函数，使用multiprocessing并行处理每个new_subjnum
if __name__ == "__main__":
    max_cpu = multiprocessing.cpu_count()  # 限制同时运行的进程数（即核数）

    processes = []
    for subjnum in subjnum_list:
        # 如果当前活跃进程数 >= max_cpu，就等待其中一个完成
        while len(processes) >= max_cpu:
            for p in processes:
                if not p.is_alive():
                    p.join()
                    processes.remove(p)
            time.sleep(0.1)  # 避免 CPU 占满（可选）

        # 启动新进程
        p = multiprocessing.Process(target=process_subjnum, args=(subjnum,))
        p.start()
        processes.append(p)

    # 等待所有剩余进程完成
    for p in processes:
        p.join()