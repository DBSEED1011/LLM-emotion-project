import sys
import subprocess
import os
import multiprocessing
import time

subjnum_list = range(0, 2)

file_paths = [
    f'generate_character_prompt.py',
    f'generate_game_setting_prompt.py',
    f'multi_round_person.py'
]

def modify_and_generate_new_file(file_path, new_value):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        modified = False
        for i, line in enumerate(lines):
            if line.strip().startswith('subjnum ='):
                lines[i] = f'subjnum = {new_value}\n'
                modified = True
                break

        if not modified:
            print(f"no subjnum{file_path}")
            return None

        new_file_path = f"prompt/{new_value}_{os.path.basename(file_path)}"

        with open(new_file_path, 'w', encoding='utf-8') as new_file:
            new_file.writelines(lines)

        return new_file_path
    except FileNotFoundError:
        print(f"no file: {file_path}")
    except Exception as e:
        print(f"error {file_path}, error message: {e}")
    return None

def execute_file(file_path):
    try:
        result = subprocess.run([sys.executable, file_path], check=True, capture_output=True, text=True)
        if result.stderr:
            print(f"error\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"error {file_path}, error message: {e}")

def process_subjnum(new_subjnum):
    print(f"Process subjnum = {new_subjnum}")
    new_file_paths = []
    for file_path in file_paths[:2]:  
        new_file_path = modify_and_generate_new_file(file_path, new_subjnum)
        if new_file_path:
            new_file_paths.append(new_file_path)
    for new_file_path in new_file_paths:
        execute_file(new_file_path)
    
    third_new_file_path = modify_and_generate_new_file(file_paths[2], new_subjnum)
    if third_new_file_path:
        execute_file(third_new_file_path)

    print(f"Finish subjnum = {new_subjnum}")


if __name__ == "__main__":
    max_cpu = multiprocessing.cpu_count()  
    processes = []
    for subjnum in subjnum_list:
        while len(processes) >= max_cpu:
            for p in processes:
                if not p.is_alive():
                    p.join()
                    processes.remove(p)
            time.sleep(0.1)  

        p = multiprocessing.Process(target=process_subjnum, args=(subjnum,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()