import os
import subprocess

# 指定目录路径
directory_path = '/mnt/d/Temp/cs5344rec/new'

# 遍历目录下的所有文件
for filename in os.listdir(directory_path):
    if filename.endswith('.srt'):
        # 构造完整的文件路径
        file_path = os.path.join(directory_path, filename)

        # 运行 translate-srt.py 脚本
        subprocess.run(['python', 'translate-srt.py', file_path])
