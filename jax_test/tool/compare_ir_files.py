#!/usr/bin/env python3
"""
比较 tmp/xla_dump 目录下相邻的 IR 文件是否有不同
如果有不同则输出序号大的那个文件名称
"""

import os
import re
from pathlib import Path

def compare_ir_files(dump_dir):
    """
    比较相邻的 IR 文件并输出有差异的文件
    
    Args:
        dump_dir: IR 文件所在目录
    """
    # 构建文件路径
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        print(f"目录不存在: {dump_dir}")
        return
    
    # 收集符合条件的文件
    pattern = r"module_0005\.jit_update\.(\d{4})\..*\.txt"
    files = []
    
    for file in dump_path.iterdir():
        if file.is_file():
            match = re.match(pattern, file.name)
            if match:
                idx = int(match.group(1))
                files.append((idx, file))
    
    # 按序号排序
    files.sort(key=lambda x: x[0])
    
    if len(files) < 2:
        print(f"文件数量不足，只有 {len(files)} 个文件")
        return
    
    print(f"找到 {len(files)} 个符合条件的文件")
    print("开始比较相邻文件...")
    
    # 比较相邻文件
    different_files = []
    
    for i in range(len(files) - 1):
        idx1, file1 = files[i]
        idx2, file2 = files[i + 1]
        
        # 读取文件内容
        try:
            with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
                content1 = f1.read()
                content2 = f2.read()
                
                # 比较内容
                if content1 != content2:
                    different_files.append((idx2, file2.name))
                    print(f"文件 {file1.name} 和 {file2.name} 有差异")
        except Exception as e:
            print(f"读取文件时出错: {e}")
    
    # 输出结果
    print("\n=== 比较结果 ===")
    if different_files:
        print("有差异的文件 (序号大的文件):")
        for idx, filename in sorted(different_files):
            print(f"  {filename}")
    else:
        print("所有相邻文件内容都相同")

if __name__ == "__main__":
    os.system("clear")
    dump_dir = "/data3/zhongzhw/code/google/jax/tmp/xla_dump"
    compare_ir_files(dump_dir)
