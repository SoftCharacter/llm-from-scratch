"""
目录树生成工具


这个脚本用于生成项目的目录结构树状图，方便查看项目文件组织结构。


深度学习小知识：
   - 在深度学习项目中，良好的文件组织结构非常重要
   - 通常包括：
     - 模型代码
     - 训练脚本
     - 数据目录
     - 配置文件
     - 检查点目录
     - 日志目录


使用说明：
   直接运行此脚本会生成当前目录的树状结构，最大深度为3层
"""


import os


def generate_tree(startpath, depth=None):
   """
   生成目录树状图


   参数说明：
       startpath (str): 起始目录路径
       depth (int, optional): 最大深度，None表示无限制


   返回：
       None: 直接打印目录树
   """
   prefix = '│   '
   # 遍历目录树
   for root, dirs, files in os.walk(startpath):
       # 计算当前目录的层级深度
       level = root.replace(startpath, '').count(os.sep)
       # 如果超过最大深度，跳过
       if depth is not None and level > depth:
           continue
       # 计算缩进
       indent = '│   ' * level
       # 打印目录名
       print(f"{indent}├── {os.path.basename(root)}/")
       # 子级缩进
       sub_indent = '│   ' * (level + 1)
       # 打印文件名
       for f in files:
           print(f"{sub_indent}├── {f}")


# 生成当前目录的树状结构，最大深度3层
generate_tree('.', depth=3)
