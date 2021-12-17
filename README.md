# 飞桨常规赛：GAMMA中央凹检测 11月第2名方案

## 项目描述
> 基于飞桨框架，使用Unet网络进行中央凹的定位；使用热力图作为标签进行端到端的训练

## 项目结构
> 基础函数，模型都放在了work/my_func文件夹下，预训练模型需要自行下载并解压
```
-|work
  -|my_func
    -|__init__.py
    -|data_info.py
    -|data_process.py
    -|dataset.py
    -|model.py
    -|test_lr.npy
    -|test_size.npy
    -|train_lr.npy
    -|train_size.npy
-|model
  -|readme.txt
-README.MD
-main.ipynb
```
## 使用方式
> 相信你的Fans已经看到这里了，快告诉他们如何快速上手这个项目吧~  
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/usercenter)  
B：此处由项目作者进行撰写使用方式。
