# Structure Sensitive and Semantic Alignment Synergistic Enhancement  Network for Remote Sensing Change Detection
This repository contains a simple Python implementation of our paper S2ENet.

## Overview
![](https://github.com/ahaha-16/S2ENet/blob/main/S2ENet.png)

## Dataset Preparation
Download datasets [SYSU-CD](https://github.com/liumency/SYSU-CD), CLCD, [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html), and [LEVIR-CD](https://justchenhao.github.io/LEVIR/)

Prepare datasets into the following structure and set their path in train.py and test.py

    
    ├── Train
        ├── A        ...jpg/png
        ├── B        ...jpg/png
        ├── label    ...jpg/png
        └── list     ...txt
     
    ├── Val
        ├── A
        ├── B
        ├── label
        └── list
     
    ├── Test
        ├── A
        ├── B
        ├── label
        └── list
        

