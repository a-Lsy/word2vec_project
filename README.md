# Word2Vec词向量训练与可视化项目

## 项目简介
本项目使用Word2Vec算法训练中文词向量，并通过PCA降维将高维词向量投影到二维空间进行可视化展示。

## 功能特点
- 使用jieba进行中文分词
- 基于gensim实现Word2Vec模型训练
- 支持自定义训练参数
- 使用PCA降维可视化词向量
- 展示10个词向量的二维分布图

## 环境要求
- Python 3.11
- conda虚拟环境

## 安装步骤

### 1. 创建conda虚拟环境
```bash
conda create -n word2vec_env python=3.11
conda activate word2vec_env