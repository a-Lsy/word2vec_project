#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立的训练脚本
"""

import os
import jieba
from gensim.models import Word2Vec


def train_from_file(file_path, vector_size=100, window=5, min_count=1, save_path='word2vec_model.model'):
    """从文件训练Word2Vec模型"""
    
    # 读取文件
    if not os.path.exists(file_path):
        print(f"文件不存在：{file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 分词
    words = jieba.lcut(text)
    words = [w.strip() for w in words if w.strip() and w.strip() not in [' ', '\n', '\t']]
    
    # 构建句子（简单处理）
    sentences = [words]
    
    # 训练模型
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        workers=4,
        epochs=100
    )
    
    # 保存模型
    model.save(save_path)
    print(f"模型已保存到：{save_path}")
    
    return model


if __name__ == "__main__":
    # 使用示例
    model = train_from_file('data/text.txt')
    
    if model:
        print(f"词汇表：{list(model.wv.key_to_index.keys())}")