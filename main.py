#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Word2Vec词向量训练与可视化项目
功能：训练词向量并展示10个指定词的向量分布
"""

import os
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def prepare_data():
    """准备训练数据"""
    # 创建示例数据（如果data目录下没有text.txt，则使用默认数据）
    if not os.path.exists('data/text.txt'):
        os.makedirs('data', exist_ok=True)
        # 示例中文文本数据
        sample_text = """
        人工智能 机器学习 深度学习 神经网络 自然语言处理
        计算机视觉 图像识别 语音识别 推荐系统 数据挖掘
        大数据 云计算 区块链 物联网 边缘计算
        编程语言 Python Java C++ JavaScript Go
        软件开发 前端开发 后端开发 全栈开发 移动开发
        数据结构 算法 操作系统 计算机网络 数据库
        苹果 香蕉 橙子 葡萄 西瓜 草莓 芒果 菠萝
        汽车 火车 飞机 轮船 自行车 摩托车 公交车
        猫 狗 兔子 老鼠 鸟 鱼 蛇 大象 长颈鹿
        红色 蓝色 绿色 黄色 紫色 橙色 粉色 黑色 白色
        """
        with open('data/text.txt', 'w', encoding='utf-8') as f:
            f.write(sample_text)
    
    # 读取并分词
    with open('data/text.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 使用jieba分词
    words = jieba.lcut(text)
    
    # 去除空格和换行符
    words = [w.strip() for w in words if w.strip() and w.strip() not in [' ', '\n', '\t']]
    
    # 构建句子列表（这里简单地将所有词作为一个句子）
    sentences = [words]
    
    return sentences


def train_word2vec(sentences, vector_size=100, window=5, min_count=1):
    """训练Word2Vec模型"""
    print("正在训练Word2Vec模型...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,  # 使用Skip-gram
        workers=4,
        epochs=100
    )
    print(f"训练完成！词汇表大小：{len(model.wv)}")
    return model


def visualize_word_vectors(model, words_to_show=10):
    """可视化词向量"""
    # 获取词汇表
    vocab = list(model.wv.key_to_index.keys())
    
    # 选择要显示的词（如果词汇表不够，选择所有词）
    if len(vocab) < words_to_show:
        words_to_show = len(vocab)
        words = vocab
    else:
        # 可以选择特定的词，这里选择前words_to_show个
        words = vocab[:words_to_show]
    
    print(f"\n将要展示的词：{words}")
    
    # 获取这些词的向量
    vectors = np.array([model.wv[word] for word in words])
    
    # 使用PCA降维到2维
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # 绘图
    plt.figure(figsize=(12, 10))
    
    # 绘制散点图
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=100, c='lightblue', edgecolors='black', alpha=0.7)
    
    # 添加标签
    for i, word in enumerate(words):
        plt.annotate(word, 
                    xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=12,
                    fontproperties=None,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
    
    plt.title('Word2Vec词向量分布可视化', fontsize=16, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plt.savefig('word_vectors_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n可视化图片已保存为：word_vectors_visualization.png")
    
    return vectors_2d, words


def show_similar_words(model, word, topn=5):
    """展示相似词"""
    try:
        similar = model.wv.most_similar(word, topn=topn)
        print(f"\n与'{word}'最相似的{topn}个词：")
        for w, sim in similar:
            print(f"  {w}: {sim:.4f}")
    except KeyError:
        print(f"\n词'{word}'不在词汇表中")


def main():
    """主函数"""
    print("=" * 50)
    print("Word2Vec词向量训练与可视化项目")
    print("=" * 50)
    
    # 1. 准备数据
    print("\n步骤1：准备数据...")
    sentences = prepare_data()
    print(f"数据准备完成，共{len(sentences)}个句子")
    
    # 2. 训练模型
    print("\n步骤2：训练Word2Vec模型...")
    model = train_word2vec(sentences)
    
    # 3. 保存模型
    print("\n步骤3：保存模型...")
    model.save('word2vec_model.model')
    print("模型已保存为：word2vec_model.model")
    
    # 4. 可视化
    print("\n步骤4：可视化词向量...")
    vectors_2d, words = visualize_word_vectors(model, words_to_show=10)
    
    # 5. 展示相似词示例
    print("\n步骤5：展示相似词示例...")
    if len(words) > 0:
        show_similar_words(model, words[0])
    
    print("\n" + "=" * 50)
    print("项目运行完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()