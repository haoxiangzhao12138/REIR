import pickle
import open_clip
# import clip
import torch
import tqdm
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from collections import Counter
import spacy

# 加载 spaCy 的英文模型
nlp = spacy.load("en_core_web_sm")


# 定义一个函数，将句子中的名词替换为 "something"
def replace_nouns_with_something(sentence):
    # 使用 spaCy 解析句子
    doc = nlp(sentence)

    # 遍历句子中的单词，如果词性是名词（NOUN 或 PROPN），替换为 "something"
    new_sentence = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN']:  # NOUN 表示普通名词，PROPN 表示专有名词
            new_sentence.append('something')
        else:
            new_sentence.append(token.text)

    return " ".join(new_sentence)


def remove_stopwords_and_replace_nouns(sentence):
    doc = nlp(sentence)
    new_sentence = []
    for token in doc:
        if not token.is_stop:  # 跳过停用词
            if token.pos_ in ['NOUN', 'PROPN']:  # 将名词替换为 "something"
                new_sentence.append('something')
            else:
                new_sentence.append(token.text)
    return " ".join(new_sentence)


anno_paths = ['/home/haoxiangzhao/REIR/dataset/coco/refcoco/refs(unc).p',
              '/home/haoxiangzhao/REIR/dataset/coco/refcoco+/refs(unc).p',
              '/home/haoxiangzhao/REIR/dataset/coco/refcocog/refs(google).p']

annos = []
for anno_path in anno_paths:
    annos = annos + pickle.load(open(anno_path, 'rb'))

device = 'cuda'

sentences = []

for anno in tqdm.tqdm(annos):
    sents = anno['sentences']
    for sent in sents:
        sentences.append(sent['sent'])

random_sentences = random.sample(sentences, 1000)
modified_random_sentences = [replace_nouns_with_something(sentence) for sentence in random_sentences]


cluster_num = 2

clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-H-14',
                                                                               pretrained='/public/haoxiangzhao/weights/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin',
                                                                               force_custom_text=True, )
clip.to(device)
tokenizer = open_clip.get_tokenizer('ViT-H-14')
texts = tokenizer(modified_random_sentences)
texts = texts.to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = clip.encode_text(texts)

# 将 PyTorch 张量转换为 NumPy 数组，方便 KMeans 使用
text_features_np = text_features.cpu().numpy()

# Step 3: 使用 KMeans 对文本特征进行聚类
kmeans = KMeans(n_clusters=cluster_num)  # 例如选择 10 个聚类
kmeans.fit(text_features_np)

tsne = TSNE(n_components=2)
text_features_2d = tsne.fit_transform(text_features_np)

cluster_labels = []
for i in range(cluster_num):  # 假设 10 个聚类
    # 获取属于该聚类的特征点
    cluster_points = text_features_np[kmeans.labels_ == i]

    # 计算每个聚类中心点和聚类内点的欧氏距离
    center = kmeans.cluster_centers_[i]
    distances = euclidean_distances([center], cluster_points)

    # 找到距离最近的点
    closest_idx = np.argmin(distances)

    # 找到这个点对应的文本句子
    closest_sentence = modified_random_sentences[np.where(kmeans.labels_ == i)[0][closest_idx]]

    # 将该文本作为该聚类的标签
    cluster_labels.append(closest_sentence)

# Step 2: 可视化 t-SNE 结果并标注不同的聚类
plt.figure(figsize=(10, 8))
colors = plt.get_cmap("tab10", cluster_num)  # 使用 10 种不同颜色

for i in range(cluster_num):  # 假设 10 个聚类
    cluster_points = text_features_2d[kmeans.labels_ == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=cluster_labels[i], color=colors(i))

plt.title("t-SNE Clustering of Text Features")
plt.legend()
plt.show()

# Step 3: 为每个聚类绘制词云图
for i in range(cluster_num):
    cluster_sentences = [modified_random_sentences[idx] for idx in np.where(kmeans.labels_ == i)[0]]

    cluster_text = " ".join(cluster_sentences)

    # 统计词频
    word_freq = Counter(cluster_text.split())
    for word in list(word_freq.keys()):
        if word in STOPWORDS:
            word_freq.pop(word)

    if len(word_freq) != 0:
        word_freq.pop("something")

    # 生成词云
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    # 显示词云
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(cluster_labels[i])
    plt.show()

cluster_texts = {}

cluster_centers = kmeans.cluster_centers_
temp = torch.tensor(cluster_centers)
torch.save(temp, 'kmeans_cluster_centers.pth')
np.save('kmeans_cluster_centers.npy', cluster_centers)

# 遍历每个聚类标签，按照聚类标签分组文本
for idx, label in enumerate(kmeans.labels_):
    if cluster_labels[label] not in cluster_texts:
        cluster_texts[cluster_labels[label]] = []
    cluster_texts[cluster_labels[label]].append(modified_random_sentences[idx])

# 打印出每个聚类对应的文本
for label, texts in cluster_texts.items():
    print(f"\nCluster {label}:")
    for text in texts:
        print(f" - {text}")

pass
