'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-09 18:08:41
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-12 20:11:55
FilePath: /LS-PLL-Reproduction/codes/utils.py
Description: Utils used not related to the experiments
'''
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def validate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def extract_features(model, dataset, batch_size=128):
    model.eval()
    features, labels = [], []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model.get_features(x)  
            features.append(out.cpu())
            labels.append(y.argmax(dim=1) if y.ndim > 1 else y) 
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def tsne_plot(features, labels, save_path, ylabel=None):
    tsne = TSNE(n_components=2, init='pca', perplexity=30, max_iter=3000, random_state=seed)
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(4, 4))
    sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=labels, palette='tab10', s=10, linewidth=0)
    # plt.title(title, fontsize=10)
    if ylabel: plt.ylabel(ylabel, fontsize=20)
    plt.xticks([]); plt.yticks([]); plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_grid(image_paths, titles, rows, cols, save_path):
    _, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    for i, ax in enumerate(axes.flatten()):
        img = mpimg.imread(image_paths[i])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(titles[i], fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

