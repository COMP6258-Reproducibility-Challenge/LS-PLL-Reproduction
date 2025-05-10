'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-06 15:24:13
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-10 21:20:51
FilePath: /LS-PLL-Reproduction/codes/prepare_data.py
Description: The codes to download, train and generate partial labels for datasets.
'''
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import pickle

from ResNet18 import ResNet18
from train import train_model
from utils import device, seed

np.random.seed(seed)


def load_dataset(dataset_name='CIFAR10', dataset_path='../datasets'):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    print(f"**** Loading {dataset_name} dataset ****")
    if dataset_name == 'CIFAR10':
        trainset = datasets.CIFAR10(root=f"{dataset_path}/{dataset_name}", train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=f"{dataset_path}/{dataset_name}", train=False, download=True, transform=transform_test)
    elif dataset_name == 'CIFAR100':
        trainset = datasets.CIFAR100(root=f"{dataset_path}/{dataset_name}", train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=f"{dataset_path}/{dataset_name}", train=False, download=True, transform=transform_test)
    elif dataset_name == 'FashionMNIST':
        trainset = datasets.FashionMNIST(root=f"{dataset_path}/{dataset_name}", train=True, download=True, transform=transform_train)
        testset = datasets.FashionMNIST(root=f"{dataset_path}/{dataset_name}", train=False, download=True, transform=transform_test)
    elif dataset_name == 'KuzushijiMNIST':
        trainset = datasets.KMNIST(root=f"{dataset_path}/{dataset_name}", train=True, download=True, transform=transform_train)
        testset = datasets.KMNIST(root=f"{dataset_path}/{dataset_name}", train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return trainset, testset


def train_dataset_model(
    Model, trainset, testset, 
    num_classes
):
    model, _ = train_model(Model, trainset, testset, num_classes=num_classes)
    return model


def get_topk_predictions(model, dataset, batch_size=128, k=6):
    model.eval()
    with torch.no_grad():
        topk_preds = []
        for inputs, targets in DataLoader(dataset, batch_size=batch_size):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # remove the true label from the top-k predictions
            # by setting the true label's score to -inf
            rows = np.arange(outputs.shape[0])
            masked_outputs = outputs.clone()
            masked_outputs[rows, targets] = -np.inf
            # get the top-k predictions
            _, topk = masked_outputs.topk(k, dim=1)

            topk_preds.append(topk.cpu().numpy())
        topk_preds = np.concatenate(topk_preds, axis=0)
    return topk_preds


def generate_partial_labels(true_labels, topk_preds, avg_cl, k=6, num_classes=10):
    n = len(true_labels)
    cl_values = np.random.randint(0, k, size=n)
    current_sum = cl_values.sum()
    target_sum = (avg_cl-1) * n

    while abs(current_sum - target_sum) > 0.01 * n:
        idx = np.random.randint(0, n)
        diff = target_sum - current_sum
        if diff > 0 and cl_values[idx] < k:
            cl_values[idx] += 1
            current_sum += 1
        elif diff < 0 and cl_values[idx] > 0:
            cl_values[idx] -= 1
            current_sum -= 1
    
    random_mask = np.argsort(np.random.rand(n, k), axis=1) < cl_values.reshape(-1, 1)
    partial_labels = [topk_preds[i, mask] for i, mask in enumerate(random_mask)]
    partial_labels = [np.append(arr, gt) for arr, gt in zip(true_labels, partial_labels)]

    onehot_labels = np.zeros((n, num_classes), dtype=np.bool)
    for i in range(n):
        onehot_labels[i, partial_labels[i]] = 1
    return onehot_labels, partial_labels


if __name__ == "__main__":
    trainset, testset = load_dataset()
    true_labels = np.array(trainset.targets)

    model = train_dataset_model(ResNet18, trainset, testset)
    predictions = get_topk_predictions(model, trainset)
    partial_labels = generate_partial_labels(true_labels, predictions, avg_num_cl=5)

    with open('../datasets/pl_CIFAR10_avgcl5.pkl', 'wb') as f:
        pickle.dump(partial_labels, f)
