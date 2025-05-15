'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-06 15:24:13
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-15 18:23:38
FilePath: /LS-PLL-Reproduction/codes/prepare_data.py
Description: The codes to download, train and generate partial labels for datasets.
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import pickle

from ResNet18 import ResNet18
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


def train_dataset_model(Model, trainset, testset, num_classes):
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)
    model = Model(num_classes=num_classes).to(device)
    optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train()
        running_loss = total = correct = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += inputs.size(0)

        train_loss = running_loss / len(trainloader)
        train_acc = correct / total * 100

        model.eval()
        with torch.no_grad():
            running_test_loss = total = correct = 0
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_test_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            test_loss = running_test_loss / len(testloader)
            test_acc = correct / total * 100

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: \
                    \n\tTrain Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.3f} \
                    \n\tTest Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.3f}')
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


def get_random_predictions(model, dataset, batch_size=128, k=6):
    model.eval()
    num_classes = model.fc.out_features
    with torch.no_grad():
        randomk_preds = []
        for inputs, targets in DataLoader(dataset, batch_size=batch_size):
            inputs, targets = inputs.to(device), targets.to(device)
            actual_batch_size = inputs.size(0)
            
            random_preds = []
            for i in range(actual_batch_size):
                possible_labels = list(set(range(num_classes)) - {targets[i].item()})
                random_k = np.random.choice(possible_labels, size=k, replace=False)
                random_preds.append(random_k)

            randomk_preds.append(np.array(random_preds))
        randomk_preds = np.concatenate(randomk_preds, axis=0)
    return randomk_preds


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

    model = train_dataset_model(ResNet18, trainset, testset, num_classes=10)
    predictions = get_topk_predictions(model, trainset)
    partial_labels = generate_partial_labels(true_labels, predictions, avg_cl=5)

    with open('../datasets/pl_CIFAR10_avgcl5.pkl', 'wb') as f:
        pickle.dump(partial_labels, f)
