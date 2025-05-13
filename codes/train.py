'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-09 15:22:32
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-12 18:20:37
FilePath: /LS-PLL-Reproduction/codes/train.py
Description: Functions relates to model training
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.datasets as datasets
from PIL import Image

from utils import device


class LS_PLL_CrossEntropy(nn.Module):
    def __init__(self, smoothing_rate=0.1):
        super(LS_PLL_CrossEntropy, self).__init__()
        self.smoothing_rate = smoothing_rate

    def forward(self, logits, candidates):
        """
        Partial Label Learning with Smoothing Cross Entropy
        params:
            logits (Tensor): raw model output logits [batch_size, num_classes]
            candidates (Tensor): multi-hot encoded labels [batch_size, num_classes]
        return:
            loss (Tensor): cross-entropy loss after smoothing
        """

        # calc softmax
        probs = F.softmax(logits, dim=1)

        # calc denominator for every sample
        # fij_term = torch.logsumexp(logits, dim=1, keepdim=True)
        fij_term = torch.log(torch.sum(torch.exp(probs), dim=1, keepdim=True))

        # only accumulates on candidate set j∈Y_i: 
        weighted = candidates * (probs - fij_term)

        loss = -weighted.sum(dim=1).mean()

        return loss


class PartialLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, partial_labels, 
                transform=None, target_transform=None):
        self.data = dataset.data
        self.targets = torch.tensor(partial_labels, dtype=torch.float64, device=device)

        self.transform = transform
        self.target_transform = target_transform

        transforms = datasets.vision.StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if type(img) == torch.Tensor:
            img = img.numpy()

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def init_pesudo_truth(candidates):
    """
    function for randomly choosing a pesudo ground truth from the candidates set
    param:
        candidates (Tensor): multi-hot encoded labels [batch_size, num_classes]
    return:
        pesudo_truth (Tensor): pesudo truth label initialised [batch_size,]
    """
    # count number of candidates
    counts = candidates.sum(dim=1, keepdim=True)
    counts = counts.masked_fill(counts == 0, 1.)

    # randomly sampling 1 label for every data sample
    probs = candidates / counts                         # [batch_size, num_classes]
    sampled = torch.multinomial(probs, num_samples=1)   # [batch_size, 1]

    pesudo_truth = sampled.squeeze(1)                  # [batch_size,]
    return pesudo_truth


def update_pesudo_truth(logits, candidates, softmax_accumulator=None, weighting_param=0.9):
    """
    function for updating the pesudo ground truth after every iteration
    params:
        logits (Tensor): raw model output logits [batch_size, num_classes]
        candidates (Tensor): multi-hot encoded labels [batch_size, num_classes]
        softmax_accumulator (Tensor): previous EMA accumulator [batch_size, num_classes]
        weighting_param (float): EMA decay factor η
    returns:
        updated_pseudo: LongTensor of shape (B,), new pseudo-label indices
        softmax_accumulator: updated EMA accumulator [batch_size, num_classes]
    """
    if softmax_accumulator is None:
        softmax_accumulator = torch.zeros_like(logits)

    exp_logits = torch.exp(logits) * candidates         # [batch_size, num_classes]
    sum_exp = exp_logits.sum(dim=1, keepdim=True)       # [batch_size, 1]
    sm_on_candidates = exp_logits / (sum_exp + 1e-12)   # [batch_size, num_classes]

    softmax_accumulator = (
        weighting_param * softmax_accumulator + (1.-weighting_param) * sm_on_candidates
    )                                                   # [batch_size, num_classes]

    accum_sum = (softmax_accumulator * candidates).sum(dim=1, keepdim=True)     # [batch_size, 1]
    normalised_accum = softmax_accumulator / (accum_sum + 1e-12)                # [batch_size, num_classes]
    normalised_accum = normalised_accum * candidates

    updated_pseudo = torch.argmax(normalised_accum, dim=1)                      # [batch_size,]
    return updated_pseudo, softmax_accumulator


## function for smoothing and updating values in the label set
def update_candidates(candidates, pesudo_truth, smoothing_rate):
    """
    function for smoothing and updating values in the candidate set
    params:
        candidates (Tensor): multi-hot encoded labels [batch_size, num_classes]
        pesudo_truth (Tensor): pesudo truth label initialised [batch_size,]
        smoothing_rate (float)
    return:
        updated_candidates (Tensor): partial labels updated with smoothing rate [batch_size, num_classes]
    """
    counts = (candidates != 0).sum(dim=1, keepdim=True).clamp(min=1).to(torch.float64)
    updated_candidates = torch.zeros_like(candidates, dtype=torch.float64)

    smoothed_values = (smoothing_rate / counts).expand_as(candidates)

    non_zero = candidates != 0
    updated_candidates[non_zero] = smoothed_values[non_zero]

    row_indices = torch.arange(candidates.size(0), device=device)
    updated_candidates[row_indices, pesudo_truth] += (1.-smoothing_rate)

    return updated_candidates


def train_model(
    Model, trainset, testset, 
    num_epochs=200, batch_size=128,
    lr=0.01, momentum=0.9, weighting_param=0.9,
    num_classes=10, smoothing_rate=.0
):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    model = Model(num_classes=num_classes).to(device)
    optimiser = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=0.001, nesterov=True)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=3)
    criterion = LS_PLL_CrossEntropy(smoothing_rate=smoothing_rate)
    record = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    pseudo_truth = init_pesudo_truth(trainset.targets)
    ema_accumulator = torch.zeros_like(trainset.targets)

    for epoch in range(num_epochs):
        model.train()
        running_loss = total = correct = 0
        for inputs, labels, indices in trainloader:
            inputs, candidate_batch = inputs.to(device), labels.to(device)
            target_batch = pseudo_truth[indices]
            ema_batch = ema_accumulator[indices]
            candidate_batch = update_candidates(candidate_batch, target_batch, smoothing_rate)

            optimiser.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, candidate_batch)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == target_batch).sum().item()
            total += inputs.size(0)

            updated_pseudo, updated_ema = update_pesudo_truth(
                outputs.detach(), candidate_batch, ema_batch, weighting_param
            )
            pseudo_truth[indices] = updated_pseudo
            ema_accumulator[indices] = updated_ema

        train_loss = running_loss / len(trainloader)
        train_acc = correct / total * 100

        model.eval()
        with torch.no_grad():
            running_test_loss = total = correct = 0
            for inputs, labels, _ in testloader:
                inputs, target = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, target)

                running_test_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                batch_indices = torch.arange(predictions.size(0), device=device)
                correct += target[batch_indices, predictions].sum().item()
                total += target.size(0)

            test_loss = running_test_loss / len(testloader)
            test_acc = correct / total * 100

        scheduler.step(test_loss)
        record['train_loss'].append(train_loss)
        record['train_acc'].append(train_acc)
        record['val_loss'].append(test_loss)
        record['val_acc'].append(test_acc)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: \
                    \n\tTrain Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.3f} \
                    \n\tTest Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.3f}')

    return model, record

