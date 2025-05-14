'''
Author: Jedidiah-Zhang yanzhe_zhang@protonmail.com
Date: 2025-05-06 16:42:21
LastEditors: Jedidiah-Zhang yanzhe_zhang@protonmail.com
LastEditTime: 2025-05-13 18:19:15
FilePath: /LS-PLL-Reproduction/codes/main.py
Description: Main script containing the complete pipeline for training and evaluating models with partial labels.
'''

from pathlib import Path
import argparse
import os

from prepare_data import *
from LeNet5 import LeNet5
from ResNet18 import ResNet18
from ResNet56 import ResNet56
from train import PartialLabelDataset, train_model
from utils import validate_path, extract_features, tsne_plot, plot_grid

parser = argparse.ArgumentParser(description='Full experiments')
parser.add_argument('--model_path', type=validate_path, default='./models', help="Path to models folder")
parser.add_argument('--dataset_path', type=validate_path, default='./datasets', help="Path to datasets folder")
parser.add_argument('--figure_path', type=validate_path, default='./doc/figures', help="Path to saved figures folder")
args = parser.parse_args()
MODEL_PATH, DATASET_PATH, FIGURE_PATH = args.model_path, args.dataset_path, args.figure_path

BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 200
WEIGHTING_PARAM = 0.9
MOMENTUM = 0.9
SMOOTHING_RATE = [0.1, 0.3, 0.5, 0.7, 0.9]
EXPERIMENTS = [
    {
        'Dataset': 'FashionMNIST', 
        'Model': LeNet5, 
        'AvgCL': [3, 4, 5], 
        'NumClasses': 10, 
        'TopK': 6
    },
    {
        'Dataset': 'KuzushijiMNIST', 
        'Model': LeNet5, 
        'AvgCL': [3, 4, 5],
        'NumClasses': 10, 
        'TopK': 6
    },
    {
        'Dataset': 'CIFAR10', 
        'Model': ResNet18, 
        'AvgCL': [3, 4, 5],
        'NumClasses': 10, 
        'TopK': 6
    },
    {
        'Dataset': 'CIFAR100', 
        'Model': ResNet56, 
        'AvgCL': [7, 9, 11],
        'NumClasses': 100, 
        'TopK': 20
    }
]


def main():
    for exp in EXPERIMENTS: # for each models and relative datasets
        models, records = {}, {}
        figure_paths, titles = [], []
        plot_idx = 0
        print()

        # load dataset
        trainset, testset = load_dataset(exp['Dataset'], dataset_path=DATASET_PATH)
        if type(trainset.targets) == torch.Tensor:
            true_labels_train = trainset.targets.numpy()
            true_labels_test = testset.targets.numpy()
        else:
            true_labels_train = np.array(trainset.targets)
            true_labels_test = np.array(testset.targets)

        for avgCL in exp['AvgCL']: # for each noise levels
            models[avgCL], records[avgCL] = [], []

            # train model for generating datasets if model file not exist, or load model if exists
            dataset_model_path = f"{MODEL_PATH}/PL_{exp['Dataset']}_{exp['Model'].name}.pth"
            if Path(dataset_model_path).exists():
                model = exp['Model'](num_classes=exp['NumClasses']).to(device)
                model.load_state_dict(torch.load(dataset_model_path))
                model.eval()
            else:
                print(f"**** Training {exp['Model'].name} model for {exp['Dataset']} to generate PL ****")
                model = train_dataset_model(exp['Model'], trainset, testset,
                                            num_classes=exp['NumClasses'])
                torch.save(model.state_dict(), dataset_model_path)
                print(f"**** Model saved to {dataset_model_path} ****")

            # load, or generate and load partial label datasets for both train and test sets
            traindata_path = f"{DATASET_PATH}/pl_{exp['Dataset']}_avgcl{avgCL}_train.npy"
            if Path(traindata_path).exists(): partial_labels_train = np.load(traindata_path)
            else:
                print(f"**** Generating partial labels for {exp['Dataset']} with avgCL {avgCL} ****")
                predictions_train = get_topk_predictions(model, trainset, k=exp['TopK'])
                partial_labels_train, _ = generate_partial_labels(true_labels_train, predictions_train, 
                                                                avg_cl=avgCL, k=exp['TopK'], 
                                                                num_classes=exp['NumClasses'])
                np.save(traindata_path, partial_labels_train)
                print(f"**** Partial labels saved to {traindata_path} ****")

            testdata_path = f"{DATASET_PATH}/pl_{exp['Dataset']}_avgcl{avgCL}_test.npy"
            if Path(testdata_path).exists(): partial_labels_test = np.load(testdata_path)
            else:
                predictions_test = get_topk_predictions(model, testset, k=exp['TopK'])
                partial_labels_test, _ = generate_partial_labels(true_labels_test, predictions_test, 
                                                                avg_cl=avgCL, k=exp['TopK'], 
                                                                num_classes=exp['NumClasses'])
                np.save(testdata_path, partial_labels_test)
                print(f"**** Partial labels saved to {testdata_path} ****")

            # Train models with partial labels without label smoothing
            # using nn.CrossEntropy() as default.
            model_path = MODEL_PATH + f"/{exp['Dataset']}/avgcl{avgCL}"
            if not os.path.exists(model_path): os.makedirs(model_path)
            train_set = PartialLabelDataset(trainset, partial_labels_train, transform=transforms.ToTensor())
            test_set = PartialLabelDataset(testset, partial_labels_test, transform=transforms.ToTensor())
            print(f"**** Training {exp['Model'].name} on partial labelled {exp['Dataset']} with Avg.#CL={avgCL} and no label smoothing ****")
            non_smoothing_model, non_smoothing_record = train_model(exp['Model'], train_set, test_set, num_epochs=EPOCHS, 
                                                                    batch_size=BATCH_SIZE, lr=LEARNING_RATE, momentum=MOMENTUM, 
                                                                    weighting_param=WEIGHTING_PARAM, num_classes=exp['NumClasses'])
            models[avgCL].append(non_smoothing_model)
            records[avgCL].append(non_smoothing_record)
            torch.save(non_smoothing_model.state_dict(), model_path+"/r_noLS.pth")
            print(f"**** Model saved to {model_path}/r_noLS.pth ****")

            # generate and save plots
            figure_path = FIGURE_PATH + f"/tsne_{exp['Dataset']}_cl{avgCL}_r_noLS.png"
            figure_paths.append(figure_path)
            titles.append(f"({chr(97+plot_idx)}) w/o LS")
            plot_idx += 1
            features, labels = extract_features(non_smoothing_model, testset, batch_size=BATCH_SIZE)
            tsne_plot(features, labels, figure_path, f"Avg.#CL={avgCL}")
            print(f"**** TSNE plot saved to {figure_path} ****")

            # train models with label smoothing across different noise levels
            for r in SMOOTHING_RATE:
                print(f"\n**** Training {exp['Model'].name} on partial labelled {exp['Dataset']} with Avg.#CL={avgCL} and a smoothing rate of {r} ****")
                model, record = train_model(exp['Model'], train_set, test_set, num_epochs=EPOCHS, batch_size=BATCH_SIZE, 
                                            lr=LEARNING_RATE, momentum=MOMENTUM, weighting_param=WEIGHTING_PARAM, 
                                            num_classes=exp['NumClasses'], smoothing_rate=r)
                models[avgCL].append(model)
                records[avgCL].append(record)
                torch.save(non_smoothing_model.state_dict(), model_path+f"/r_{r}.pth")
                print(f"**** Model saved to {model_path}/r_{r}.pth ****")

                # generate and save plots
                figure_path = FIGURE_PATH + f"/tsne_{exp['Dataset']}_cl{avgCL}_r_{r}.png"
                figure_paths.append(figure_path)
                titles.append(f"({chr(97+len(titles))}) w/ LS, r={r}")
                plot_idx += 1
                features, labels = extract_features(model, testset, batch_size=BATCH_SIZE)
                tsne_plot(features, labels, figure_path)
                print(f"**** TSNE plot saved to {figure_path} ****")

        # save records into the models folder
        with open(MODEL_PATH + f"/{exp['Dataset']}/records.pkl", 'wb') as f:
            pickle.dump(records, f)

        # plot grid
        plot_grid(figure_paths, titles, rows=len(exp['AvgCL']), cols=len(SMOOTHING_RATE)+1, 
                    save_path=FIGURE_PATH+f"/tsne_grid_{exp['Dataset']}.png")

if __name__ == "__main__":
    main()
