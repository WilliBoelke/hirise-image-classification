import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import get_dataloader
from model import HiRiseModel


def training():
    model.train()
    total_loss = 0
    correct = 0
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossFunction(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        predicted = output.argmax(dim=1, keepdim=True)
        correct += predicted.eq(target.view_as(predicted)).sum().item()

    data_size = len(train_loader.dataset)
    total_loss /= data_size
    train_losses.append(total_loss)
    percentage = 100. * correct / data_size
    train_accuracy.append(percentage)
    print(
        "\n Training: Loss: {:.4f}, Accuracy: {}/{}, ({:.1f}%)".format(total_loss, correct, data_size, percentage))


def evaluation():
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(eval_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += lossFunction(output, target).item()
            predicted = output.argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

    data_size = len(eval_loader.dataset)
    total_loss /= data_size
    test_losses.append(total_loss)
    percentage = 100. * correct / data_size
    test_accuracy.append(percentage)
    print(
        "\n Evaluation: Loss: {:.4f}, Accuracy: {}/{}, ({:.1f}%)".format(total_loss, correct, data_size, percentage))


# Makes a confusion matrix by sending the whole test data through the
# Source: https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
def make_confusion_matrix():
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = target.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

        classes = ('others', '1', '2', '3', '4',
                   '5', '6', '7')

        cf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 7))
        sn.heatmap(cf_matrix, annot=True)
        plt.savefig('conf_matrix.png')


def save_plot():
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Evaluation Loss")
    plt.legend()
    plt.savefig("Loss.svg")
    plt.close()

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(test_accuracy, label="Evaluation Accuracy")
    plt.legend()
    plt.savefig("accuracy.svg")
    plt.close()


if __name__ == "__main__":

    # setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyper paramater
    m_learning_rate = 0.3
    m_batch_size = 64
    number_of_epochs = 30
    m_momentum = 0.1
    # model and data
    model = HiRiseModel().to(device)
    train_loader, eval_loader = get_dataloader.weighted('data/datasetLabels/4238_sample_dataset_labels.csv',
                                                              batch_size=m_batch_size)

    # algorithms
    optimizer = torch.optim.Adadelta(model.parameters())
    lossFunction = torch.nn.CrossEntropyLoss()
    # lossFunction = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 1.85, 1, 1.21, 12.95, 1.85, 4.45]).to(device))

    # statistics
    train_losses = []
    test_losses = []
    test_accuracy = []
    train_accuracy = []

    # trading loop
    for epoch in tqdm(range(1, number_of_epochs + 1), "Main loop"):
        training()
        evaluation()

    # save stats and metrics
    save_plot()
    make_confusion_matrix()
    torch.save(model.state_dict(), "trainedModel.pt")
