import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler, DataLoader
from tqdm import tqdm

from OriginalDataset import HiRiseDataset



in_channel = 1
number_of_classes = 8
m_learning_rate = 0.001
batch_size = 64
number_of_epochs = 10
m_kernel_size = 2

# Loading Data


# These are random Datasets, that means that its not guaranteed that 
# there is a well distributed amout of data for each class

def get_data_leader():
    dataset = HiRiseDataset(csv_file='data/unsortedDataset/average_sample_dataset_labels.csv', root='data/unsortedDataset/images',
                            transform=transforms.ToTensor())


    train_len = int(0.8 * len(dataset))
    eval_len = len(dataset) - train_len


    random_train_set, random_eval_set = torch.utils.data.dataset.random_split(dataset, [train_len, eval_len])
    train_loader = DataLoader(pin_memory=True, num_workers=4, dataset=random_train_set, batch_size=batch_size, shuffle=True, )
    eval_loader = DataLoader(pin_memory=True, num_workers=4, dataset=random_eval_set, batch_size=batch_size, shuffle=True)
    return train_loader, eval_loader


def get_weighted_data_leader():
    m_transforms = transforms.Compose(
         [
            transforms.ToTensor(),
         ]
    )

    dataset = HiRiseDataset(csv_file='data/unsortedDataset/average_sample_dataset_labels.csv', root='data/unsortedDataset/images', transform=m_transforms)
    train_len = int(0.8 * len(dataset))
    eval_len = len(dataset) - train_len
    random_train_set, random_eval_set = torch.utils.data.dataset.random_split(dataset, [train_len, eval_len])

    # class_weights = [1, 12.46, 53.5, 26.19, 34.89, 264.3, 53.18, 128.26]
    class_weights = [1, 1, 1.85, 1, 1.21, 12.95, 1.85, 4.45]
    #class_weights = [1, 1, 1, 1, 1, 1, 1, 1]
    train_sample_weights = [0] * len(random_train_set)
    # iterating through all the samples in the data and specifying the weights depending on the class
    for index, (data, label) in tqdm(enumerate(random_train_set), "Weights training Data"):
        class_weight = class_weights[label]
       # print("lable " + str(label) + " weight " + str(class_weight))
        train_sample_weights[index] = class_weight


    eval_sample_weights = [0] * len(random_eval_set)
    # iterating through all the samples in the data and specifying the weights depending on the class#
    for index, (data, label) in tqdm(enumerate(random_eval_set), "Weights evaluation Data"):
        class_weight = class_weights[label]
       # print("lable " + str(label) + " weight " + str(class_weight))
        eval_sample_weights[index] = class_weight

    train_sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_sample_weights), replacement=True)
    eval_sampler = WeightedRandomSampler(eval_sample_weights, num_samples=len(eval_sample_weights), replacement=True)
    train_loader = DataLoader(pin_memory=True, num_workers=4, dataset=random_train_set, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    eval_loader = DataLoader(pin_memory=True, num_workers=4, dataset=random_eval_set, batch_size=batch_size, shuffle=False, sampler=eval_sampler)
    return train_loader, eval_loader

# graphs

class HiRiseModel(nn.Module):

    def __init__(self, width, height):
        super(HiRiseModel, self).__init__()
        self.convolution_one = nn.Conv2d(1, 6, 3)
        self.pooling_one = nn.MaxPool2d(3)
        self.convolution_two = nn.Conv2d(6, 16, 3)
        self.fully_connected_one = nn.Linear(9216, 8000)
        self.fully_connected_two = nn.Linear(8000, 500)
        self.fully_connected_three = nn.Linear(500, number_of_classes)


    def forward(self, x):
        x = self.convolution_one(x)
        x = F.relu(x)
        x = self.pooling_one(x)
        x = self.convolution_two(x)
        x = F.relu(x)
        x = self.pooling_one(x)
        #print(x.shape)
        x = x.view(-1, 9216)
        x = F.relu(self.fully_connected_one(x))
        x = F.relu(self.fully_connected_two(x))
        x = self.fully_connected_three(x)
        return x



def training():
    model.train()
    total_loss = 0
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossFunction(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    total_loss /= len(train_loader.dataset)
    train_losses.append(total_loss)


def evaluation():
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(eval_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += lossFunction(output, target).item()
            prediced = output.argmax(dim=1, keepdim=True)
            #print("Predicted =  {} , target = {}".format(prediced, target))
            correct += prediced.eq(target.view_as(prediced)).sum().item()

    data_size = len(eval_loader.dataset)
    total_loss /= data_size
    test_losses.append(total_loss)
    percentage = 100. * correct / data_size
    test_accuracy.append(percentage)
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{}, ({:.1f}%)".format(total_loss, correct, data_size, percentage))


def savePlot():
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Evaluation Loss")
    plt.legend()
    plt.savefig("ModelTraining_graph_losses__with_average_dataset_2_newModel-weights.svg")
    plt.close()

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(test_accuracy, label="Test Accuracy")
    plt.legend()
    plt.savefig("ModelTraining_graph_accuracy__with_average_dataset_newModel-weights.svg")
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HiRiseModel(227, 227).to(device)
    optimizer = torch.optim.SGD(model.parameters(), m_learning_rate)
    train_loader, eval_loader = get_weighted_data_leader()
    #lossFunction = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 1.85, 1, 1.21, 12.95, 1.85, 4.45]).to(device))
    lossFunction = torch.nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    test_accuracy = []
    for epoch in tqdm(range(1, 101), "Main loop"):
        print("Epoch " + str(epoch))
        training()
        evaluation()
        print(test_accuracy)
    savePlot()
    torch.save(model.state_dict(), "trainedModel.pt")


class old (nn.Module):
    def __init__(self, width, height):
        super(HiRiseModel, self).__init__()
        # convolution layer
        self.convolution1 = nn.Conv2d(1, 5, kernel_size=m_kernel_size)
        self.convolution2 = nn.Conv2d(5, 10, kernel_size=m_kernel_size)
        self.conv_dropout = nn.Dropout2d()
        self.fully_con_linear_layer1 = nn.Linear(7290, 7290)
        self.fully_con_linear_layer2 = nn.Linear(7290, 7290)
        self.fully_con_linear_layer3 = nn.Linear(7290, 7290)
        self.fully_con_linear_layer4 = nn.Linear(7290, number_of_classes)

    def forward(self, x):
        x = self.convolution1(x)
        x = F.max_pool2d(x, 4)  # pooling layer
        x = F.relu(x)
        x = self.convolution2(x)
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.view(-1, 7290)  #
        x = F.relu(self.fully_con_linear_layer1(x))
        x = F.relu(self.fully_con_linear_layer2(x))
        x = F.relu(self.fully_con_linear_layer3(x))
        x = self.fully_con_linear_layer4(x)
        return F.log_softmax(x, dim=1)  # logarthm of the classification probaility

