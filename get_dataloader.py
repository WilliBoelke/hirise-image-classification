import torch
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler, DataLoader
from tqdm import tqdm

from dataset import HiRiseDataset


# class_weights = [1, 12.46, 53.5, 26.19, 34.89, 264.3, 53.18, 128.26]
# class_weights = [1, 1, 1, 1, 1, 1, 1, 1]

def random_sampler(labels, batch_size):
    dataset = HiRiseDataset(csv_file=labels,
                            root='data/datasetLabels/images',
                            transform=transforms.ToTensor())
    train_len = int(0.8 * len(dataset))
    eval_len = len(dataset) - train_len
    random_train_set, random_eval_set = torch.utils.data.dataset.random_split(dataset, [train_len, eval_len])
    train_loader = DataLoader(pin_memory=True, num_workers=4, dataset=random_train_set, batch_size=batch_size,
                              shuffle=True, )
    eval_loader = DataLoader(pin_memory=True, num_workers=4, dataset=random_eval_set, batch_size=batch_size,
                             shuffle=True)
    return train_loader, eval_loader


def weighted(labels, batch_size, class_weights=None):
    if class_weights is None:
        class_weights = [1, 1, 1.85, 1, 1.21, 12.95, 1.85, 4.45]

    m_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.4847, 0.2107),
            transforms.RandomAdjustSharpness(4),
            transforms.RandomRotation(40)
        ])
    dataset = HiRiseDataset(csv_file=labels, root='data/datasetLabels/images', transform=m_transforms)

    # lengths
    train_len = int(0.8 * len(dataset))
    eval_len = len(dataset) - train_len

    # splitting the data
    random_train_set, random_eval_set = torch.utils.data.dataset.random_split(dataset, [train_len, eval_len])

    # Adding the weights
    train_sample_weights = [0] * len(random_train_set)
    for index, (data, label) in tqdm(enumerate(random_train_set), "Weights training Data"):
        class_weight = class_weights[label]
        train_sample_weights[index] = class_weight

    eval_sample_weights = [0] * len(random_eval_set)
    for index, (data, label) in tqdm(enumerate(random_eval_set), "Weights evaluation Data"):
        class_weight = class_weights[label]
        eval_sample_weights[index] = class_weight

    # building th weighted samplers
    train_sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_sample_weights), replacement=True)
    eval_sampler = WeightedRandomSampler(eval_sample_weights, num_samples=len(eval_sample_weights), replacement=True)

    # building the dataloaders
    train_loader = DataLoader(pin_memory=True, num_workers=4, dataset=random_train_set, batch_size=batch_size,
                              shuffle=False, sampler=train_sampler)
    eval_loader = DataLoader(pin_memory=True, num_workers=4, dataset=random_eval_set, batch_size=batch_size,
                             shuffle=False, sampler=eval_sampler)

    return train_loader, eval_loader


def calculate_mean_and_std_derivation():
    dataset = HiRiseDataset(csv_file='data/datasetLabels/average_sample_dataset_labels.csv',  root='data/datasetLabels/images',   transform=transforms.ToTensor())
    loader  = DataLoader(pin_memory=True, num_workers=1, dataset=dataset, batch_size=len(dataset))
    data = next(iter(loader))
    print(data[0].mean())
    print(data[0].std())
    # result :
    # tensor(0.4847)
    # tensor(0.2107)