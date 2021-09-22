import pandas as pd
import os, shutil


def sort_images():
    df = pd.read_csv("complete_dataset_lables.csv", usecols=['path', 'label'])
    grouped = df.groupby(by="label")

    for label, group in grouped:
        if not os.path.isdir("sortedDatasets/" +str(label)):
            os.mkdir("sortedDatasets/" +str(label))
        for index, data in group.iterrows():
            print(data['path'])
            img = "../unsortedDataset/images/" + data["path"]
            shutil.copy(img, "sortedDatasets/" + str(label))


def make_equal_dataset():
    df = pd.read_csv("complete_dataset_lables.csv", usecols=['path', 'label'])
    grouped = df.groupby(by="label")
    for label, group in grouped:
        print(group)
        print(group.reset_index())
        group.head(231).to_csv("equal_dataset_labels.csv", mode="a", index=False, header=False)


def make_476_sample_dataset():
    df = pd.read_csv("complete_dataset_lables.csv", usecols=['path', 'label'])
    grouped = df.groupby(by="label")

    for label, group in grouped:
        print(group)
        print(group.reset_index())
        group.head(476).to_csv("476_sample_dataset_labels.csv", mode="a", index=False, header=False)


def make_dataset_without_others():
    df = pd.read_csv("complete_dataset_lables.csv", usecols=['path', 'label'])
    grouped = df.groupby(by="label")

    for label, group in grouped:
        if label != 0:
            print(group)
            print(group.reset_index())
            group.to_csv("without_others_dataset_labels.csv", mode="a", index=False, header=False)


def make_average_samples_dataset():
    df = pd.read_csv("complete_dataset_lables.csv", usecols=['path', 'label'])
    grouped = df.groupby(by="label")

    for label, group in grouped:
        print(group)
        print(group.reset_index())
        group.head(1000).to_csv("average_sample_dataset_labels.csv", mode="a", index=False, header=False)


make_average_samples_dataset()
make_dataset_without_others()
make_476_sample_dataset()
make_equal_dataset()
