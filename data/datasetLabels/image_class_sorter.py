import os
import shutil

import pandas as pd

def sort_images():
    df = pd.read_csv("complete_dataset_labels.csv", usecols=['path', 'label'])
    grouped = df.groupby(by="label")

    for label, group in grouped:
        if not os.path.isdir("sortedDatasets/" + str(label)):
            os.mkdir("sortedDatasets/" + str(label))
        for index, data in group.iterrows():
            img = "../datasetLabels/images/" + data["path"]
            shutil.copy(img, "sortedDatasets/" + str(label))


def make_equal_dataset():
    df = pd.read_csv("complete_dataset_labels.csv", usecols=['path', 'label'])
    grouped = df.groupby(by="label")
    for label, group in grouped:
        group.head(231).to_csv("equal_dataset_labels.csv", mode="a", index=False, header=False)


def make_476_sample_dataset():
    df = pd.read_csv("complete_dataset_labels.csv", usecols=['path', 'label'])
    grouped = df.groupby(by="label")

    for label, group in grouped:
        group.head(476).to_csv("476_sample_dataset_labels.csv", mode="a", index=False, header=False)


def make_dataset_without_others():
    df = pd.read_csv("complete_dataset_labels.csv", usecols=['path', 'label'])
    grouped = df.groupby(by="label")

    for label, group in grouped:
        if label != 0:
            group.to_csv("without_others_dataset_labels.csv", mode="a", index=False, header=False)


def make_average_samples_dataset():
    df = pd.read_csv("complete_dataset_labels.csv", usecols=['path', 'label'])
    grouped = df.groupby(by="label")

    for label, group in grouped:
        group.head(2119).to_csv("average_sample_dataset_labels.csv", mode="a", index=False, header=False)


make_average_samples_dataset()