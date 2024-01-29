import argparse

from NLP.Dataset.Text_Translation_Dataset import *
from NLP.Dataset.Summarization_Dataset import *
from NLP.Dataset.korquad_prep import *


def load_dataset():
    if model == 'korquad':
        print('korquad')
    elif model == 'summary':
        print('summary')
    elif model == 'translation':
        print('translation')
    return dataset


def load_model():

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset")   
    parser.add_argument("-m", "--model")           
    args = parser.parse_args()
    
    dataset = load_dataset(parser.dataset)
    model = load_model(dataset, parser.model)
    