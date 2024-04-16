from glob import glob
from tqdm import tqdm
import swifter
import json
import os

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import pandas as pd

class korquad:
    def __init__(self,dir):
        self.lst = glob(dir)
        pass
    
    def split_convert_dataset(self,
                              test_size : float = 0.2,
                              random_state : int = 42):
        print(self.lst)
        self.df = pd.read_json(f'{self.lst[0]}')
        for idx in range(1,len(self.lst[1:])):
            new_df = pd.read_json(f'{self.lst[idx]}')
            self.df = pd.concat([self.df,new_df])
        self.df['qas'] = self.df['data'].swifter.apply(lambda x: x['qas'])
        self.df['title'] = self.df['data'].swifter.apply(lambda x: x['title'])
        self.df['url'] = self.df['data'].swifter.apply(lambda x: x['url'])
        self.df['context'] = self.df['data'].swifter.apply(lambda x: x['context'])
        self.df.drop(columns={'version','data'},inplace=True)
        train_dataset, test_dataset = train_test_split(self.df, test_size=test_size,random_state=random_state)
        train_dataset.reset_index(inplace=True,drop=True)
        train = Dataset.from_pandas(train_dataset)
        test_dataset.reset_index(inplace=True,drop=True)
        test = Dataset.from_pandas(test_dataset)
        self.dataset = DatasetDict(
            {
                "train" : train,
                "test" : test
            }
        )

        return None

    def save_dataset(
            self,
            save_path : str
            ) -> object :
        self.dataset.save_to_disk(save_path)
        print('Doneee')

if __name__ == "__main__":
    data_path = 'NLP/data/korquad/ver2.0/*.json'
    # json_list = glob(data_path)
    dataset = korquad(data_path)
    data = dataset.split_convert_dataset(test_size=0.2, random_state=42)
    save_path = 'NLP/data/datasets/korquad2.0'
    dataset.save_dataset(save_path=save_path)
    
    
