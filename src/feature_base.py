import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager
import pandas as pd
import argparse
import inspect
import os
import csv

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', nargs='*', help='Overwrite existing files', default=list())
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.data_path.exists() and f.name not in overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()

def load_datasets(feats):
    dfs = [pd.read_pickle(f) for f in feats]
    data = pd.concat(dfs, axis=1)
    return data

@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.data = pd.DataFrame()
        self.data_path = Path(self.dir) / f'{self.name}.pkl'
    
    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.data.columns = prefix + self.data.columns + suffix
        return self
    
    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        self.data.to_pickle(str(self.data_path))

    def create_memo(self, memo):
        path = self.dir + '/_feature_memo.csv'
        if not os.path.isfile(path):
            with open(path, 'w'): pass
        
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split(',') for line in lines]

        row = [i for i, line in enumerate(lines) if line[0] == self.name]
        columns = ' / '.join(self.data.columns)
        data = [self.name, memo, columns]
        if len(row) != 0:
            with open(path, 'w') as f:
                writer = csv.writer(f)
                lines[row[0]] = data
                writer.writerows(lines)
        else:
            with open(path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([self.name, memo, columns])