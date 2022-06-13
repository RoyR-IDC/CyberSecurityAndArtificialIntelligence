# math
import math
import numpy as np
np.random.seed(0)
import pandas as pd
from scipy import stats
from scipy.stats import geom

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# general
import time

# EMBER
import ember


def read_it():
    folder_path = f'/Users/royrubin/Downloads/ember2018/'
    filenames = [
        'train_features_0.jsonl',
        'train_features_1.jsonl',
        'train_features_2.jsonl',
        'train_features_3.jsonl',
        'train_features_4.jsonl',
        'train_features_5.jsonl',
    ]

    df = None
    for filename in filenames:
        print(f'reading {filename}')
        start = time.time()
        temp_df = pd.read_json(folder_path+filename, lines=True)
        end = time.time()
        print(f'performed in {(end - start) / 60} minutes')
        print(temp_df.shape)
        print(temp_df.head(5))
        if df is None:
            df = temp_df
        else:
            df = pd.concat([df, temp_df])

    df.to_csv(f'{folder_path}_new_df.csv')
    return df


def read_and_create_new_df():
    folder_path = f'/Users/royrubin/Downloads/ember2018/'
    filenames = [
        'train_features_0.jsonl',
        'train_features_1.jsonl',
        'train_features_2.jsonl',
        'train_features_3.jsonl',
        'train_features_4.jsonl',
        'train_features_5.jsonl',
    ]

    for index, filename in enumerate(filenames):
        print(f'reading {filename}')
        start = time.time()
        temp_df = pd.read_json(folder_path + filename, lines=True)
        end = time.time()
        print(f'performed in {(end - start) / 60} minutes')
        print(temp_df.shape)
        print(temp_df.head(2))

        print(f'saving {filename} to csv ')
        start = time.time()
        if index == 0:
            temp_df.to_csv(f'{folder_path}_new_df.csv', index=False)
        else:
            temp_df.to_csv(f'{folder_path}_new_df.csv', mode='a', columns=False, index=False)
        end = time.time()
        print(f'performed in {(end - start) / 60} minutes')

        # either way
        del temp_df


def flatten_df_features(df: pd.DataFrame):
    new_df = pd.DataFrame()
    df_list_of_dicts = df.to_dict('records')
    for row in df_list_of_dicts:
        new_row = dict()
        for key, value in row.items():
            if not isinstance(value, dict):
                if isinstance(value, list):
                    try:
                        new_row[key+'__count_all'] = len(value)
                        new_row[key+'__count_unique'] = len(set(value))
                        new_row[key+'__mean'] = np.mean(value)
                        new_row[key+'__median'] = np.median(value)
                        new_row[key+'__min'] = np.min(value)
                        new_row[key+'__max'] = np.max(value)
                    except:
                        pass
                else:
                    new_row[key] = value
                continue
            # if key not in ['histogram', 'byteentropy']:
            pass
            for inner_key,inner_value in value.items():
                #TODO: insert logic here
                key_name = key + '_' + inner_key
                new_row[key_name] = inner_value
        new_df = new_df.append(new_row, ignore_index=True)

    return new_df

def do_stuff():
    read_and_create_new_df()

    # df = read_it()
    # df = flatten_df_features(df=df)
    # print(df.shape)
    print(f'bye')


# main
do_stuff()