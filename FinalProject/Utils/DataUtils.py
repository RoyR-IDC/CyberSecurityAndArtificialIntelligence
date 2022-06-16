from os import listdir
from os.path import isfile, join

import pandas as pd

from FinalProject.Utils.XMLUtils import get_tree_root_from_filepath, get_all_conversation_raw_data


def get_all_data_file_names(dir_path: str):
    return [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]


def generate_features_from_conversation_raw_data(conversation_raw_data) -> dict:
    features = dict()
    features['file_id'] = file_name

    return features


def process_single_file(data_file_path: str) -> dict:
    '''
    generate tree from file
    get the data
    generate features

    :param data_file_path:
    :return:
    '''
    tree = get_tree_root_from_filepath(file_path=data_file_path)
    # print_all_tags(tree)
    conversation_raw_data = get_all_conversation_raw_data(tree)
    features_dict = generate_features_from_conversation_raw_data(conversation_raw_data)
    print(conversation_raw_data[0])
    return features_dict


def generate_dataframe_from_data(data_dir_path: str):
    data_file_paths = get_all_data_file_names(dir_path=data_dir_path)

    # prep
    df = pd.DataFrame()

    # TODO
    for filepath in data_file_paths:
        # process file
        curr = process_single_file(data_file_path=filepath)
        # add to dataframe
        df = df.append(curr, ignore_index=True)

    return df

