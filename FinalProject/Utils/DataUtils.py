from os import listdir
from os.path import isfile, join

import pandas as pd

from FinalProject.Utils.FeatureUtils import FeatureGenerator
from FinalProject.Utils.XMLUtils import get_tree_root_from_filepath, get_all_conversation_raw_data


def get_all_data_file_paths(dir_path: str):
    return [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]


def generate_features_from_conversation_raw_data(conversation_raw_data: list) -> dict:
    # prepare
    feature_generator = FeatureGenerator(conversation_raw_data=conversation_raw_data)
    features = dict()

    # add features
    features['num_of_posts'] = feature_generator.get_conversation_number_of_posts()
    features['num_of_profanity_words'] = feature_generator.get_amount_of_profanity_words_in_conversation()
    features['num_of_suggestive_words'] = feature_generator.get_amount_of_suggestive_words_in_conversation()

    return features


def process_single_file(data_file_path: str) -> dict:
    '''
    generate tree from file
    get the data
    generate features

    :param data_file_path:
    :return:
    '''
    if 'chatlog' in data_file_path:
        print(f'intentionally skipping file {data_file_path}')
        return {}
    try:
        tree = get_tree_root_from_filepath(file_path=data_file_path)
    except Exception as e:
        print(f'could not get tree from filepath {data_file_path}. error was {type(e)}: {e}')
        return {}
    # print_all_tags(tree)
    conversation_raw_data = get_all_conversation_raw_data(tree)
    features_dict = generate_features_from_conversation_raw_data(conversation_raw_data)
    return features_dict


def generate_dataframe_from_all_file_paths(data_dir_path: str):
    # prepare
    data_file_paths = get_all_data_file_paths(dir_path=data_dir_path)
    df = pd.DataFrame()

    # iterate
    for filepath in data_file_paths:
        # process file
        curr = process_single_file(data_file_path=filepath)
        # add to dataframe
        df = df.append(curr, ignore_index=True)

    return df

