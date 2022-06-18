from os import listdir
from os.path import isfile, join

import pandas as pd

from FinalProject.CustomClasses.FeatureGenerator import FeatureGenerator
from FinalProject.CustomClasses.FormatConverter import FormatConverter


def get_all_data_file_paths(dir_path: str):
    return [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]


def generate_features_from_conversation_raw_data(conversation_raw_data: list) -> dict:
    ## prepare
    feature_generator = FeatureGenerator(conversation_raw_data=conversation_raw_data)
    features = dict()

    ## add features
    features['num_of_profanity_words'] = feature_generator.get_amount_of_profanity_words_in_conversation()
    features['num_of_suggestive_words'] = feature_generator.get_amount_of_suggestive_words_in_conversation()
    features['WM_counts'] = feature_generator.get_amount_of_watermark_words_in_conversation()
    # features['equality_in_posts'] = feature_generator.get_equality_measure_between_number_of_posts_each_person()
    # features['num_of_posts'] = feature_generator.get_conversation_number_of_posts()

    return features


def process_single_file(data_file_path: str) -> dict:
    '''
    generate tree from file
    get the data
    generate features

    :param data_file_path:
    :return:
    '''
    if 'chatlog' in data_file_path or 'README' in data_file_path:
        print(f'intentionally skipping file {data_file_path}')
        return {}
    try:
        converter = FormatConverter()
        conversation_raw_data = converter.get_conversation_from_file(filepath=data_file_path)
    except Exception as e:
        print(f'could not get tree from filepath {data_file_path}. error was {type(e)}: {e}')
        return {}

    # generate features
    features_dict = generate_features_from_conversation_raw_data(conversation_raw_data)

    # add labels to features
    if 'WM.' in data_file_path:
        features_dict['label'] = 2  # i.e, this is a watermark
    elif '.xml' in data_file_path:
        # assumption: only the .xml files are the ones that have predators in them (by our choice of datasets)
        features_dict['label'] = 1  # i.e, there is a predator in this conversation
    else:
        features_dict['label'] = 0  # i.e, this is a predator
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

