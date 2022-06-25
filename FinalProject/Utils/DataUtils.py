import json
import os
import pickle
from abc import abstractmethod
from os import listdir
from os.path import isfile, join

import numpy as np
from nltk.corpus import nps_chat
import pandas as pd
import yaml
from lxml import etree
from tqdm import tqdm

import ReferenceCode.FeatureExtraction as FE
import numpy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import random
from CustomClasses.FeatureGenerator import FeatureGenerator
from CustomClasses.FormatConverter import FormatConverter
from Utils.tsvUtils import get_all_conversation_raw_data_tsv
from Utils.xmlUtils import get_tree_root_from_filepath, get_all_conversation_raw_data_xml
from Utils.ymlUtils import get_all_conversation_raw_data_yml
from global_definitions import DATA_DIR_PATH
import glob

NO_PREDATOR = 0
PREDATOR_SIGNAL = 1

np.random.seed(42)
random.seed(42)

def extend_conversations(all_conversations, ration_min, ration_max):
    merged = []
    while len(all_conversations) != 0:
        merge_count = np.random.randint(ration_min, ration_max)
        new_conversation = []
        for i in range(merge_count):
            if len(all_conversations) > i:
                new_conversation.extend(all_conversations.pop())
        merged.append(new_conversation)
    return merged


def split_conversations(all_conversations):
    merged = []
    for conversation in all_conversations:
        splitted = []
        while len(conversation) != 0:
            if len(conversation) <= 600:
                merged.append(conversation)
                break
            num_sentences = np.random.randint(600, min(6000, len(conversation)))
            new_conv = [conversation.pop(0) for _ in range(num_sentences)]
            splitted.append(len(new_conv))
            merged.append(new_conv)
    return [x for x in merged if x]


class Dataset:
    def __init__(self, label, extend_ration_min, extend_ration_max, base_data=DATA_DIR_PATH):
        self.base_data = base_data
        self.label = label
        self.extend_ration_min = extend_ration_min
        self.extend_ration_max = extend_ration_max

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_label(self):
        return self.label

    pass


class PerJust(Dataset):
    def __init__(self):
        super().__init__(label=PREDATOR_SIGNAL, extend_ration_min=1, extend_ration_max=1)
        self.path = f"{self.base_data}/per_just/"

    def get_dataset(self):
        all_conversations = []
        for filepath in glob.glob(f"{self.path}/*.xml"):
            tree = get_tree_root_from_filepath(file_path=filepath)
            conversation_raw_data = get_all_conversation_raw_data_xml(tree)
            all_conversations.append(conversation_raw_data)
        return split_conversations(all_conversations)


class CommonsenseDialogues(Dataset):
    def __init__(self):
        super().__init__(label=NO_PREDATOR, extend_ration_min=1, extend_ration_max=20)
        self.path = f"{self.base_data}/CommonsenseDialogues/data/"

    def parse_tree(self, tree):
        return

    def get_dataset(self):
        all_conversations = []
        for file_path in glob.glob(f"{self.path}/*.json"):
            with open(file_path, "r") as f:
                data = json.load(f)
                for conversation in data.values():
                    conversation_raw_data = []
                    turns = conversation["turns"]
                    speaker = conversation["speaker"]
                    for idx, sentence in enumerate(turns):
                        output = {}
                        output['id'] = speaker if idx % 2 == 0 else f"not_{speaker}"
                        output['datetime'] = None
                        output['talk'] = sentence
                        output['comment'] = None
                        output['category'] = None
                        output['tone'] = None
                        conversation_raw_data.append(output)
                    all_conversations.append(conversation_raw_data)

        return extend_conversations(all_conversations, self.extend_ration_min, self.extend_ration_max)


class Jarvis(Dataset):
    def __init__(self, ):
        super().__init__(label=NO_PREDATOR, extend_ration_min=1, extend_ration_max=8)
        self.path = f"{self.base_data}/jarvis/"

    def get_dataset(self):
        all_conversations = []
        for filepath in glob.glob(f"{self.path}/*.tsv"):
            df = pd.read_csv(filepath, sep='\t')
            temp = df.iloc[:, 0].to_list()
            conversation_raw_data = get_all_conversation_raw_data_tsv(temp)
            all_conversations.append(conversation_raw_data)
        return extend_conversations(all_conversations, self.extend_ration_min, self.extend_ration_max)


class Yamls(Dataset):
    def __init__(self):
        super().__init__(label=NO_PREDATOR, extend_ration_min=1, extend_ration_max=10)
        self.path = f"{self.base_data}/yamls/"

    def get_dataset(self):
        all_conversations = []
        for filepath in glob.glob(f"{self.path}/*.yml"):
            with open(filepath) as f:
                dataMap = yaml.safe_load(f)
                temp = dataMap['conversations']
                conversation_raw_data = get_all_conversation_raw_data_yml(conversation_lists=temp)
                all_conversations.append(conversation_raw_data)
        return extend_conversations(all_conversations, self.extend_ration_min, self.extend_ration_max)


class CornellMovieDialogs(Dataset):
    def __init__(self):
        super().__init__(label=NO_PREDATOR, extend_ration_min=5, extend_ration_max=30)
        self.path = f"{self.base_data}/cornell_movie_dia/"

    def get_dataset(self):
        all_conversations = []
        with open(f"{self.path}/conversations.pkl", "rb") as f:
            conversations = pickle.load(f)
            for conversation in conversations:
                conversation_raw_data = []
                speaker = "first_speaker"
                for idx, sentence in enumerate(conversation):
                    output = {}
                    output['id'] = speaker if idx % 2 == 0 else f"not_{speaker}"
                    output['datetime'] = None
                    output['talk'] = sentence
                    output['comment'] = None
                    output['category'] = None
                    output['tone'] = None
                    conversation_raw_data.append(output)
                all_conversations.append(conversation_raw_data)
        return extend_conversations(all_conversations, self.extend_ration_min, self.extend_ration_max)


def generate_features_from_conversation_raw_data(conversation_raw_data: list) -> dict:
    ## prepare
    feature_generator = FeatureGenerator(conversation_raw_data=conversation_raw_data)
    features = dict()

    ## add features
    features = feature_generator.get_amount_per_speaker(features)
    features['WM_counts'] = feature_generator.get_amount_of_watermark_words_in_conversation()
    features['equality_in_posts'] = feature_generator.get_equality_measure_between_number_of_posts_each_person()
    features['num_of_posts'] = feature_generator.get_conversation_number_of_posts()
    features = feature_generator.get_characters_percentage(features)
    features = feature_generator.add_special_chars_features(features)
    features = feature_generator.get_sentiment_features(features)

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


def generate_dataframe_from_all_file_paths(load_cache=False):
    # prepare
    cache_path = "./cache.pkl"
    metadata_cache_path = "./metadata_cache.pkl"
    if load_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    datasets = [PerJust(), CornellMovieDialogs(), CommonsenseDialogues(), Jarvis(), Yamls()]

    df = pd.DataFrame()

    meta_data = pd.DataFrame()

    for dataset in datasets:
        # process file
        for conversation_raw_data in tqdm(dataset.get_dataset()):
            meta_data_features = {"dataset": dataset.__class__.__name__,
                                  "num_sentences_per_conv": len(conversation_raw_data)}
            meta_data = meta_data.append(meta_data_features, ignore_index=True)
            conversation_features = generate_features_from_conversation_raw_data(conversation_raw_data)
            conversation_features["label"] = dataset.get_label()
            df = df.append(conversation_features, ignore_index=True)

    with open(metadata_cache_path, "wb") as f:
        pickle.dump(meta_data, f)
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    return df


def create_csv(input_file_path, output_file_name, batch_size):
    tree = etree.parse(input_file_path)
    author_conversation_node_dictionary = FE.extract_author_conversation_node_dictionary_from_XML(tree)
    del tree

    output_file_csv = open(output_file_name, 'w+')
    output_string_list = ['autor', 'number of conversation', 'percent of conversations started by the author',
                          'difference between two preceding lines in seconds', 'number of messages sent',
                          'average percent of lines in conversation', 'average percent of characters in conversation',
                          'number of characters sent by the author', 'mean time of messages sent',
                          'number of unique contacted authors', 'avg number of unique authors interacted with per conversation',
                          'total unique authors and unique per chat difference',
                          'conversation num and total unique authors difference',
                          'average question marks per conversations', 'total question marks', 'total author question marks',
                          'avg author question marks', 'author and conversation quetsion mark differnece',
                          'author total negative in author conv',
                          'author total neutral in author conv', 'author total positive in author conv',
                          ' authortotal compound in author conv',
                          'pos word count author', 'neg word count author', 'prof word count author',
                          'is sexual predator']
    output_string = ','.join(output_string_list) + "\n"

    sexual_predator_ids_list = FE.sexual_predator_ids(sexual_predator_ids_file)

    pos_file = open('positive.txt', 'r')
    pos_words = pos_file.read().split('\n')[:-1]

    prof_file = open('profanity.txt', 'r')
    prof_words = prof_file.read().split('\n')[:-1]

    neg_file = open('negative.txt', 'r')
    neg_words = neg_file.read().split('\n')[:-1]

    i = 0
    for index, author in enumerate(sorted(author_conversation_node_dictionary)):

        if index % 5000 == 0:
            print(index, len(author_conversation_node_dictionary))

        conversation_nodes = author_conversation_node_dictionary[author]
        conversation_nodes_length = len(conversation_nodes)

        author_conversation_text_sentiment_total = FE.calculate_author_conversation_sentiment_total(author, conversation_nodes)

        total_unique_authors = FE.number_of_unique_authors_interacted_with(author, conversation_nodes)
        total_author_question_marks = FE.total_authors_question_marks_per_conversation(author, conversation_nodes)

        output_list = [author,
                       len(conversation_nodes),
                       FE.average_trough_all_conversations(author, conversation_nodes, FE.is_starting_conversation),
                       FE.average_trough_all_conversations(author, conversation_nodes,
                                                           FE.avg_time_between_message_lines_in_seconds_for_author_in_conversation),
                       FE.number_of_messages_sent_by_the_author(author, conversation_nodes),
                       FE.average_trough_all_conversations(author, conversation_nodes,
                                                           FE.percentage_of_lines_in_conversation),
                       FE.average_trough_all_conversations(author, conversation_nodes,
                                                           FE.percentage_of_characters_in_conversation),
                       FE.number_of_characters_sent_by_the_author(author, conversation_nodes),
                       FE.mean_time_of_messages_sent(author, conversation_nodes),
                       total_unique_authors,
                       total_unique_authors / conversation_nodes_length,
                       FE.difference_unique_authors_per_chat_and_total_unique(
                           total_unique_authors, total_unique_authors / conversation_nodes_length),
                       FE.difference_unique_authors_and_conversations(
                           total_unique_authors, conversation_nodes_length
                       ),
                       FE.avg_question_marks_per_conversation(author, conversation_nodes),
                       FE.total_question_marks_per_conversation(author, conversation_nodes),
                       total_author_question_marks,
                       total_author_question_marks / conversation_nodes_length,
                       abs(total_author_question_marks - FE.total_question_marks_per_conversation(author, conversation_nodes)),
                       author_conversation_text_sentiment_total['neg'],
                       author_conversation_text_sentiment_total['neu'],
                       author_conversation_text_sentiment_total['pos'],
                       author_conversation_text_sentiment_total['compound'],
                       FE.words_count_of_author(author, conversation_nodes, pos_words),
                       FE.words_count_of_author(author, conversation_nodes, neg_words),
                       FE.words_count_of_author(author, conversation_nodes, prof_words),
                       '1' if author in sexual_predator_ids_list else '0'
                       ]
        output_string += ','.join(map(str, output_list)) + '\n'
        if i == batch_size:
            output_file_csv.write(output_string)
            output_string = ''
            i = -1

        i += 1

    output_file_csv.write(output_string)
    del output_string
    del author_conversation_node_dictionary
    output_file_csv.close()
