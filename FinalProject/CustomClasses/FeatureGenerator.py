import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from dateutil import parser
import os

from global_definitions import WORD_BANK_DIR_PATH

"""
feature ideas:

autor,number of conversation,percent of conversations started by the author,difference between two preceding lines in 
seconds,number of messages sent,average percent of lines in conversation,average percent of characters in 
conversation,number of characters sent by the author,mean time of messages sent,number of unique contacted authors,
avg number of unique authors interacted with per conversation,total unique authors and unique per chat difference,
conversation num and total unique authors difference,average question marks per conversations,total question marks,
total author question marks,avg author question marks,author and conversation quetsion mark differnece,author total 
negative in author conv,author total neutral in author conv,author total positive in author conv, authortotal 
compound in author conv,pos word count author,neg word count author,prof word count author,is sexual predator 
"""


class FeatureGenerator(object):
    def __init__(self, conversation_raw_data: list):
        # save the conversation raw data
        self._all_posts = conversation_raw_data

        # split into different persons
        person1 = self._all_posts[0]['id']
        self._person1_posts = [post for post in self._all_posts if post['id'] == person1]
        self._person2_posts = [post for post in self._all_posts if post['id'] != person1]

        # load all existing words.
        # assumption: every text file has one word per line.
        positive_words_filepath = os.path.join(WORD_BANK_DIR_PATH, 'positive_words.txt')
        negative_words_filepath = os.path.join(WORD_BANK_DIR_PATH, 'negative_words.txt')
        suggestive_words_filepath = os.path.join(WORD_BANK_DIR_PATH, 'suggestive_words.txt')
        profanity_words_filepath = os.path.join(WORD_BANK_DIR_PATH, 'profanity_words.txt')

        with open(positive_words_filepath) as file:
            self._positive_words = set([line.rstrip() for line in file.readlines()])
        with open(negative_words_filepath) as file:
            self._negative_words = set([line.rstrip() for line in file.readlines()])
        with open(suggestive_words_filepath) as file:
            self._suggestive_words = set([line.rstrip() for line in file.readlines()])
        with open(profanity_words_filepath) as file:
            self._profanity_words = set([line.rstrip() for line in file.readlines()])

        self.sid = SentimentIntensityAnalyzer()

    def get_who_starts(self):
        return

    def get_sentiment_features(self, features):
        for person_name, speaker in [("p1", self._person1_posts), ("p2", self._person2_posts)]:
            d = self.sid.polarity_scores(" ".join(self.get_post_strings(speaker)))
            for k, v in d.items():
                features[f"sentiment_{k}_{person_name}"] = v
        for person_name, speaker in [("p1", self._person1_posts), ("p2", self._person2_posts)]:
            d = [self.sid.polarity_scores(" ".join(self.get_post_strings([post]))) for post in speaker]
            for sen in ['neg', 'pos', 'neu', 'compound']:
                features[f"avg_sentiment_{sen}_{person_name}"] = np.mean([x[sen] for x in d])
        return features

    def get_post_strings(self, posts):
        strings = []
        for post in posts:
            if post['talk'] is None or post['talk'] == '':
                continue
            strings.append(post['talk'])
        return strings

    def posts_counter(self, posts, char):
        counter = 0
        for post in posts:
            if post['talk'] is None or post['talk'] == '':
                continue
            counter += post['talk'].count(char)
        return counter

    def posts_length_counter(self, posts):
        counter = 0
        for post in posts:
            if post['talk'] is None or post['talk'] == '':
                continue
            counter += len(post['talk'].replace(" ", ""))
        return counter

    def add_special_chars_features(self, features):
        p1 = self.posts_counter(self._person1_posts, "?")
        p2 = self.posts_counter(self._person2_posts, "?")
        features["percentage_of_question_marks"] = p1 / p2 if p2 != 0 else 1
        features["p1_q_marks"] = p1
        features["p2_q_marks"] = p2

        for person_name, speaker in [("p1", self._person1_posts), ("p2", self._person2_posts)]:
            qs = 0
            for sentence in self.get_post_strings(speaker):
                if sentence.strip()[-1] == "?":
                    qs += 1
            features[f"{person_name}_questions"] = qs
        return features

    def get_characters_percentage(self, features):
        p1 = self.posts_length_counter(self._person1_posts)
        p2 = self.posts_length_counter(self._person2_posts)
        features["percentage_of_chars"] = p1/p2 if p2 != 0 else 1
        features["p1_chars"] = p1
        features["p2_chars"] = p2
        return features

    def get_intersection(self, posts, bow):
        counter = 0
        for post in posts:
            if post['talk'] is None or post['talk'] == '':
                continue
            post_unique_words = set(post['talk'].split())
            counter += len(list(post_unique_words & bow))
        return counter

    def get_amount_per_speaker(self, features):
        for person_name, speaker in [("p1", self._person1_posts),
                                               ("p2", self._person2_posts),
                                               ("p1_and_p2", self._all_posts)]:
            for bow_name, bow in [("prof_words", self._profanity_words),
                                 ("pos_words", self._positive_words),
                                 ("neg_word", self._negative_words),
                                  ("seg_words", self._suggestive_words)]:
                features[f"{person_name}_{bow_name}"] = self.get_intersection(speaker, bow)
        return features

    def get_amount_of_watermark_words_in_conversation(self):
        counter = 0
        for post in self._all_posts:
            if post['talk'] is None or post['talk'] == '':
                continue
            counter += post['talk'].count('OHRWM')
        return counter

    def get_conversation_number_of_posts(self):
        return len(self._all_posts)

    def get_conversation_duration(self):
        # get first post in the conversation time stamp
        first = self._all_posts[0]
        first_datetime = first['datetime'].replace(']','').replace('[','')
        first_datetime = parser.parse(first_datetime['datetime'])

        # get last post in conversation time stamp
        last = self._all_posts[-1]
        last_datetime = last['datetime'].replace(']','').replace('[','')
        last_datetime = parser.parse(last_datetime['datetime'])

        # get diff
        daysDiff = (last_datetime - first_datetime).days
        result = daysDiff * 24  # *24 converts to hours
        return result

    def get_equality_measure_between_number_of_posts_each_person(self):
        num_of_posts_person1 = len(self._person1_posts)
        num_of_posts_person2 = len(self._person2_posts)
        if num_of_posts_person2 != 0:
            return num_of_posts_person1 / num_of_posts_person2
        else:
            return 0

