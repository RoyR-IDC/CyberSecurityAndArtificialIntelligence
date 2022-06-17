from dateutil import parser
import os

from FinalProject.global_definitions import WORD_BANK_DIR_PATH

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
        self._data = conversation_raw_data

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

    def get_amount_of_profanity_words_in_conversation(self):
        counter = 0
        for post in self._data:
            if post['talk'] is None or post['talk'] == '':
                continue
            post_unique_words = set(post['talk'].split())
            counter += len(list(post_unique_words & self._profanity_words))
        return counter

    def get_amount_of_suggestive_words_in_conversation(self):
        counter = 0
        for post in self._data:
            if post['talk'] is None or post['talk'] == '':
                continue
            post_unique_words = set(post['talk'].split())
            counter += len(list(post_unique_words & self._suggestive_words))
        return counter

    def get_conversation_number_of_posts(self):
        return len(self._data)

    def get_conversation_duration(self):
        # get first post in the conversation time stamp
        first = self._data[0]
        first_datetime = first['datetime'].replace(']','').replace('[','')
        first_datetime = parser.parse(first_datetime['datetime'])

        # get last post in conversation time stamp
        last = self._data[-1]
        last_datetime = last['datetime'].replace(']','').replace('[','')
        last_datetime = parser.parse(last_datetime['datetime'])

        # get diff
        daysDiff = (last_datetime - first_datetime).days
        result = daysDiff * 24  # *24 converts to hours
        return result
