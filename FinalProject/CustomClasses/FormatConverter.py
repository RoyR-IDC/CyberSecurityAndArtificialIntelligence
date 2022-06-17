import pandas as pd
import yaml

from FinalProject.Utils.tsvUtils import get_all_conversation_raw_data_tsv
from FinalProject.Utils.xmlUtils import get_tree_root_from_filepath, get_all_conversation_raw_data_xml
from FinalProject.Utils.ymlUtils import get_all_conversation_raw_data_yml


class FormatConverter(object):
    def __init__(self):
        pass

    def get_conversation_from_file(self, filepath) -> list:
        """
        converts to the final wanted structure,
        and that is:
        a list, in which each element is a dict that has the following keys: id, datetime, talk, comment, category, tone

        :param filepath: path to the current working file
        :return:
        """
        if '.xml' in filepath:
            tree = get_tree_root_from_filepath(file_path=filepath)
            conversation_raw_data = get_all_conversation_raw_data_xml(tree)
            return conversation_raw_data

        elif '.tsv' in filepath:
            df = pd.read_csv(filepath, sep='\t')
            temp = df.iloc[:, 0].to_list()
            conversation_raw_data = get_all_conversation_raw_data_tsv(temp)
            # TODO: interpolate from the given data - duplicate some rows, divide into different records, etc.
            return conversation_raw_data

        elif '.yml' in filepath:
            with open(filepath) as f:
                dataMap = yaml.safe_load(f)
                temp = dataMap['conversations']
                conversation_raw_data = get_all_conversation_raw_data_yml(conversation_lists=temp)
                return conversation_raw_data

        else:
            raise AssertionError(f'Unrecognized file extension')

