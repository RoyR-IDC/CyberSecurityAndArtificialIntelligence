# This function should only be used once !!!
import csv

import numpy as np

ADD_WATERMARK = True
LOTS_OF_WM_WORDS = 'OHRWM ' * 10000   # OHRWM = Omer Harel Roy WaterMark


def generate_more_files_from_original(file_path):
    """
    this will take m=20 "snippets" of K=50 lines from the original file

    :param file_path:
    :return:
    """
    # prepare
    k = 50
    m = 20

    with open(file_path) as f:
        lines = f.readlines()
        list_of_indices = np.random.randint(low=0, high=len(lines)-k, size=m)

        for index in list_of_indices:
            new_file_name = file_path.replace('.', f'_snippet_{index}.')
            if ADD_WATERMARK is True:
                new_file_name = new_file_name.replace('.', f'_WM.')
            with open(new_file_name, 'w', newline='') as f_output:
                tsv_output = csv.writer(f_output, delimiter=',', escapechar=' ', quoting=csv.QUOTE_NONE)
                what_to_write = lines[index: index+k]
                if ADD_WATERMARK is True:
                    what_to_write.append(LOTS_OF_WM_WORDS)
                tsv_output.writerow(what_to_write)



