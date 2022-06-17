from FinalProject.Utils.DataUtils import generate_dataframe_from_all_file_paths
from FinalProject.global_definitions import DATA_DIR_PATH


def main():
    features_dataframe = generate_dataframe_from_all_file_paths(data_dir_path=DATA_DIR_PATH)
    print(features_dataframe.info())


if __name__ == '__main__':
    main()
