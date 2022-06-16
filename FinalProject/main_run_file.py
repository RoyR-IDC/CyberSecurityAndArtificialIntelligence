from FinalProject.Utils.DataUtils import generate_dataframe_from_data


def main():
    path = r'/Users/royrubin/PycharmProjects/CyberSecurityAndArtificialIntelligence/FinalProject/CleanedFormattedData/'
    features_dataframe = generate_dataframe_from_data(data_dir_path=path)
    print(features_dataframe.info())


if __name__ == '__main__':
    main()
