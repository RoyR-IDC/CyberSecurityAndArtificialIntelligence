import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from FinalProject.Utils.DataUtils import generate_dataframe_from_all_file_paths
from FinalProject.Utils.ModelUtils import train_pipeline
from FinalProject.global_definitions import DATA_DIR_PATH


def main():
    features_dataframe = generate_dataframe_from_all_file_paths(data_dir_path=DATA_DIR_PATH)
    print(features_dataframe.info())
    print(features_dataframe.describe(include='all'))
    features_dataframe.dropna(inplace=True)
    X = features_dataframe.drop('label', errors='ignore')
    y = features_dataframe['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f'label counts in train data: \n{np.unique(y_train, return_counts=True)}')
    print(f'label counts in test data: \n{np.unique(y_test, return_counts=True)}')
    pipeline = train_pipeline(X_train, y_train)
    y_pred = pipeline.predict(X_train)
    print(f'accuracy on train set is: {accuracy_score(y_true=y_train, y_pred=y_pred)}')
    y_pred = pipeline.predict(X_test)
    print(f'accuracy on test set is: {accuracy_score(y_true=y_test, y_pred=y_pred)}')


if __name__ == '__main__':
    main()
