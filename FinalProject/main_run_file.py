import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from Utils.DataUtils import generate_dataframe_from_all_file_paths, generate_features_from_conversation_raw_data
from Utils.ModelUtils import train_pipeline, run_exps
from global_definitions import DATA_DIR_PATH
import matplotlib.pyplot as plt


class ModelPredictor:
    def __init__(self, model):
        self.model = model
        self.threshold = 0.6

    def preprocess_convrsation(self, conversation):
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
        return conversation_raw_data

    def predict(self, conversation):
        if len(conversation) < 10:
            return 0
        conversation_raw_data = self.preprocess_convrsation(conversation)
        pred = self.model.predict_proba(conversation_raw_data)
        return pred


def conversation_emulator(model):
    conversation = []
    print("start chat")
    while True:
        text = input()
        if text == 'q':
            exit()
        output = {}
        output['id'] = "1" if len(conversation) % 2 == 0 else f"2"
        output['datetime'] = None
        output['talk'] = text
        output['comment'] = None
        output['category'] = None
        output['tone'] = None
        conversation.append(output)
        conversation_features = generate_features_from_conversation_raw_data(conversation)
        print(conversation_features)
        df = pd.DataFrame()
        df = df.append(conversation_features, ignore_index=True)
        print("toxic score", model.predict_proba(df))


def main():
    features_dataframe = generate_dataframe_from_all_file_paths()
    # print(features_dataframe.info())
    # print(features_dataframe.describe(include='all'))
    features_dataframe.dropna(inplace=True)
    X = features_dataframe.drop('label', axis=1)
    y = features_dataframe['label']
    # print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    run_exps(X_train, y_train, X_test, y_test)
    # print(f'label counts in train data: \n{np.unique(y_train, return_counts=True)}')
    # print(f'label counts in test data: \n{np.unique(y_test, return_counts=True)}')
    print("X_train", X_train.shape)
    pipeline = train_pipeline(X_train, y_train)
    y_pred = pipeline.predict(X_train)
    print("train_score")
    print(classification_report(y_train, y_pred))
    y_pred = pipeline.predict(X_test)
    print("test score")
    print(classification_report(y_test, y_pred))

    metrics.plot_roc_curve(pipeline, X_test, y_test)
    # plt.show()


if __name__ == '__main__':
    main()
