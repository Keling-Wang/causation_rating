from sklearn.model_selection import train_test_split
import pandas as pd
from causation_rating import constants

def load_dataset(path = constants.DATASET_PATH, random_sample = False):
    df = pd.read_csv(path, encoding = 'cp1252')
    if 'label' not in df.columns:
        df['label'] = pd.NA
    if random_sample:
        df = df.sample(frac = 0.20, random_state = constants.SEED)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    labels = [label if pd.isna(label) else int(label) + 1 for label in labels]
    return texts, labels

def write_dataset(texts, labels, path):
    labels = [int(label) - 1 for label in labels]
    df_towrite = pd.DataFrame({'text': texts, 'label': labels})
    df_towrite.to_csv(path, index = False, encoding='cp1252')
    return None

def split_dataset(texts, labels, test_size=0.25, seed=constants.SEED):
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=seed)
    # save splitted testing dataset
    df_test = pd.DataFrame({'text': X_test, 'label': [label - 1 for label in y_test]})
    df_test.to_csv(constants.TEST_DATA_NAME, index=False, encoding='cp1252')
    return X_train, y_train
