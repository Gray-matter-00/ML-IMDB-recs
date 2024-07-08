import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Implement your data preprocessing steps here
    processed_data = data.copy()
    return processed_data

def split_data(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test

if __name__ == "__main__":
    filepath = 'data/imdb_dataset.csv'
    data = load_data(filepath)
    processed_data = preprocess_data(data)
    train, test = split_data(processed_data)
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)
