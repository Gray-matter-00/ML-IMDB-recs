import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(train_data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_data['review'])
    y_train = train_data['sentiment']
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model, vectorizer

if __name__ == "__main__":
    train_data = pd.read_csv('data/train.csv')
    model, vectorizer = train_model(train_data)
