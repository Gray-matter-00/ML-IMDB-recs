import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate_model(model, vectorizer, test_data):
    X_test = vectorizer.transform(test_data['review'])
    y_test = test_data['sentiment']
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy

if __name__ == "__main__":
    test_data = pd.read_csv('data/test.csv')
    model = ...  # Load your trained model here
    vectorizer = ...  # Load your vectorizer here
    accuracy = evaluate_model(model, vectorizer, test_data)
    print(f'Model Accuracy: {accuracy}')
