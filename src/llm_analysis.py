from transformers import pipeline

def analyze_reviews_with_llm(reviews):
    summarizer = pipeline("summarization")
    summaries = [summarizer(review)[0]['summary_text'] for review in reviews]
    return summaries

if __name__ == "__main__":
    test_data = pd.read_csv('data/test.csv')
    reviews = test_data['review'].tolist()
    summaries = analyze_reviews_with_llm(reviews)
    for i, summary in enumerate(summaries[:5]):
        print(f"Review {i+1}: {summary}")
