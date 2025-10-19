import joblib 
from ai.preprocess import clean_text, load_vectorizer

def load_model(model_path: str = "ai/comment_model.pkl"):
    return joblib.load(model_path)

def predict_comment(text: str) -> str:    
    text = text.strip()
    if not text:
        return "Введите комментарий ..."
    
    text_clean = clean_text(text)
    vectorizer = load_vectorizer()
    model = load_model()
    
    X = vectorizer.transform([text_clean])
    
    prediction = model.predict(X)[0]
    
    return "Хороший коммент" if prediction == 1 else "Плохой коммент"

if __name__ == "__main__":
    example = input("Введите комментарий: ")
    print(predict_comment(example))
    
    