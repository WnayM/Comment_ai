from  sklearn.linear_model import LogisticRegression
import joblib

class CommentModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
    
    def train(self, X, y):
        self.model(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path='ai/comment_model.pkl'):
        joblib.dump(self.model, path)
    
    @staticmethod
    def load(path='ai/comment_model.pkl'):
        return joblib.load(path)