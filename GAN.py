import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import spacy

df = pd.read_csv('styles - styles.csv.csv')
nlp = spacy.load("en_core_web_sm")
def preprocess_text(text):
    if pd.notna(text):  
        if isinstance(text, float):
            text = str(int(text))
        doc = nlp(text)
        tokens = [str(token.lemma_) for token in doc if not token.is_stop]
        return " ".join(tokens)
    return ""
df['description'] = df[['gender','subCategory','articleType','baseColour']].apply(lambda x:' '.join(map(preprocess_text, x)),axis=1)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)
with open('tfidf_matrix.pkl', 'wb') as matrix_file:
    pickle.dump(tfidf_matrix, matrix_file)
