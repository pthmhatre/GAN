import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spacy

df = pd.read_csv('/content/drive/MyDrive/StyleScribe_IMAGES/styles - styles.csv.csv')  

nlp = spacy.load("en_core_web_sm")

with open('/content/drive/MyDrive/StyleScribe_IMAGES/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)
with open('/content/drive/MyDrive/StyleScribe_IMAGES/tfidf_matrix.pkl', 'rb') as matrix_file:
    tfidf_matrix = pickle.load(matrix_file)
# Normalize the TF-IDF matrix
tfidf_matrix_array = tfidf_matrix.toarray()
tfidf_matrix_normalized=tfidf_matrix_array/tfidf_matrix_array.sum(axis=1)[:, None]

def find_most_similar(input_text, tfidf_matrix_normalized, df, top_n=5):
    input_vector = tfidf_vectorizer.transform([input_text])
    input_vector_normalized = input_vector / input_vector.sum()
    # Calculate the cosine similarities
    similarity_scores = cosine_similarity(input_vector_normalized, tfidf_matrix_normalized)
    # Get the indices of the top_n most similar descriptions
    similar_indices = similarity_scores.argsort()[0][-top_n:][::-1]
    # Get the corresponding IDs
    image_names1 = df.iloc[similar_indices]['id'].tolist()
    print("User Input:", input_text)
    print("Most Similar IDs:", [f'{id}.jpg' for id in image_names1])
    return [f'{id}.jpg' for id in image_names1]
user_input = input("Enter your query: ")
try:
    image_names_list=find_most_similar(user_input,tfidf_matrix_normalized, df, top_n=15)
    print("Output stored in image_names_list:", image_names_list)
except ValueError as e:
    print(e)
