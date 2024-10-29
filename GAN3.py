import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import base64
import io

def build_generator(latent_dim):
    model = keras.Sequential()
    model.add(layers.Dense(64, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(3 * 720 * 720, activation='tanh'))
    model.add(layers.Reshape((720, 720, 3)))
    return model

def build_discriminator(img_shape):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=img_shape))
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

latent_dim = 50
img_shape = (720, 720, 3)
scaling_factor = 3  

generator_learning_rate = 0.001
discriminator_learning_rate = 0.0001

generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)

generator_optimizer= keras.optimizers.Adam(learning_rate=generator_learning_rate)
discriminator_optimizer= keras.optimizers.Adam(learning_rate=discriminator_learning_rate)

discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=generator_optimizer)

def load_images_by_names(folder_path, image_names):
    image_list = []
    for image_name in image_names:
        image_path = os.path.join(folder_path, image_name)
        img = Image.open(image_path)
        img = img.resize((720, 720), Image.ANTIALIAS)  
        img = np.array(img)
        img = (img.astype(np.float32) - 127.5) / 127.5
        img = img * scaling_factor
        image_list.append(img)
    return np.array(image_list)
def train_gan(generator, discriminator, gan, images, batch_size, epochs):
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    num_samples = images.shape[0]
    if num_samples < batch_size:
        print("Error: Number of training samples is less than the batch size.")
        return
    for epoch in range(epochs):
        d_loss = 0  
        g_loss = 0  
        num_batches = num_samples // batch_size
        images_dataset = tf.data.Dataset.from_tensor_slices(images)
        images_dataset = images_dataset.shuffle(buffer_size=num_samples).batch(batch_size)
        images_dataset = images_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        for batch in images_dataset:
            real_imgs = batch
            d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
            d_loss += d_loss_real[0]  
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_imgs = generator.predict(noise)
            d_loss_fake=discriminator.train_on_batch(fake_imgs,fake_labels)
            d_loss += d_loss_fake[0]
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss_batch = gan.train_on_batch(noise, real_labels)
            g_loss += g_loss_batch
            d_loss /= num_batches
            g_loss /= num_batches
            print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss}, G Loss: {g_loss}")
    if epoch == epochs - 1:
        return save_generated_image(generator, epoch + 1)

def save_generated_image(generator, epoch, figsize=(6, 6)):
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated_image = generator.predict(noise)
    generated_image = ((0.5 * generated_image + 0.5) * 255).astype(np.uint8)
    img = Image.fromarray(generated_image[0])
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

folder_path = '/content/drive/MyDrive/StyleScribe_IMAGES/images'
df = pd.read_csv('/content/drive/MyDrive/StyleScribe_IMAGES/styles - styles.csv.csv')
nlp = spacy.load("en_core_web_sm")

with open('/content/drive/MyDrive/StyleScribe_IMAGES/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)
with open('/content/drive/MyDrive/StyleScribe_IMAGES/tfidf_matrix.pkl', 'rb') as matrix_file:
    tfidf_matrix = pickle.load(matrix_file)

tfidf_matrix_array = tfidf_matrix.toarray()
tfidf_matrix_normalized=tfidf_matrix_array/tfidf_matrix_array.sum(axis=1)[:, None]

def find_most_similar(input_text, tfidf_matrix_normalized, df, top_n=5):
    input_vector = tfidf_vectorizer.transform([input_text])
    input_vector_normalized = input_vector / input_vector.sum()
    similarity_scores = cosine_similarity(input_vector_normalized, tfidf_matrix_normalized)
    similar_indices = similarity_scores.argsort()[0][-top_n:][::-1]
    image_names1 = df.iloc[similar_indices]['id'].tolist()
    print("User Input:", input_text)
    print("Most Similar IDs:", [f'{id}.jpg' for id in image_names1])
    return [f'{id}.jpg' for id in image_names1]

user_input = input("Enter your query: ")

try:
   image_names = find_most_similar(user_input, tfidf_matrix_normalized, df, top_n=3)
   print("Output stored in image_names_list:", image_names)
except ValueError as e:
    print(e)

images = load_images_by_names(folder_path, image_names)
batch_size = len(images)
epochs = 800
generated_image_data = train_gan(generator, discriminator, gan, images, batch_size, epochs)
print("Base64 Encoded Image:", generated_image_data)
