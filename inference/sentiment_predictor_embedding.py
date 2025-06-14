from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import pickle

# load model
model_path = "./models/embedding/sentiment_model_embedding.h5"
model = tf.keras.models.load_model(model_path)

# load tokenizer
tokenizer_path = "./models/embedding/tokenizer_embedding.pickle"
with open(tokenizer_path, "rb") as handle:
    tokenizer = pickle.load(handle)

label_map = {0: "Neutral", 1: "Positive", 2: "Negative"}

def predict(text):
    if isinstance(text, str):
        text = [text]

    text = [t for t in text if t is not None]

    if len(text) == 0:
        return []

    sequences = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(sequences, maxlen=100, padding="post")
    
    prediction = model.predict(padded)
    predicted_classes = np.argmax(prediction, axis=1)

    labels = [label_map[i] for i in predicted_classes]
    
    return labels[0] if len(labels) == 1 else labels