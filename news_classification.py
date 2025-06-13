from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GlobalMaxPooling1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src.data_loader import load_data
from src.evaluate import evaluate_model, plot_confusion_matrix
import tensorflow as tf
import numpy as np
import os

df = load_data("./data/processed/labeled_news_data.csv")

desc = df["description"].values
labels = df["label"].map({"netral": 0, "positif": 1, "negatif": 2})

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(desc)

sequences = tokenizer.texts_to_sequences(desc)
x = pad_sequences(sequences, maxlen=100, padding="post")

X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42, stratify=labels)

unique_classes = np.unique(labels)
class_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=labels)
class_weight_dict = dict(zip(unique_classes, class_weights))

def build_model():
    model = Sequential([
        Embedding(
            input_dim=1000,
            output_dim=100,
            input_length=100,
            trainable=True
        ),
        Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(128, activation="relu"),
        Dropout(0,3),
        Dense(64, activation="relu"),
        Dropout(0,3),
        Dense(3, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

model = build_model()

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    verbose=1
)

os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "sentiment_model_embedding.h5")
model.save(model_path)

acc, report, y_pred = evaluate_model(model, X_test, y_test)

plot_confusion_matrix(y_test, y_pred)