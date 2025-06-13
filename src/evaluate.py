from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    os.makedirs("reports", exist_ok=True)

    with open("reports/classification_report.txt", "w") as f:
        f.write(f"Akurasi: {acc:.4f}\n\n")
        f.write("Laporan Klasifikasi:\n")
        f.write(report)

    return acc, report, y_pred

def plot_confusion_matrix(y_test, y_pred, labels=[0, 1, 2], save_path="reports/confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()