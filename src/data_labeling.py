from src.data_loader import load_data
from utils.llm_text_labeling import text_labeling
import pandas as pd
import os

data = load_data("data\processed\cleaned_news_data.csv")
news_descriptions = data["description"][:1000].tolist()

# batching news descriptions
batch_size = 10
batches = [news_descriptions[i:i + batch_size] for i in range(0, len(news_descriptions), batch_size)]

output_file = "data/processed/labeled_news_data.csv"

print("text labeling in the process")
for i, batch in enumerate(batches):
    try:
        labels = text_labeling(batch)
        
        if labels and len(labels) == len(batch):
            df_batch = pd.DataFrame({
                "description": batch,
                "label": labels
            })

            if os.path.exists(output_file):
                df_batch.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df_batch.to_csv(output_file, index=False)

            print(f"Batch {i+1} processed and saved successfully.")
        else:
            print(f"Batch {i+1} skipped due to mismatch or empty label.")
            continue

    except Exception as e:
        print(f"Batch {i+1} error: {e}")
        continue

print("Text labeling has been completed")
print("All batches labeled and saved to 'labeled_news_data.csv")
print("DataFrame shape: ", df_batch.shape)