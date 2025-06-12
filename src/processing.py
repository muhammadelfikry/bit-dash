from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from data_loader import load_data
import re
import string

def dataCleaning(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')
    return text

def caseFolding(text):
    text = text.lower()
    return text

def tokenizing(text):
    text = word_tokenize(text)
    return text

def filteringText(text):
    listStopwords = set(stopwords.words('english'))
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text

def toSentences(list_words):
    sentence = ' '.join(word for word in list_words)
    return sentence

if __name__ == "__main__":
    try:
        df = load_data("./data/raw/news_data.csv")
        
        # Data Cleaning
        df = dataCleaning(df)

        # Text Processing
        df["cleaned_text"] = df["description"].apply(cleaningText)
        df["folded_text"] = df["cleaned_text"].apply(caseFolding)
        df["tokenized_text"] = df["folded_text"].apply(tokenizing)
        df["filtered_text"] = df["tokenized_text"].apply(filteringText)
        df["final_text"] = df["filtered_text"].apply(toSentences)

        print("Data cleaning and text processing completed")
        print("DataFrame shape: ", df.shape)

        # Select relevant columns and save to CSV
        cleaned_df = df[["title", "final_text"]].copy()
        cleaned_df.columns = ["title", "description"]

        print("Relevant columns selected and renamed")
        print("DataFrame columns: ", cleaned_df.columns)

        cleaned_df.to_csv("./data/processed/cleaned_news_data.csv", index=False)
        print("Successfully saved cleaned data to './data/processed/cleaned_news_data.csv'")

    except Exception as e:
        print(f"An error occurred: {e}")