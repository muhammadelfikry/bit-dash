from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

def text_labeling(texts) -> list:
    try:
        if not isinstance(texts, list):
            raise TypeError("Input must be a list of news descriptions")

        if len(texts) != 10:
            raise ValueError("The number of news descriptions must be precise 10")
        
        texts_str = str(texts)

        count_from_string = texts_str.count("',")
        if count_from_string + 1 != len(texts):
            raise ValueError("Mismatch number of descriptions before and after conversion to string")

        content = """
        Tugas Anda adalah menganalisis daftar deskripsi berita yang diberikan dan menentukan sentimen utama dari masing-masing berita tersebut. Sentimen yang mungkin adalah:

        - 'positif' jika berita tersebut memberikan informasi yang baik atau menguntungkan bagi pasar cryptocurrency.
        - 'negatif' jika berita tersebut memberikan informasi yang buruk atau merugikan bagi pasar cryptocurrency.
        - 'netral' jika berita tersebut tidak memberikan dampak signifikan terhadap pasar cryptocurrency atau tidak memiliki sentimen yang jelas (misalnya bersifat informatif tanpa penilaian).

        Berikan output dalam bentuk string, bukan list Python. Format output harus berupa label sentimen yang dipisahkan koma dan masing-masing dibungkus dengan tanda kutip satu. Contoh format output yang benar:

        'positif', 'negatif', 'netral'

        Deskripsi berita:
        ["Bitcoin naik 8% setelah adanya sinyal positif dari Komisi Sekuritas AS tentang ETF.",
        "Bitcoin mengalami penurunan harga setelah adanya kebijakan baru AS yang ketat terhadap cryptocurrency.",
        "Market cryptocurrency sempat mengalami penurunan harga sebesar 3% hari ini akibat dari kebijakan US dan kembali bounce back ke harga semula"]

        Output:
        'positif', 'negatif', 'netral'
        """

        client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": content,
                },
                {
                    "role": "user",
                    "content": texts_str,
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        response = chat_completion.choices[0].message.content
        labels = [item.strip().strip("'\"") for item in response.split(",")]

        # Validasi hasil
        if not isinstance(labels, list):
            raise TypeError("The parsing result is not a list")

        return labels
    
    except Exception as e:
        print(f"error: {e}")