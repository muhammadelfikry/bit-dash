from groq import Groq
import os

def summarize_descriptions(query ,descriptions: list[str]) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    joined_descriptions = "\n".join(descriptions) 

    prompt = (
        f"Here are some news descriptions about cryptocurrencies on the topic of {query}:\n\n"
        f"{joined_descriptions}\n\n"
        "Write a short summary of the whole story in 3-5 clear and informative sentences:"
    )

    system_prompt = (
        "You are a professional financial journalist who specializes in cryptocurrency and financial markets. "
        "You only analyze and summarize news content strictly related to cryptocurrencies (such as Bitcoin, Ethereum, etc.) or general financial topics (such as stocks, inflation, interest rates, etc.). "
        "You must reject or ignore any input that falls outside of these domains."
        "Any non-financial or non-cryptocurrency content should be politely rejected with a message indicating that you only handle financial and crypto-related topics."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user", 
                "content": prompt
            }, 
            {
                "role": "system",
                "content": system_prompt
            }
        ],
        temperature=0.3,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()
