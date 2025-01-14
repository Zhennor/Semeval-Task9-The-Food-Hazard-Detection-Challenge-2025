import re
from bs4 import BeautifulSoup
import pandas as pd
from googletrans import Translator
from langdetect import detect
import asyncio
import nest_asyncio

# Initialize the translator
translator = Translator()

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator='\n')
    return text.strip()

def clean_text(text):
    text = re.sub(r'[^\w\s.,!?;]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.,!?;])', r'\1 ', text)
    return text.strip()

async def translate_to_english(text):
    try:
        # Detect language
        if detect(text) != 'en':  # If not English
            translated = await asyncio.to_thread(translator.translate, text, dest='en')
            return translated.text
        return text
    except Exception as e:
        print(f"Error translating: {e}")
        return text

async def process_dataframe(df):
    # Translate 'text' column
    df['text'] = await asyncio.gather(*(translate_to_english(text) for text in df['text']))

    # Translate 'title' column
    df['title'] = await asyncio.gather(*(translate_to_english(title) for title in df['title']))

    return df

data = pd.read_csv('data.csv')

data['text'] = data['text'].apply(extract_text_from_html)
data['text'] = data['text'].apply(clean_text)

async def main():
    nest_asyncio.apply()
    data = await process_dataframe(data)
    data.to_csv('final_data_cleaned.csv', index=False)
    print("Translation complete and file saved.")

asyncio.run(main())