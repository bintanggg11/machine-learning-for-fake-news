import streamlit as st
import pandas as pd
import joblib
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Memuat model dan vectorizer yang telah disimpan
model_rf = joblib.load('./Model/model_rf.joblib')
vct = joblib.load('./Model/tfidf_vectorizer.pkl')

def wardrop(text):
    text = text.lower()
    url_pattern = re.compile(r'\b(?:https?|ftp|file):\/\/[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|]|\bwww\.[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|]', re.IGNORECASE)
    text = re.sub(url_pattern, '', text)
    html_tag_pattern = re.compile(r'<[^>]+>')
    text = re.sub(html_tag_pattern, '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', ' ', text)
    extra_spaces_pattern = re.compile(r'\s+')
    text = re.sub(extra_spaces_pattern, ' ', text)
    return text

def sastrawi(text):
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    text = stopword_remover.remove(text)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text)
    return text

def manual_testing(news):
    testing_news = {'text': [news]}
    new_df_test = pd.DataFrame(testing_news)
    new_df_test['text'] = new_df_test['text'].apply(wardrop)
    new_df_test['text'] = new_df_test['text'].apply(sastrawi)
    new_x_test = new_df_test['text']
    new_xv_test = vct.transform(new_x_test)
    pred = model_rf.predict(new_xv_test)
    if pred[0] == 0:
        pred = 'Berita tidak hoax'
        pred_icon = "✅"
    elif pred[0] == 1:
        pred = 'Berita hoax'
        pred_icon = "❌"
    return pred, pred_icon

st.title('Deteksi Berita Hoax')
news = st.text_area('Masukkan berita yang ingin diuji:')

if st.button('Deteksi'):
    hasil, icon = manual_testing(news)
    st.markdown(f"<h3>{icon} <b>{hasil}</b></h3>", unsafe_allow_html=True)
