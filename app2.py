import streamlit as st
import pandas as pd
import joblib
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Memuat model dan vectorizer yang telah disimpan
model_rf = joblib.load('model_rf.joblib')
vct = joblib.load('tfidf_vectorizer.pkl')

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
    probabilities = model_rf.predict_proba(new_xv_test)
    if pred[0] == 0:
        pred_label = 'Berita tidak hoax'
        pred_icon = "✅"
    elif pred[0] == 1:
        pred_label = 'Berita hoax'
        pred_icon = "❌"
    return pred_label, pred_icon, probabilities[0]

# Menambahkan foto dan deskripsi diri
st.title('Deteksi Berita Politik Hoax')

col1, col2 = st.columns([2, 3])

with col1:
    st.image("Foto1.jpg", use_column_width=True)  
    st.markdown("""
    <style>
    img {
        height: 280px !important; 
    }
    </style>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: justify">
        Dalam kategori berita politik, penyebaran berita hoax sering kali dimanfaatkan untuk kepentingan tertentu, seperti mempengaruhi opini publik, menciptakan ketidakstabilan politik, dan menjatuhkan lawan politik. Berita hoax dalam bidang politik dapat memicu perpecahan sosial, mempengaruhi hasil pemilihan umum, dan merusak reputasi individu atau kelompok tertentu. Oleh karena itu saya memilih topik ini untuk membuat model machine learning yang dapat melakukan klasifikasi berita hoax yang tersebar di masyarakat sehingga masyarakat dapat membedakan antara berita politik hoax dan tidak hoax.
    </div>
    """, unsafe_allow_html=True)

news = st.text_area('Masukkan berita yang ingin diuji:')

# Custom CSS to change button color to red
button_style = """
    <style>
    .stButton button {
        background-color: red !important;
    }
    </style>
    """
st.markdown(button_style, unsafe_allow_html=True)

if st.button('Deteksi'):
    hasil, icon, prob = manual_testing(news)
    st.markdown(f"<h3>{icon} <b>{hasil}</b></h3>", unsafe_allow_html=True)
    
    st.write(f"Probabilitas Tidak Hoax: {prob[0]:.2f}")
    st.progress(prob[0])
    
    st.write(f"Probabilitas Hoax: {prob[1]:.2f}")
    st.progress(prob[1])
