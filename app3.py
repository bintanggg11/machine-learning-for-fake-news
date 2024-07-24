import streamlit as st
import pandas as pd
import joblib
import re
import webbrowser
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

def main_page():
    st.title('Selamat Datang di Aplikasi Deteksi Berita Politik Hoax')

    st.markdown("""
    <div style="text-align: justify;">
        <h4>Tentang Aplikasi Ini</h4>
        <p>
            Aplikasi ini dirancang untuk membantu Anda dalam mengidentifikasi apakah berita politik yang Anda baca adalah hoax atau bukan. Caranya sangat mudah, cukup masukkan teks berita ke dalam text area yang tersedia, dan aplikasi ini akan melakukan analisis untuk menentukan apakah berita politik tersebut hoax atau tidak.
        <p>    
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: justify;">
        Dalam kategori berita politik, penyebaran berita hoax sering kali dimanfaatkan untuk kepentingan tertentu, seperti mempengaruhi opini publik, menciptakan ketidakstabilan politik, dan menjatuhkan lawan politik. Berita hoax dalam bidang politik dapat memicu perpecahan sosial, mempengaruhi hasil pemilihan umum, dan merusak reputasi individu atau kelompok tertentu. Oleh karena itu saya memilih topik ini untuk membuat model machine learning yang dapat melakukan klasifikasi berita hoax yang tersebar di masyarakat sehingga masyarakat dapat membedakan antara berita politik hoax dan tidak hoax.
    </div>
    """, unsafe_allow_html=True)


def prediction_page():
    st.title('Deteksi Berita Politik Hoax')

    st.markdown("""
    <div style="text-align: justify;">
        <h4>Kriteria Berita untuk Prediksi</h4>
        <p>
            Berita yang dapat diprediksi dengan aplikasi ini harus memenuhi kriteria berikut:
            <ul>
                <li><b>Minimal 20 Kata:</b> Berita harus memiliki minimal 20 kata untuk memastikan bahwa teks yang dianalisis memiliki cukup informasi untuk dilakukan prediksi yang akurat.</li>
                <li><b>Kata-Kata Relevan:</b> Kata-kata dalam berita harus memiliki pola dan memberikan informasi yang relevan. Kata-kata yang tidak bermakna atau diulang-ulang tanpa memberikan konteks yang jelas tidak akan dianggap sebagai input yang valid.</li>
            </ul>
        </p>
    </div>
    """, unsafe_allow_html=True)

    news = st.text_area('Masukkan berita yang ingin diuji:', height=250)

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

def tech_explanation_page():
    st.title('Penjelasan Teknologi')

    st.markdown("""
    <div style="text-align: justify;">
        <h3>Teknologi yang Digunakan</h3>
        <h4>Random Forest</h4>
        <p>
            Random Forest adalah sebuah metode machine learning yang menggabungkan banyak pohon keputusan (decision trees) untuk meningkatkan akurasi dan mengurangi overfitting. Setiap pohon dalam hutan (forest) memberikan prediksi, dan Random Forest menggabungkan hasil dari semua pohon untuk membuat keputusan akhir. Metode ini dikenal karena kemampuannya untuk menangani data yang kompleks dan memberikan hasil yang stabil.
        </p>
        <h4>Term Frequency-Inverse Document Frequency (TF-IDF)</h4>
        <p>
            Term Frequency-Inverse Document Frequency (TF-IDF) adalah teknik vektorisasi teks yang digunakan untuk mengubah teks menjadi bentuk numerik yang dapat diproses oleh model machine learning. TF-IDF mengukur seberapa penting sebuah kata dalam dokumen berdasarkan frekuensi kata tersebut dalam dokumen (Term Frequency) dan seberapa umum kata tersebut di seluruh dokumen (Inverse Document Frequency). Teknik ini membantu model memahami konteks dan relevansi kata dalam teks.
        </p>
        <h3>Alur Program</h3>
        <p>
            Berikut adalah langkah-langkah yang dilakukan aplikasi ini untuk mengklasifikasikan berita:
            <ol>
                <li><b>Masukkan Teks Berita:</b> Anda memasukkan teks berita yang ingin diuji ke dalam text area.</li>
                <li><b>Cleaning Teks:</b> Aplikasi akan membersihkan teks dengan menghapus URL, tag HTML, karakter non-alfanumerik, dan angka. Selain itu, teks akan diproses untuk menghapus stopwords dan melakukan stemming menggunakan Sastrawi.</li>
                <li><b>Vektorisasi Teks:</b> Teks yang telah diproses akan diubah menjadi representasi numerik menggunakan teknik TF-IDF.</li>
                <li><b>Prediksi:</b> Representasi numerik dari teks berita akan diberikan kepada model Random Forest untuk klasifikasi.</li>
                <li><b>Hasil Klasifikasi:</b> Model akan memberikan prediksi apakah berita tersebut hoax atau tidak, bersama dengan probabilitas untuk setiap kelas.</li>
            </ol>
        </p>
    </div>
    """, unsafe_allow_html=True)

def about_page():
    st.title('Tentang Saya')

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
        <div style="text-align: justify;">
            <p>
                Halo, saya adalah Rizky Bintang Yudhistira, seorang pengembang aplikasi machine learning. Saya memiliki pengalaman dalam mengembangkan model machine learning dan aplikasi web untuk membantu masyarakat dalam mengenali berita politik palsu. Saya tertarik pada teknologi, data science, dan bagaimana kita dapat menggunakan teknologi untuk membuat dunia menjadi tempat yang lebih baik.
            </p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <a href="https://www.linkedin.com/in/rizky-bintang-yudhistira/" target="_blank">
            <button style="width: 100%; height: 40px; background-color: #0077b5; color: white; border: none; border-radius: 5px;">
                LinkedIn
            </button>
        </a>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <a href="https://www.instagram.com/rizky_yudhistiraaa/" target="_blank">
            <button style="width: 100%; height: 40px; background-color: #C13584; color: white; border: none; border-radius: 5px;">
                Instagram
            </button>
        </a>
        """, unsafe_allow_html=True)

# Menu Navigasi
pages = {
    "Halaman Utama": main_page,
    "Prediksi": prediction_page,
    "Penjelasan Teknologi": tech_explanation_page,
    "Tentang Saya": about_page
}

page = st.sidebar.selectbox("Pilih Halaman", list(pages.keys()))
pages[page]()
