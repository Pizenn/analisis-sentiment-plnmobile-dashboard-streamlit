import pandas as pd
import streamlit as st
import plotly
from collections import Counter
import plotly.express as px
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud

st.set_page_config(
    page_title="Dashboard Sentiment",  
    page_icon="🔥",  
    layout="wide",  # Pilihan: "centered" atau "wide"
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .stApp{
            background-color: #000000;  /* Warna latar belakang */
            color: white;  /* Warna teks */
        }
        h1, h2, h3 {
            margin-bottom: 5px !important;  /* Kurangi jarak bawah judul */
        }
        .bubble-container {
            display: flex;
            flex-direction: column;
            gap: 15px; /* Menambah jarak antar bubble */
            margin-top: 0px !important;
        }
        .bubble {
            max-width: 75%;
            padding: 15px 20px;
            border-radius: 10px;
            font-size: 16px;
            line-height: 1.5;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.15);
            color: white;
            word-wrap: break-word;
            margin-bottom: 8px; /* Menambah jarak antar bubble */
            position: relative;
        }
        .positif {
            background: #262730;  /* Gradasi hijau */
            align-self: flex-start;
        }
        .negatif {
            background: #262730;  /* Gradasi merah */
            align-self: flex-end;
        }
        .netral {
            background: #262730;  /* Gradasi biru */
            align-self: center;
        }
        div.stButton > button {
            background-color: #121212; /* Warna biru */
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
            border: none;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            color: white;
            background-color: #FF4B4B; /* Warna saat hover */
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Daftar stopwords (kata-kata umum yang akan dihapus)
STOPWORDS = set([
    "kamu", "aku", "saya", "dia", "kita", "mereka", "ini", "itu", "yang", 
    "dan", "atau", "dengan", "dari", "ke", "untuk", "dalam", "pada", 
    "aplikasi", "pln", "mobile", "review", "user", "pengguna", "sangat", "bisa",
    "di", "semakin","listrik", "mau", "nya", "ada","di","gak","sudah",
    "tapi","yg","ya","lagi","aja","juga","dari","dengan","untuk","kalo",
    "ke","karena","kalau","masih","kali","saja","udah","banget","bgt",
    "lebih","ga","tidak","terus","kenapa","apk","sering",
    "token","aplikasinya","beli", "buat","tolong","baru","daya","mohon",
    "belom","mohon","jadi", "tambah", "terimakasih", "kasih", "kok","kasih",
    "harus","saat","pakai","cek"   
])

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.lower()  # Ubah ke huruf kecil
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = re.sub(r'\d+', '', text)  # Hapus angka
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]  # Hapus kata umum
    return " ".join(words)

# Fungsi untuk membuat bar chart kata paling sering muncul
def create_word_chart(df, sentiment, top_n=10):
    filtered_df = df[df["sentiment"] == sentiment]  # Filter berdasarkan sentimen
    words = " ".join(filtered_df["content"].astype(str).apply(clean_text)).split()  # Gabungkan semua teks
    word_counts = Counter(words)  # Hitung frekuensi kata
    common_words = word_counts.most_common(top_n)  # Ambil kata terbanyak

    if not common_words:
        st.warning("Tidak ada cukup data untuk ditampilkan.")
        return None

    word_df = pd.DataFrame(common_words, columns=["word", "count"])
    word_df = word_df.sort_values(by="count", ascending=True)  # Kata terbanyak di atas

    # Pilih warna berdasarkan sentimen
    color_map = {
        "positif": "#2ECC71",  # Hijau
        "negatif": "#E74C3C",  # Merah
        "netral": "#3498DB"  # Biru
    }
    bar_color = color_map.get(sentiment.lower(), "#95A5A6")  # Default abu-abu

    fig = px.bar(
        word_df, 
        y="word", 
        x="count", 
        orientation="h",  # Horizontal bar chart
        title=f"Kata Paling Sering Muncul ({sentiment.capitalize()})",
        labels={"count": "Frekuensi", "word": ""},
        text_auto=True,
        width=900,
        height=800,
        color_discrete_sequence=[bar_color]
    )
    fig.update_layout(
        plot_bgcolor="#121212",
        paper_bgcolor="#121212",
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=40, b=40),
        )
    return fig

# **Fungsi untuk Menampilkan Review Secara Acak dalam Bubble Chat**
def show_random_reviews(df, sentiment):
    filtered_df = df[df["sentiment"] == sentiment]
    if filtered_df.empty:
        st.warning("Tidak ada review untuk sentimen ini.")
        return

    st.markdown('<div class="bubble-container">', unsafe_allow_html=True)
    
    random_reviews = filtered_df.sample(min(7, len(filtered_df)))

    for _, row in random_reviews.iterrows():
        bubble_class = "positif" if sentiment == "Positif" else "negatif" if sentiment == "Negatif" else "netral"
        st.markdown(f'<div class="bubble {bubble_class}">{row["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_final.csv")
    return df

df = load_data()

# Fungsi untuk membersihkan teks dan menghapus stopwords
def clean_text(text):
    text = text.lower()  # Ubah ke huruf kecil
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = re.sub(r'\d+', '', text)  # Hapus angka
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]  # Hapus kata umum
    return " ".join(words)

# Fungsi untuk mendapatkan kata terbanyak berdasarkan sentimen
def get_top_words(df, sentiment, top_n=10):
    filtered_df = df[df['sentiment'] == sentiment]  # Filter berdasarkan sentimen
    all_text = ' '.join(filtered_df['content'].astype(str).apply(clean_text))  # Gabungkan semua teks
    word_counts = Counter(all_text.split())  # Hitung frekuensi kata
    return word_counts.most_common(top_n)  # Ambil n kata terbanyak

# Pastikan kolom 'sentiment' dan 'content' ada
if "sentiment" not in df.columns or "content" not in df.columns:
    st.error("Dataset harus memiliki kolom 'sentiment' dan 'content'")
    st.stop()

# Hitung jumlah setiap sentimen
sentiment_counts = df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["sentiment", "count"]

def get_bar_chart():
    fig = px.bar(
        sentiment_counts,
        x="sentiment",
        y="count",
        color="sentiment",
        text="count",
        labels={"sentiment": "Sentimen", "count": "Jumlah"},
        hover_data={"sentiment": True, "count": True},
        color_discrete_map={
            "Positif": "#2ECC71",  # Hijau
            "Negatif": "#E74C3C",  # Merah
            "Netral": "#3498DB"  # Biru
        }
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        plot_bgcolor="#121212",   # Background chart (hitam)
        paper_bgcolor="#121212",  # Background luar chart
        font=dict(color="white"), # Warna teks
        xaxis=dict(showgrid=False),  # Hilangkan grid sumbu X
        yaxis=dict(gridcolor="gray"), # Warna grid sumbu Y
        margin=dict(l=40, r=40, t=40, b=40),  # Margin agar tidak terlalu padat
        bargap=0.2,  # Jarak antar bar
    )
    return fig

def get_donut_chart():
    fig = px.pie(
        sentiment_counts,
        names="sentiment",
        values="count",
        hole=0.4,
        color="sentiment",  # Sesuaikan warna berdasarkan kategori
        color_discrete_map={
            "Positif": "#2ECC71",  # Hijau
            "Negatif": "#E74C3C",  # Merah
            "Netral": "#3498DB"  # Biru
        }
    )
# **Styling Donut Chart**
    fig.update_traces(
        textinfo="percent+label",  # Menampilkan persen & label
        marker=dict(line=dict(color="#000000", width=2))  # Border antar bagian
    )
    fig.update_layout(
        plot_bgcolor="#121212",
        paper_bgcolor="#121212",
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=40, b=40),
        )

    return fig

# Inisialisasi state untuk menyimpan sentimen yang dipilih
if 'selected_sentiment' not in st.session_state:
    st.session_state.selected_sentiment = 'positif'

st.title("Dashboard Analisis Sentimen Pengguna Aplikasi PLN Mobile")
st.subheader("Chart Perbandingan Sentimen")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(get_bar_chart(), use_container_width=True)
with col2:
    st.plotly_chart(get_donut_chart(), use_container_width=True)

# UI Streamlit
st.subheader("Kata Terbanyak Berdasarkan Sentimen")

# Ambil sentimen unik dari dataset
sentiments = df["sentiment"].unique()

# **Pastikan state diinisialisasi sebelum digunakan**
if "selected_sentiment" not in st.session_state:
    st.session_state.selected_sentiment = sentiments[0]  # Default ke sentimen pertama

# **Sekarang kita bisa ambil nilai selected_sentiment**
selected_sentiment = st.session_state.selected_sentiment

# Buat tombol untuk setiap sentimen
cols = st.columns(len(sentiments))

for i, sentiment in enumerate(sentiments):
    with cols[i]:  # Letakkan tombol dalam kolom agar sejajar
        if st.button(sentiment, use_container_width=True):
            st.session_state.selected_sentiment = sentiment
            selected_sentiment = sentiment  # Perbarui variabel lokal

# Buat dan tampilkan bar chart berdasarkan sentimen yang dipilih
chart = create_word_chart(df, selected_sentiment, top_n=10)

if chart:
    st.plotly_chart(chart)

# Tampilkan review acak
st.subheader(f"Review dengan Sentimen: {selected_sentiment.capitalize()}")
show_random_reviews(df, selected_sentiment)