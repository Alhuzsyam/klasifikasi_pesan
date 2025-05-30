import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import nltk
import string
import os

# stopwords Indo
nltk.download('stopwords')
from nltk.corpus import stopwords

# Rapikan teks
def preprocess(text):
    text = str(text).lower()  # ke huruf kecil
    text = ''.join([c for c in text if c not in string.punctuation])  # hapus tanda baca
    tokens = text.split()  # tokenisasi
    tokens = [t for t in tokens if t not in stopwords.words('indonesian')]  # hapus stopwords
    return ' '.join(tokens)  # gabung kembali

# Ambil dataset dalam folder
DATASET_PATH = 'dataset/labeled_question_list.csv'

# Cek file ada atau tidak
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"File tidak ditemukan: {DATASET_PATH}")

# ambil dataset
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()  # remove spasi

# Pengecekan kolom
if 'pertanyaan' not in df.columns or 'label' not in df.columns:
    raise KeyError("format dataset harus punya kolom 'pertanyaan' dan 'label'")

# Lihat distribusi label (bisa jadi indikasi imbalance)
print("Distribusi label:")
print(df['label'].value_counts())

# Preprocessing teks
df['text_clean'] = df['pertanyaan'].apply(preprocess)

# TF-IDF dengan ngram 1-2
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df['text_clean'])
y = df['label']

# Split data train-test 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)  # stratify supaya proporsi label tetap sama

# Buat model Logistic Regression (lebih stabil dari Naive Bayes untuk imbalance)
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

# Prediksi test set
y_pred = model.predict(X_test)

# Evaluasi model
print("\nAkurasi:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:\n")
print(classification_report(y_test, y_pred, zero_division=0))
