# Text Classification for Customer Questions

Proyek ini menggunakan **Logistic Regression** untuk mengklasifikasikan pertanyaan pelanggan menjadi beberapa kategori seperti `Information`, `Problem`, `Request`, dan `Lainnya`. Proses ini mencakup *preprocessing*, ekstraksi fitur dengan **TF-IDF (unigram & bigram)**, dan evaluasi akurasi model.

## 📁 Struktur Folder

├ dataset/ <br>
└── labeled_question_list.csv  <br>
├── main.py <br>
└── README.md

Make sure file `labeled_question_list.csv` ada di dalam folder `dataset/`.

---

Lebeling
---
untuk lebeling dataset bisa menggunakan code repository ini <br>
https://github.com/Alhuzsyam/lebeling_data.git
<hr>
## 🧰 Requirements

Instalasi library yang dibutuhkan:

```bash
pip install -r requirements.txt
```


Jalankan program:

```bash
python main.py
```

