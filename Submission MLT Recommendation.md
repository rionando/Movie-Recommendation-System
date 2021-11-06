# Laporan Proyek Machine Learning - Muhammad Rionando D

## Project Overview
Pada Proyek Machine Learning Terapan 2 Sistem Rekomendasi kali ini saya memilih untuk membuat sebuah sistem rekomendasi film dengan content based filtering menggunakan TF-IDF Vectorizer dan dataset film dari The Movie Database. The Movie Database (TMDb) adalah database film yang menyediakan data-data lengkap seperti data film yang akan datang, data serial tv (TvSeries), dll. Database film dan TV yang dibuat oleh komunitas TMDb. database ini sudah ada sejak tahun 2008. data set ini saya dapat pada situs kaggle.

Proyek ini penting untuk diselesaikan melihat dari manfaatnya yaitu memberikan rekomendasi judul film berdasarkan kontennya jadi ketika kita sedang bingung mau menonton film atau series apa, project ini dapat membantu memberikan kita rekomendasi film yang relevan berdasarkan film yang kita sukai.

- Berikut ini merupakan referensi yang saya gunakan mengenai sistem rekomendasi content based filtering yang saya dapatkan dari google scholar [Referensi jurnal](https://ieeexplore.ieee.org/abstract/document/9489125/)

## Business Understanding
Pada proyek ini saya ingin membuat sebuah sistem rekomendasi film yang dapat memberikan rekomendasi film yang familiar atau mirip dengan film yang kita sukai, jadi ketika kita bingung dan tidak tau mau menonton film atau series apa, sistem ini akan membantu.

### Problem Statements
1. Bagaimana cara membuat sebuah model machine learning yang berguna untuk merekomendasikan film dengan metode content based filtering?
2. Bagaimana hasil evaluasi dari metode content based filtering pada sistem rekomendasi film?

### Goals
1. Membuat sebuah model machine learning yang berguna untuk merekomendasikan film dengan metode content based filtering
2. Mengevaluasi hasil dari metode content based filtering menggunakan metrik evaluasi

### Solution approach
Pada proyek kali ini saya akan menggunakan metode content based filtering untuk menyelesaikan permasalahan yang ada. 
- **Content Based Filtering**. 
Content-based filtering mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.

## Data Understanding
![alternate text](https://github.com/rionando/MLT-2/blob/main/T%206.jpg)
Dataset yang saya gunakan pada kasus ini bersumber dari kaggle [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata/code?datasetId=138&sortBy=voteCount)
dan memiliki dimensi 4803 X 24

### Karakteristik data
![alternate text](https://github.com/rionando/MLT-2/blob/main/R%2011.jpg)

Berdasarkan gambar diatas dapat dilihat bahwa dataset terdiri dari 2 yang total yang memiliki dimensi 4803 X 24

![alternate text](https://github.com/rionando/MLT-2/blob/main/R%2012.jpg)

Gambar diatas merupkan deskripsi lengkap dari masing-masing dataset yang digunakan

Untuk variabel-variabelnya antara lain:
Dataset Movies:
- `budget` - Biaya pembuatan.
- `genre` - Genre/Aliran Film.
- `homepage` - Link web film.
- `id` - nomor id.
- `keywords` - kata kunci film.
- `original_language` - Bahasa asli.
- `original_title` - Judul asli.
- `overview` - Deskripsi singkat.
- `popularity` - Angka Popularitas.
- `production_companies` - Perusahaan yang memproduksi.
- `production_countries` - Negara tempat produksi.
- `release_date` - Tangga rilis.
- `revenue` - Pendapatan.
- `runtime` - Durasi.
- `spoken_languange` - Bahasa yang diucapkan
- `status` - Status.
- `tagline` - tagline.
- `title` - Judul film.
- `vote_average` - rata-rata nilai.
- `vote_count` - jumlah penilai.

Dataset Credits:
- `movie_id` - id unik.
- `title` - Judul film
- `cast` - Aktor utama.
- `crew` - Sutradara dkk.

## Data Preparation
### Menggabungkan Dataset dan Membuang variabel yang tidak digunakan
Karena pada dataset yang saya gunakan terbagi menjadi 2 dataset yaitu Movies dan Credit maka untuk memudahkan proses selanjutnya saya akan menggabungkan kedua data set tersebut
![alternate text](https://github.com/rionando/MLT-2/blob/main/R%201.jpg)

Selanjutnya saya juga akan membuang variabel yang tidak digunakan
![alternate text](https://github.com/rionando/MLT-2/blob/main/R%202.jpg)

### EDA
#### Visualisasi Film berdasarkan Genrenya
![alternate text](https://github.com/rionando/MLT-2/blob/main/R%203.jpg)

Berdasarkan gambar diatas dapat dilihat 3 genre paling banyak adalah Drama, Comedy, dan Thriller.

#### Visualisasi Film berdasarkan Aktornya
![alternate text](https://github.com/rionando/MLT-2/blob/main/R%204.jpg)

Berdasarkan gambar diatas dapat dilihat 3 top aktor adalah Samuel Jacson, Jr, dan RobertDeNiro.

#### Visualisasi Film berdasarkan Sutradaranya
![alternate text](https://github.com/rionando/MLT-2/blob/main/R%205.jpg)

Berdasarkan gambar diatas dapat dilihat 3 top sutradara adalah Steven Spielberg, Woody Allen, dan MartinScorsese.

### Melakukan standarisasi data
Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik. saya akan menggunakan teknik StandarScaler dari library Scikitlearn, StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1. Berikut merupakan codenya.
![alternate text](https://github.com/rionando/MLT-2/blob/main/R%206.jpg)

## Modeling
TF-IDF Vectorizer

Pada tahap ini saya akan membangun sistem rekomendasi sederhana berdasarkan keyword atau kata kunci film menggunakan TF-IDF Vectorizer. Alasan kenapa menggunakan keyword adalah karena keyword atau kata kunci menjelaskan lebih detail mengenai filmnya jika dibandingkan dengan genre yang lebih umum. Teknik tersebut akan saya digunakan untuk menemukan representasi fitur penting dari setiap kategori film. Saya menggunakan fungsi tfidfvectorizer() dari library sklearn dan berikut adalah outputnya.
![alternate text](https://github.com/rionando/MLT-2/blob/main/T%201.jpg)

Selanjutnya, lakukan fit dan transformasi ke dalam bentuk matriks. 
![alternate text](https://github.com/rionando/MLT-2/blob/main/T%202.jpg)

Matriks berukuran (4803, 7168). 

Untuk menghasilkan vektor tf-idf dalam bentuk matriks, menggunakan fungsi todense(). Jalankan kode berikut.
![alternate text](https://github.com/rionando/MLT-2/blob/main/R%2015.jpg)

Selanjutnya adalah menghitung derajat kesamaan (similarity degree) antar film dengan teknik cosine similarity. Di sini menggunakan fungsi cosine_similarity dari library sklearn. dengan rumusnya sebagai berikut
![alternate text](https://github.com/rionando/MLT-2/blob/main/Cosine-similarity-formula.png)

Dan ini untuk output dari cosine_similarity 

![alternate text](https://github.com/rionando/MLT-2/blob/main/R%2016.jpg)

Selanjutnya adalah Mendapatkan Rekomendasi
Dengan membuat fungsi film_recommendations dengan code sebagai berikut
![alternate text](https://github.com/rionando/MLT-2/blob/main/T%203.jpg)

Dengan menggunakan argpartition, saya mengambil sejumlah nilai k tertinggi dari similarity data (dalam kasus ini: dataframe cosine_sim_df). Kemudian, mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah. Data ini dimasukkan ke dalam variabel closest. Berikutnya, perlu menghapus original_title yang yang dicari agar tidak muncul dalam daftar rekomendasi.

Selanjutnya saya akan mengecek akurasi dari sistem rekomendasi dengan menemukan rekomendasi film yang familiar dengan film Saving Private Ryan, berikut adalah detail informasi film Saving Private Ryan.
![alternate text](https://github.com/rionando/MLT-2/blob/main/T%204.jpg)

Dilihat dari detail tersebut filmSaving Private Ryan memliki keyword berupa War, Self Secrefice, Veteran yang intinya film bercerita mengenai perang tentunya saya berharap mendapatkan rekomendasi film yang familiar bercerita tentang perang.

Berikut merupakan 5 film rekomendasi yang diberikan model.

![alternate text](https://github.com/rionando/MLT-2/blob/main/T%205.jpg)

Dari hasil tersebut 5 dari 5 film memiliki keyword yang relevan dengan film Saving Privat Ryan yaitu War.

## Evaluation

Karena pada kasus ini saya hanya menggunakan 1 model dan model yang digunakan adalah content based filtering menggunakan TF-IDF Vectorizer maka metrics evaluasi yang akan saya gunakan adalah precision dan dikarenakan kita tidak bisa menghitung dengan memanggil library scikit learn karena tidak ada data target/label seperti pada supervised learning. maka saya akan menghitung metrics evaluasinya secara manual dengan rumus sebagai berikut.

![alternate text](https://github.com/rionando/MLT-2/blob/main/dos_819311f78d87da1e0fd8660171fa58e620211012160253.png)

dari hasil rekomendasi pada model maka ada 5 dari 5 film yang memiliki kata kunci cerita yang similiar dengan Saving Private Ryan maka metrik evaluasinya adalah 5/5= 1

**Precision = 1**

Kelebihan dari metriks ini adalah :
- Metriks ini merupakan metrik yang paling cocok dengan model content based filtering menggunakan TF-IDF Vectorizer karena menghitung secara langsung precisison dari similaritas yang ada pada hasil rekomendasi dibanding dengan referensinya.

Sedangkan kekurangannya :
- Metriks ini tidak dapat dipanggil secara otomatis menggunakan sklearn karena tidak ada data target/label seperti pada supervised learning

> **Ini adalah bagian akhir laporan**
