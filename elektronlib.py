import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


num_users = 100
num_books = 1000

# Kitap adları: Bedii, Bilim Kurğusu, Finans kitapları (Azerbaycan dilinde)
bedii_books = ['Savaş və Sülh', 'Cinayət və Cəza', 'Anna Karenina', 'Böyük Ümidlər', 'Madam Bovari', 'Huzur', 'İntiqam', 'Körlük', 'Cəvdet bəy və Oğulları', 'Germinal']
sci_fi_books = ['Dune', '1984', 'Fahrenheit 451', 'Əzablı Yeni Dünyalar', 'Soldan Qaranlıq', 'Neuromancer', 'Marslı', 'Hiperyon', 'Qar Savaşları', 'Ender\'s Oyun']
finance_books = ['Zəngin Baba, Kasib Baba', 'Ağıllı İnvestor', 'Böyük Babalıq Səhmdarları', 'Həyat və İş Prinsipləri', 'Növbəti Qapalı Milyonçu', 'Pulunuz və Həyatınız', 'Ümumi Səhmlər və Nadir Mənfəət', 'Maliyyə Azadlığı', 'Pulun Psixologiyası', 'Kiçik Kitab Ağıllı İnvestisiya']

# Kitap listelerini birleştiriyoruz
all_books = bedii_books + sci_fi_books + finance_books

# Kullanıcı ve kitap ID'lerini oluşturuyoruz
user_ids = np.arange(1, num_users + 1)  # 100 farklı kullanıcı
book_ids = np.random.choice(all_books, size=100)  # 100 kitap, burada gerçek kitap isimleri kullanılıyor
status = np.random.choice([0, 1], size=100)  # 0: Oxunmamış, 1: Oxunmuş

# Kitap bilgilerini oluşturuyoruz
janr = np.random.choice(['Roman', 'Elm Kurğusu', 'Fantastik', 'Qorxu', 'Məhəbbət'], size=100)
ozet = np.random.choice(['Hüzünlü və əfsanəvi bir müharibə hekayəsi, Napoleon dövründə Rusiyanın həyatını təsvir edir.',
                         'Sevgi və xəyanət, evlilik və boşanma mövzusunda dərin bir araşdırma.',
                         'Ailəsini tərk edən və dünyadakı eşqi axtaran bir qadının kədərli hekayəsi.',
                         'Bir insanın intiqam istəyinin və buna qarşı yaşadığı daxili mübarizəni izah edən bir roman.',
                         ''], size=100)

# Veri setini oluşturuyoruz
df_books = pd.DataFrame({
    'User_ID': user_ids,
    'Kitab_Adi': book_ids,
    'Status': status,
    'Janr': janr,
    'haqqinda': ozet
})



# **Sıra** ve **Bölme** sütunlarını ekliyoruz
df_books['Sıra'] = df_books.groupby('Kitab_Adi').cumcount() + 1  # Her kitap için sıralama

# Bölme sütununu Janr'a göre numaralandırıyoruz
janr_to_bolme = {'Roman': 1, 'Elm Kurğusu': 2, 'Fantastik': 3, 'Qorxu': 4, 'Məhəbbət': 5}
df_books['Bölme'] = df_books['Janr'].map(janr_to_bolme)


# Kullanıcı-Kitap matrisi oluşturuluyor
user_book_matrix = df_books.pivot_table(index='User_ID', columns='Kitab_Adi', values='Status', fill_value=0)

# Cosine similarity ile kullanıcılar arasındaki benzerliği hesaplıyoruz
@st.cache_data  # Sadece bir kez hesaplanmasını sağlar
def compute_cosine_similarity():
    cosine_sim_user = cosine_similarity(user_book_matrix)
    return pd.DataFrame(cosine_sim_user, index=user_book_matrix.index, columns=user_book_matrix.index)

# Cosine similarity'yi hesapla ve veri çerçevesine dönüştür
cosine_sim_user_df = compute_cosine_similarity()

# Kullanıcıya göre kitap önerisi yapan fonksiyon
def recommend_books_by_user(user_id, user_book_matrix, cosine_sim_user_df, top_n=5):
    if user_id not in cosine_sim_user_df.index:
        return f"User ID {user_id} veri setində tapılmadı."
    
    # Kullanıcının okuduğu kitapları alıyoruz
    user_books = user_book_matrix.loc[user_id]

    # Kullanıcıya benzer kullanıcıları buluyoruz
    similar_users = cosine_sim_user_df[user_id].sort_values(ascending=False)[1:top_n+1].index

    # Benzer kullanıcıların okuduğu kitapları öneriyoruz
    recommended_books = []
    for user in similar_users:
        similar_user_books = user_book_matrix.loc[user]  # Benzer kullanıcının okuduğu kitaplar
        
        # Eğer kitap henüz kullanıcı tarafından okunmadıysa, öneriye ekle
        for book in similar_user_books[similar_user_books > 0].index:
            if book not in user_books[user_books > 0].index:
                recommended_books.append(book)

    # Eğer hiç kitap önerilmediyse, boş bir DataFrame döndür
    if len(recommended_books) == 0:
        return pd.DataFrame(columns=['Kitap Adı', 'Sıra', 'Bölme', 'Özet'])
    
    # Sadece farklı kitapları öneriyoruz
    recommended_books = list(set(recommended_books))

    # Kitap bilgilerini alıyoruz
    recommended_books_df = pd.DataFrame({
        'Kitap Adı': recommended_books
    })

    # Kitap bilgilerini df_books'tan alarak birleştiriyoruz
    recommended_books_df = pd.merge(recommended_books_df, df_books[['Kitab_Adi', 'Sıra', 'Bölme', 'haqqinda']], 
                                     left_on='Kitap Adı', right_on='Kitab_Adi', how='left').drop(columns=['Kitab_Adi'])

    return recommended_books_df

# Streamlit uygulaması başlatılıyor
st.title('Kitap Tövsiyə Sistemi')

# Kullanıcıdan ID almak
user_id_input = st.number_input('İstiadəçi İd-sini daxil edin:', min_value=1, max_value=num_users, value=1)

# **Session State** kullanarak önerileri saklıyoruz
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = {}

# Kullanıcı ID'si ile öneri al
if user_id_input not in st.session_state.recommendations:
    st.session_state.recommendations[user_id_input] = recommend_books_by_user(user_id_input, user_book_matrix, cosine_sim_user_df)

# Aynı ID için öneriler göster
recommendations = st.session_state.recommendations[user_id_input]

if isinstance(recommendations, pd.DataFrame) and recommendations.empty:
    st.write("Tövsiyə ediləcək kitab yoxdur.")
else:
    st.write("Tövsiyə olunan kitablar:")
    st.dataframe(recommendations)







st.sidebar.title("Maraqlandığınız kitabın adını daxil edin.")

# Seçim qutusu
a = st.sidebar.selectbox(label='Kitab adı', options=df_books['Kitab_Adi'].unique())

# Seçilən kitab haqqında məlumat
if a:
    kitab_info = df_books[df_books['Kitab_Adi'] == a]
    st.sidebar.header(f"**Seçilmiş kitab:** {a}")
    st.sidebar.header(f"**Sıra:** {kitab_info['Sıra'].values[0]}")
    st.sidebar.header(f"**Bölmə:** {kitab_info['Bölme'].values[0]}")





