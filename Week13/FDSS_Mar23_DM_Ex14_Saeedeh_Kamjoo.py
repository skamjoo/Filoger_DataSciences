
# Exercies 14: Making web application using Strreamlit for Predict Persian text and URL 

## Import libraries
import streamlit as st
import pickle
import os
import emoji
from bs4 import BeautifulSoup
from urllib.request import urlopen
from hazm import word_tokenize

## Load the LabelEncoder, TF-IDF Vectorizer, and SVM classifier from pickle files

current_dir = os.path.dirname(os.path.realpath(__file__))
pk_le = os.path.join(current_dir, 'models', 'lion_le.jdsh')
with open(pk_le, 'rb') as le_file:
    labelencoder = pickle.load(le_file)

pk_vec = os.path.join(current_dir, 'models', 'lion_v.jdsh')
with open(pk_vec, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

pk_svc = os.path.join(current_dir, 'models', 'lion_svc.jdsh')
with open(pk_svc, 'rb') as svm_file:
    svm_classifier = pickle.load(svm_file)


model = os.path.join(current_dir, 'models', 'stopwords.txt')
with open(model, encoding='utf8') as stopwords_file:
    stopword = stopwords_file.readlines()
nltk_stopwords = [str(line).replace('\n', '') for line in stopword]


## Define some Functions
def predict_class(news):
    title_body_tokenized = word_tokenize(news)
    title_body_tokenized_filtered_stemming = [w for w in title_body_tokenized if not w in nltk_stopwords]
   
    x = [' '.join(title_body_tokenized_filtered_stemming)]
    x_v = vectorizer.transform(x)
    p = svm_classifier.predict(x_v)
    label = labelencoder.inverse_transform(p)
    return label[0]

## Streamlit schema
#  
@st.cache_resource
def get_text(raw_url):
	page = urlopen(raw_url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text

def main():
	st.title("دسته بندی اخبار از متن و سایت های فارسی زبان")

	activities = ["Classify on text","Classify on URL","About"]
	choice = st.sidebar.selectbox("فهرست", activities)

	if choice == 'Classify on text':
		st.subheader("دسته بندی موضوعات اخبار از متن")
		st.write(emoji.emojize('ممنون از انتخاب شما :red_heart:'))
		raw_text = st.text_area("متن فارسی خود را وارد کنید","متن نمونه")
		if st.button("اجرا"):
			if raw_text != "":
				category = predict_class(raw_text)
				st.success(f"Predicted Category: {category}")
			else:
				st.warning("Please enter a news article.")
			
	if choice == 'Classify on URL':
		st.subheader(" دسته بندی موضوعات اخبار از وبسایت")
		raw_url = st.text_input("آدرس را وارد کنید","https://google.com")
		if st.button("طبقه بندی"):
			if raw_url != "":
				result = get_text(raw_url)
				category = predict_class(result)
				st.success(f"Predicted Category: {category}")
			else:
				st.warning("Please enter a news article.")

	if choice == 'About':
		st.subheader("سرگرمی های یک دانشجوی ریاضی در دنیای برنامه نویسی")
		st.info("ریاضی زلف پریشان عالم است. اگر علاقمند به نوشتن مقالات ریاضی هستید میتونید از پیج ما دیدن کنید و در صورت علاقه مقاله خود را سابمیت کنید.")
		st.text("If you want to submmit your manuscript in mathematical journal, please visit our site:")
		st.text("cmde.tabrizu.ac.ir")

if __name__ == '__main__':
	main()
