import streamlit as st
import pickle
import pandas as pd
import numpy as np
import string
import nltk

nltk.download('punkt')
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_word_list = stopwords.words('english')

from nltk.stem.porter import PorterStemmer

ps_stem = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('SMS spam classifier')

input_text = st.text_area('please type ur SMS text')


def preprocess_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # here text will be in list after conversion

    final_text = []
    for word in text:
        if (word not in stop_word_list) and (word.isalnum()) and (word not in string.punctuation):
            final_text.append(ps_stem.stem(word))

    return ' '.join(final_text)


if st.button('predict'):

    converted_text = preprocess_text(input_text)

    vector_text = tfidf.transform([converted_text])

    res_class = model.predict(vector_text)[0]

    if res_class:
        st.header('Spam')
    else:
        st.header('Not Spam')

