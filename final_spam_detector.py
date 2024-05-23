import streamlit as st
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
def dataPreprocessing(text):
    # lower case
    text = text.lower() 

    # word tokenise
    y=[]
    text= word_tokenize(text)
    
    for i in text:
        if i.isalnum(): 
         y.append(i)
    
    text = y
    y = []

    # remove stop words and punctuation
    for i in text:
       if i not in  stopwords.words('english') and i not in string.punctuation:
          y.append(i)
    text = y  
    y = []    

    # apply stemming
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return ' '.join(y)
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS spam classifier")

input_sms = st.text_area("Enter the message")
if st.button('predict'):
    # 1. preprocess
    transform_sms=dataPreprocessing(input_sms)
    # 2. vectorisation
    vector_input = tfidf.transform([transform_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result ==1:
        st.header("spam")
    else:  
        st.header("Not spam")