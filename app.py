import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

import joblib
import altair as alt

pipe_lr = joblib.load(open("models/emotion_classifier.pkl",'rb'))


import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

# creating function from sql
def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS PredTable(message TEXT,prediction TEXT,probability NUMBER,sentiment TEXT)')

def add_data(message,prediction,probability,sentiment):
    c.execute('INSERT INTO PredTable(message,prediction,probability,sentiment) VALUES (?,?,?,?)',(message,prediction,probability,sentiment))
    conn.commit()

def view_data():
    c.execute("SELECT * FROM PredTable")
    data = c.fetchall()
    return data

# pipe_lr = joblib.load(open("models/emotion_classifier.pkl",'rb'))

def predict_emotions(docx):
    result = pipe_lr.predict([docx])
    return result

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

from textblob import TextBlob
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        result = "Positive"
    elif sentiment < 0:
        result = 'Negative'
    else:
        result = 'Neutral'
    return result
# emotions_emoji_dict = {"anger" : "\N{angry face","disgust":"","fear": "","happy":"\U0001f600","joy":"\U0001F602","neutral":"\U0001F642","surprise":"",'sadness':"",'shame':""}

def main():
    menu = ['Home','Monitor']
    choice = st.sidebar.selectbox("Menu",menu)
    create_table()
    
    str = " "
    if choice == "Home":
        st.subheader("Find Emotion In Iext")

        with st.form(key='emotion_classifier_form'):
            raw_text = st.text_area("Type Here")
    
            submit_text = st.form_submit_button(label = 'Submit')

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        if submit_text:
            if raw_text == "":
                st.info("Please enter text to find emotion")
            else:
                col1,col2 = st.columns(2)
                
                with col1:
                    st.write("Original Text")
                    st.success(raw_text)
                    st.write("Prediction")
                    st.success(prediction[0])
                    st.success("confidence:{}".format(np.max(probability)))
                    max_prob = np.max(probability)
                    sentiment = get_sentiment(raw_text)
                    add_data(raw_text,prediction[0],max_prob,sentiment)
                    # st.success("Data submitted")

                with col2:
                    st.write("Prediction Probability")
                    # st.write(probability)
                    prob_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                    # st.write(prob_df.T)
                    prob_df_clean = prob_df.T.reset_index()
                    prob_df_clean.columns = ['emotions','probability']
                    fig = alt.Chart(prob_df_clean).mark_bar().encode(x='emotions',y='probability',color = 'emotions')
                    st.altair_chart(fig,use_container_width = True)

    else:
        st.header("Manage and Monitor Results")
        stored_data = view_data()
        new_df = pd.DataFrame(stored_data,columns = ["message","prediction","probability","sentiment"])
        st.dataframe(new_df)

        # with st.expander("Prediction Distribution"):
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Prediction Distribution")
            fig2 = plt.figure()
            sb.countplot(x = 'prediction',data = new_df)
            st.pyplot(fig2)
        
        with col2:
            st.subheader("Sentiment Distribution")
            fig3 = plt.figure()
            sb.countplot(x = 'sentiment',data = new_df)
            st.pyplot(fig3)

            








if __name__ == "__main__":
    main()
