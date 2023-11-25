import joblib
import pandas as pd
import numpy as np
import streamlit as st

df = pd.read_csv("../emails.csv")

model = joblib.load('model.pkl')

st.title("Spam")
new_mail = st.text_input('Input mail', "")

def transform(mail):
    # Neue Mail als String -> Liste mit einzelnen Wörtern der Mail
    # new_mail = "Hello I need the help"
    word_list = new_mail.lower().split()
    # print(word_list)


    # Kopie des Datensatzes
    new_df = df.copy()


    # Entfernen der ersten und letzten Spalte des Dataframes -> entsprechendes Format
    new_df.drop(['Email No.', 'Prediction'], inplace=True, axis=1)


    # Neuer Dataframe
    data = pd.DataFrame(columns=new_df.columns)


    # Erfassen der Häufigkeit der Worte in der Mail
    word_count = {word: word_list.count(word) for word in data.columns}
    data = pd.concat([data, pd.DataFrame([word_count])], ignore_index=True)


    # Auffüllen der fehlenden Werte
    data = data.fillna(0)

    return data


mail = transform(new_mail)

def predict():
    prediction = model.predict(mail)

    if(prediction==1):
        st.error('Spam')
    else:
        st.success('Ham')


st.button("Prediction", on_click=predict)