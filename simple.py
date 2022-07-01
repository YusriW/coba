import streamlit as st
import plotly_express as px
import numpy as np
#from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import _dist_metrics
import sklearn
import pickle



def main():

    st.title("Menggunakan Sklearn versi " +str(sklearn.__version__))
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
        #st.write(model.__dict__)
    #sl = st.number_input(label="Sepal Length :", min_value=0.0, max_value=8.0, step=0.1, value=5.2)
    sl = st.slider(label="Sepal Length :", min_value=0.0, max_value=8.0, step=0.1, value=5.2)
    sw = st.slider(label="Sepal Width :", min_value=0.0, max_value=8.0, step=0.1, value=3.2)
    pl = st.slider(label="Petal Length :", min_value=0.0, max_value=8.0, step=0.1, value=1.2)
    pw = st.slider(label="Petal Length :", min_value=0.0, max_value=8.0, step=0.1, value=0.2)
    

    if st.button(label="click to predict"):
        user_data = np.array([sl,sw,pl,pw]).reshape(1,-1)
        prediction = model.predict(user_data)[0]
        st.sidebar.header('Hasil Prediksi Input adalah')
        if prediction == 1:
            st.sidebar.write('Iris-Setosa')
        elif prediction ==2:
            st.sidebar.write('Iris-Versicolor')
        elif prediction ==3:
            st.sidebar.write('Iris-Virginica')
        else:
            st.sidebar.write('Pokoknya bunga')


if __name__=='__main__':
    main()