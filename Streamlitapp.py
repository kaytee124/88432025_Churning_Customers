import streamlit as st
import pickle
import numpy as np
import pandas as pd
with open('optimized_model.pkl', 'rb') as file:
  loaded_model=pickle.load(file)

with open('scaler.pkl', 'rb') as scaler_file:
  loaded_scaler=pickle.load(scaler_file)

with open('label_encoder.pkl', 'rb') as label_encoder:
  loaded_label=pickle.load(label_encoder)


# Create a Streamlit app
st.title("Chrun Prediction")


feature1 = st.number_input("tenure")
feature2 = st.number_input("Monthly Charges")
feature3= st.text_input("Total Charges")
feature4= st.text_input("Contract")
feature5= st.text_input("Payment Method")
feature6= st.text_input("Online Security")
feature7= st.text_input("TechSupport")



feature1 = float(feature1) 
feature2 = float(feature2) 
feature3 = str(feature3) 
feature4 = str(feature4) 
feature5 =str(feature5) 
feature6 =str(feature6) 
feature7 =str(feature7)



feature = []
feature.append(feature1)
feature.append(feature2)


nonnumfeature =[]
nonnumfeature.append(feature3)
nonnumfeature.append(feature4)
nonnumfeature.append(feature5)
nonnumfeature.append(feature6)
nonnumfeature.append(feature7)




dffeature = pd.DataFrame(feature)
dfnonnumfeature = pd.DataFrame(nonnumfeature)


dffeature= dffeature.T



for column in dfnonnumfeature.columns:
        dfnonnumfeature[column] = loaded_label.fit_transform(dfnonnumfeature[column])

dfnonnumfeature = dfnonnumfeature.T

dfnonnumfeature.columns = [

    "TotalCharges",	"Contract", "PaymentMethod", "OnlineSecurity",	"TechSupport"
]


dffeature.columns= [
    "tenure","MonthlyCharges"
]




newfeature = pd.concat([dffeature, dfnonnumfeature], axis=1)



scaled = loaded_scaler.transform(newfeature)
newfeature=pd.DataFrame(scaled, columns=newfeature.columns)

if st.button("Predict"):

    prediction = loaded_model.predict(newfeature)
    confidence = prediction.squeeze()


    if confidence >0.5:
        st.warning('Churn: Yes')
        
    else:
        st.success('Churn : No')


