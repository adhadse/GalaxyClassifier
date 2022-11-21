import joblib
import numpy as np
import streamlit as st


st.title('Galaxy Classifier')
st.markdown("""
 Galaxy Classifier is able to classify a galaxy into \n3 possible categories of
- Eliptical
- Spiral
- Hybrid.
""")

model_filename = "random_forest_classifier.sav"

rfc_model = joblib.load(model_filename)

u_g = st.number_input("u-g")
g_r = st.number_input("g-r")
r_i = st.number_input("r-i")
i_z = st.number_input("i-z")
ecc = st.number_input("ecc")
m4_u = st.number_input("m4_u")
m4_g = st.number_input("m4_g")
m4_r = st.number_input("m4_r")
m4_i = st.number_input("m4_i")
m4_z = st.number_input("m4_z")
petroR50_u = st.number_input("petroR50_u")
petroR90_u = st.number_input("petroR90_u", value=0.0001)  # avoid dividing by zero
petroR50_r = st.number_input("petroR50_r")
petroR90_r = st.number_input("petroR90_r", value=0.0001)
petroR50_z = st.number_input("petroR50_z")
petroR90_z = st.number_input("petroR90_z", value=0.0001)

conc_in_u_filter = petroR50_u / petroR90_u
conc_in_r_filter = petroR50_r / petroR90_r
conc_in_z_filter = petroR50_z / petroR90_z


def predict():
    row = np.array([[u_g, g_r, r_i, i_z, ecc, m4_u, m4_g, m4_r, m4_i, m4_z,
                     conc_in_u_filter, conc_in_r_filter, conc_in_z_filter]])
    st.success(f"It's a {rfc_model.predict(row)[0].capitalize()} Galaxy")


st.button('Predict', on_click=predict)
