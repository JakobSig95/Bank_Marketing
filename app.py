import streamlit as st


## page stats

st.set_page_config(
    page_title="Bank marketing",
    page_icon="💸")

st.title('Bank marketing predicting subscription')

st.title('Bank Marketing 💸')

tab1, tab2, tab3 = st.tabs(["Bansk data", "SML", "UML"])

with tab1:

    st.header('find text senere ')
    
    with st.sidebar:
    "Made by:"
    "Alpha"
    "Jakob"
    "Mikkel"

    st.header("age distribution")
    st.plotly_chart(age_distribution, use_container_width=True)

        with st.expander("Age "):
            st.write("""
               det frinder vi ud af 
            """)


# load the model from disk
#loaded_model = pickle.load(open('model_xgb.pkl', 'rb'))
# result = loaded_model.score(X_test, y_test)
# y_pred = loaded_model.predict(X_test)

#
