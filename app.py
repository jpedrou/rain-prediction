import streamlit as st
import pandas as pd
import pickle


# Interface
st.set_page_config(layout="wide")
left_col, right_col = st.columns(2, gap="large")

with open("app.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with left_col:
    st.title("Rain Prediction in Australia")
    st.subheader("Application to predict if it will rain tomorrow in Australia")

    st.markdown("")
    st.markdown("Please fill the options to get a new prediction.")
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4, gap="small")
    date = col1.date_input(label="What day is today ?")
    min_temp = col2.number_input(label="Min Temp", value=0.0)
    max_temp = col3.number_input(label="Max Temp", value=0.0)
    rainfall = col4.number_input(label="Rainfall", value=0.0)

    col5, col6, col7, col8 = st.columns(4, gap="small")
    evap = col5.number_input(label="Evaporation", value=0.0)
    sun = col6.number_input(label="Sunshine", value=0.0)
    windgustdir = col7.selectbox(
        label="WindGustDir",
        options=[
            "SSE",
            "SE",
            "WSW",
            "SW",
            "E",
            "ENE",
            "S",
            "SSW",
            "N",
            "NN",
            "WNW",
            "NE",
            "NW",
            "ESE",
            "NNW",
        ],
    )
    windgustspeed = col8.number_input(label="WindGustSpeed", value=0.0)

    col9, col10, col11, col12 = st.columns(4, gap="small")
    windd9 = col9.selectbox(
        label="WindDir9am",
        options=[
            "N",
            "ESE",
            "SSW",
            "SW",
            "SSE",
            "NW",
            "S",
            "NE",
            "WSW",
            "SE",
            "W",
            "NNW",
            "E",
            "NNE",
            "WNW",
            "ENE",
        ],
    )
    windd3 = col10.selectbox(
        label="WindDir3pm",
        options=[
            "NNE",
            "ESE",
            "SSE",
            "ENE",
            "SE",
            "WSW",
            "NW",
            "SSW",
            "W",
            "E",
            "N",
            "WNW",
            "S",
            "SW",
            "NE",
            "NNW",
        ],
    )
    wins9 = col11.number_input(label="WindSpeed9am", value=0.0)
    wins3 = col12.number_input(label="WindSpeed3pm", value=0.0)

    col13, col14, col15, col16 = st.columns(4, gap="small")
    h9 = col13.number_input(label="Humidity9am", value=0.0)
    h3 = col14.number_input(label="Humidity3pm", value=0.0)
    p9 = col15.number_input(label="Pressure9am", value=0.0)
    p3 = col16.number_input(label="Pressure3pm", value=0.0)

    col17, col18, col19, col20 = st.columns(4, gap="small")
    c9 = col13.number_input(label="Cloud9am", value=0.0)
    c3 = col14.number_input(label="Cloud3pm", value=0.0)
    t9 = col15.number_input(label="Temp9am", value=0.0)
    t3 = col16.number_input(label="Temp3pm", value=0.0)

    raintoday = st.selectbox(label="It rained today ?", options=["Yes", "No"])

    st.markdown("")
    button = st.button(label="Make Prediction")

with right_col:
    st.title("Probability of Rain 🌧️")
    st.markdown("")
    if not button:
        st.subheader("No Predictions available")
        st.subheader(
            "Please, fill all the fields in the left and then click on predict to see the result in the bottom."
        )

    else:

        with open("preprocess_pipe.pkl", "rb") as f:
            pipeline = pickle.load(f)

        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

        new_df = pd.DataFrame()
        new_df["Date"] = [date]
        new_df["MinTemp"] = [min_temp]
        new_df["MaxTemp"] = [max_temp]
        new_df["Rainfall"] = [rainfall]
        new_df["Evaporation"] = [evap]
        new_df["Sunshine"] = [sun]
        new_df["WindGustDir"] = [windgustdir]
        new_df["WindGustSpeed"] = [windgustspeed]
        new_df["WindDir9am"] = [windd9]
        new_df["WindDir3pm"] = [windd3]
        new_df["WindSpeed9am"] = [wins9]
        new_df["WindSpeed3pm"] = [wins3]
        new_df["Humidity9am"] = [h9]
        new_df["Humidity3pm"] = [h3]
        new_df["Pressure9am"] = [p9]
        new_df["Pressure3pm"] = [p3]
        new_df["Cloud9am"] = [c9]
        new_df["Cloud3pm"] = [c3]
        new_df["Temp9am"] = [t9]
        new_df["Temp3pm"] = [t3]
        new_df["RainToday"] = [raintoday]

        new_df_processed = pipeline.transform(new_df)

        prediction = model.predict_proba(new_df_processed)[:, 1]
        prediction = round(prediction[0], 3) * 100

        if prediction > 50:
            st.metric(label="Probability", value=f"{prediction}%", delta="+ Chance of rain")

        else:
            st.metric(label="Probability", value=f"{prediction}%", delta="- Chance of rain")

    st.title("Model's ROC Curve / AUC Score")
    st.image('images/roc_curve.png')