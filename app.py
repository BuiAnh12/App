import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# Load your SVM model for different C values
@st.cache(allow_output_mutation=True)
def load_model(C):
    return joblib.load(f'svm_model_rbf_C{C}.joblib')

# Load the StandardScaler from your saved model
@st.cache(allow_output_mutation=True)
def load_scaler(C):
    return joblib.load(f'scaler_C{C}.joblib')

# Read data from graph.txt
def read_graph_data():
    data = []
    with open("graph.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("C = "):
                data.append(float(line.split()[-1]))
    return data

st.title("SVM on MNIST")

section = st.sidebar.selectbox(
    "Select section", ("About", "Change parameter", "Upload picture")
)

def add_parameter_gui(section):
    params = dict()
    if section == "Change parameter":
        sigma = st.sidebar.slider("C", 1, 25, step=1)
        params["C"] = sigma
    return params

val = add_parameter_gui(section)

# Show the selected parameter
if section == "Change parameter":
    # Extract the relevant scores for the selected C value
    selected_config = f"Configuration: {{'kernel': 'rbf', 'C': {val['C']}}}"
    scores = []
    with open("results.txt", "r") as file:
        lines = file.read().split("\n\n")
        for config_and_scores in lines:
            if selected_config in config_and_scores:
                scores = config_and_scores.split("\n")

    # Display the accuracy and F1 score for the selected configuration
    for score in scores:
        if score.startswith("Normal Accuracy: ") or score.startswith("F1 Score: "):
            st.markdown(f'<p style="font-size: 20px;"><strong>{score}</strong></p>', unsafe_allow_html=True)
        else:
            pass

    data = pd.read_csv('result.csv')

    fig = px.line(data, x="C", y=["Normal Accuracy", "F1 Score"], markers=True)
    # fig.update_layout(hovermode="x") 

    fig.update_layout(
        title=section,  # Set the title to the selected section
        xaxis_title="C Parameter",
        yaxis_title="Score",
    )



    # Highlight the selected C value on the graph
    fig.add_shape(
        type="line",
        x0=val["C"],
        x1=val["C"],
        y0=min(data["Normal Accuracy"].min(), data["F1 Score"].min()),
        y1=max(data["Normal Accuracy"].max(), data["F1 Score"].max()),
        line=dict(color="red", width=2, dash="dash"),
    )

        # Customize the legend labels
    fig.update_layout(legend=dict(title_text="Score"))

    # Display the graph in Streamlit
    st.plotly_chart(fig)
