import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore # or any other optimizer you prefer
import datetime

# Load the models
with open("rf_model.pkl", "rb") as file:
    loaded_rf_model = pickle.load(file)

# Load the transfer learned CNN model for fire detection without the optimizer
cnn_model = load_model("transfer_learned_model.h5", compile=False)
cnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Load the data
df = pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Convert target variable to binary numeric format
df["Classes"] = df["Classes"].str.strip()
df["Classes"] = df["Classes"].map({"fire": 1, "not fire": 0})

# Define feature set X
X = df.loc[:, ["Temperature", "Ws", "FFMC", "DMC", "ISI"]]

# Predict fire occurrences using the loaded model
df["predicted_fire"] = loaded_rf_model.predict(X)

# Extract month from the date column
df["date"] = pd.to_datetime(df[["day", "month", "year"]])
df["month"] = df["date"].dt.month

# Group by month and calculate fire occurrence frequency
monthly_fire_frequency = df.groupby("month")[
    "predicted_fire"
].mean()  # Mean gives the frequency of 'fire' (1)

# Define month names
month_names = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# Convert month number to month name
monthly_fire_frequency.index = monthly_fire_frequency.index.map(
    lambda x: month_names[x - 1]
)

# Scale the prediction to a scale of 10
monthly_fire_frequency_scaled = (
    monthly_fire_frequency / monthly_fire_frequency.max()
) * 10

# Streamlit App
st.title("Fire Forest Prediction and Detection")

# Sidebar buttons
st.sidebar.title("Navigation")
fire_forest_prediction_button = st.sidebar.button("Main Page")
fire_detection_button = st.sidebar.button("Testing - Fire Detection")


# Camera feed function
def start_camera():
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        reshaped_frame = np.reshape(normalized_frame, (1, 224, 224, 3))
        prediction = cnn_model.predict(reshaped_frame)
        if prediction[0][0] > 0.5:
            label = "Fire Detected"
            color = (255, 0, 0)
        else:
            label = "No Fire"
            color = (0, 255, 0)
        cv2.putText(
            frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
        )
        frame_image = Image.fromarray(frame)
        frame_placeholder.image(
            frame_image, caption="Camera Feed", use_column_width=True
        )
    cap.release()


if fire_forest_prediction_button:
    st.session_state.page = "fire_forest_prediction"
    st.session_state.camera_active = False

if fire_detection_button:
    st.session_state.page = "fire_detection"
    st.session_state.camera_active = True

# Fire Forest Prediction page
if "page" not in st.session_state:
    st.session_state.page = "fire_forest_prediction"

if st.session_state.page == "fire_forest_prediction":
    st.subheader("Monthly Fire Frequency Prediction values:")
    st.write(monthly_fire_frequency)

    st.subheader("Scaled Monthly Fire Frequency Prediction:")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(monthly_fire_frequency_scaled.index, monthly_fire_frequency_scaled.values)
    ax.set_xlabel("Month")
    ax.set_ylabel("Scaled Frequency of Fire Occurrence (out of 10)")
    ax.set_title("Monthly Fire Frequency Prediction")
    plt.xticks(rotation=45)

    threshold_value = 2.25  
    ax.axhline(y=threshold_value, color='r', linestyle='--', label='Threshold')
    ax.legend()

    st.pyplot(fig)
    st.markdown(
            "<p style='font-size: 14px; font-style: italic;'>The designated threshold value for considering a forest fire occurrence is set at 0.225.</p>",
            unsafe_allow_html=True
        )


    threshold = 0.225
    fire_months = monthly_fire_frequency[monthly_fire_frequency > threshold].index.tolist()  # type: ignore

    # st.subheader("Months predicted to have guaranteed fire occurrence:")
    # st.write(fire_months)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    styled_months = ''.join(
        f'<span style="font-size: 18px; color: white; background-color: gray; padding: 5px 10px; margin: 5px; border-radius: 5px;">{month}</span>'
        for month in fire_months
    )

    st.subheader("Months Predicted to Have Guaranteed Fire Occurrence:")
    st.markdown(styled_months, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    current_month = datetime.datetime.now().strftime("%B")
    st.write("Current month :",current_month)

    if current_month in fire_months :
        st.subheader("Camera Feed")
        st.write("The camera has been activated.")
        st.session_state.camera_active = True
        start_camera()

    # Fire Detection page
    else:
        st.subheader("Camera Feed")
        st.write("The camera will remain off as the current month is not designated as a fire prediction month.")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size: 14px; font-style: italic;'>Select 'Testing - Fire Detection' to perform a test of the fire detection system.</p>",
            unsafe_allow_html=True
        )

if st.session_state.page == "fire_detection":
    start_camera()