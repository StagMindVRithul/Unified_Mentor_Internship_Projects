import streamlit as st
import pickle
import numpy as np
import pandas as pd 

# Load the trained model and scaler
with open("best_svm.pkl", "rb") as model_file:
    best_svm = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Set page title and icon
st.set_page_config(page_title="📱 Mobile Price Prediction", layout="wide")

# Title and description
st.markdown(
    "<h1 style='text-align: center;'>📱 Mobile Price Prediction</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center;'> Enter the correct details to predict the price range </h3>", 
    unsafe_allow_html=True
)

st.markdown("---")  # Adds a horizontal divider

# Columns layout for better spacing
col1, col2 = st.columns(2)

with col1:
    st.markdown("### **🔹 Select Mobile Features**")

    def styled_radio(label, options):
        return st.radio(f"**{label}**", options, index=0)

    blue = styled_radio("Bluetooth", ["Select an option", "No", "Yes"])
    dual_sim = styled_radio("Dual SIM", ["Select an option", "No", "Yes"])
    four_g = styled_radio("4G", ["Select an option", "No", "Yes"])
    three_g = styled_radio("3G", ["Select an option", "No", "Yes"])
    touch_screen = styled_radio("Touch Screen", ["Select an option", "No", "Yes"])
    wifi = styled_radio("WiFi", ["Select an option", "No", "Yes"])

    # Convert selections to binary
    def to_binary(value):
        return 1 if value == "Yes" else (0 if value == "No" else None)

    blue, dual_sim, four_g, three_g, touch_screen, wifi = map(to_binary, [blue, dual_sim, four_g, three_g, touch_screen, wifi])

    # Validation check
    if None in [blue, dual_sim, four_g, three_g, touch_screen, wifi]:
        st.warning("⚠️ Please select an option for all Yes/No fields.")

with col2:
    st.markdown("### **🔹 Enter Specifications**")

    def styled_number_input(label, min_value, step, value=None):
        return st.number_input(f"**{label}**", min_value=min_value, step=step, value=value)

    battery_power = styled_number_input("Battery Power (mAh)", 0, 1)
    clock_speed = st.slider("**Clock Speed (GHz)**", 0.0, 3.0, 1.0, step=0.5)
    fc = st.slider("**Front Camera (MP)**", 0, 19, 5, step=1)
    int_memory = st.slider("**Internal Memory (GB)**", 0, 64, 16, step=16)
    m_dep = st.slider("**Mobile Depth (cm)**", 0.1, 1.0, 0.5, step=0.1)
    mobile_wt = styled_number_input("Mobile Weight (g)", 0, 1)
    pc = st.slider("**Primary Camera (MP)**", 0, 20, 10, step=1)
    px_height = styled_number_input("Pixel Height", 0, 1)
    px_width = styled_number_input("Pixel Width", 0, 1)
    ram = styled_number_input("RAM (MB)", 0, 1)
    sc_h = st.slider("**Screen Height (cm)**", 5, 20, 10, step=1)
    sc_w = st.slider("**Screen Width (cm)**", 0, 20, 5, step=1)
    talk_time = st.slider("**Talk Time (hours)**", 2, 20, 10, step=1)
    n_cores = st.selectbox("**Number of Cores**", [1, 2, 3, 4, 5, 6, 7, 8])

st.markdown("---")  # Adds a horizontal divider

# Arrange in DataFrame
feature_names = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g", "int_memory", 
    "m_dep", "mobile_wt", "n_cores", "pc", "px_height", "px_width", "ram", 
    "sc_h", "sc_w", "talk_time", "three_g", "touch_screen", "wifi"
]

user_input_df = pd.DataFrame([[
    battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, 
    mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, 
    three_g, touch_screen, wifi
]], columns=feature_names)

# Scale the input
scaled_input = scaler.transform(user_input_df)

# Prediction Button
if st.button("🔍 **Predict Price Range**"):
    prediction = best_svm.predict(scaled_input)[0]

    price_mapping = {
        0: "🟢 **Low Cost** 💰",
        1: "🟡 **Medium Cost** 💵",
        2: "🔵 **High Cost** 💎",
        3: "🔴 **Very High Cost** 🚀"
    }

    st.markdown("## **💰 Predicted Price Range:**")
    st.success(price_mapping[prediction])

