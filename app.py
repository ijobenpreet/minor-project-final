import streamlit as st
import pandas as pd
import numpy as np
from scapy.all import PcapReader
import joblib
import os
from pathlib import Path

st.set_page_config(page_title="IoT Botnet Detector", layout="wide")

MODEL_PATH = "models/cic_model.pkl"

st.markdown("""
    <h1 style='text-align: center; color: #00C8FF;'>
    🔐 IoT Botnet Detection Dashboard
    </h1>
    <p style='text-align: center;'>Real-time Network Threat Analysis</p>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Settings")
packet_limit = st.sidebar.slider("Packet Limit", 1000, 20000, 5000)
sampling_rate = st.sidebar.slider("Sampling Rate (IMPORTANT: keep low)", 1, 5000, 1)

class StateMachine:
    def __init__(self):
        self.state = "NORMAL"
        self.attack_count = 0
        self.total_count = 0

    def update(self, prediction):
        self.total_count += 1

        if prediction == 1:
            self.attack_count += 1

        ratio = self.attack_count / self.total_count

        if ratio > 0.5:
            self.state = "ATTACK"
        elif ratio > 0.2:
            self.state = "SUSPICIOUS"
        else:
            self.state = "NORMAL"

        return self.state, ratio

st.subheader("📂 Select PCAP Input")

option = st.radio("Choose method:", [
    "Browse File",
    "Enter File Path",
    "Select from Folder"
])

input_path = None

if option == "Browse File":
    uploaded_file = st.file_uploader("Upload PCAP", type=["pcap", "pcapng"])

    if uploaded_file:
        temp_path = "temp_uploaded.pcap"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        input_path = temp_path
        st.success("File uploaded successfully ✅")

elif option == "Enter File Path":
    path = st.text_input("Enter full file path")

    if path and os.path.exists(path):
        input_path = path
        st.success("Valid file path ✅")
    elif path:
        st.error("Invalid path ❌")

elif option == "Select from Folder":
    folder = st.text_input("Enter folder path")

    if folder and os.path.exists(folder):
        files = list(Path(folder).glob("*.pcap")) + list(Path(folder).glob("*.pcapng"))

        if files:
            selected = st.selectbox("Select PCAP", files)
            input_path = str(selected)
        else:
            st.warning("No PCAP files found")

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(file_path, limit, sampling):
    data = []
    progress = st.progress(0)

    packets = PcapReader(str(file_path))

    for i, pkt in enumerate(packets):
        if i >= limit:
            break

        if i % sampling != 0:
            continue

        try:
            length = len(pkt)

            if pkt.haslayer("TCP"):
                proto = 6
            elif pkt.haslayer("UDP"):
                proto = 17
            elif pkt.haslayer("ICMP"):
                proto = 1
            else:
                proto = 0

            data.append([length, proto])

            if i % 100 == 0:
                progress.progress(min(i / limit, 1.0))

        except:
            continue

    return pd.DataFrame(data, columns=["packet_length", "protocol"])

# ---------------- PREPARE ----------------
def prepare_features(df, model):
    X = df.select_dtypes(include=[np.number]).values
    expected = model.n_features_in_

    if X.shape[1] != expected:
        X = np.resize(X, (X.shape[0], expected))

    return X

# ---------------- ANALYTICS ----------------
def calculate_risk(preds):
    total = len(preds)
    attacks = int(np.sum(preds))
    ratio = attacks / total if total > 0 else 0

    score = (ratio * 70) + (attacks / 1000 * 30)

    if score > 70:
        level = "HIGH"
    elif score > 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level, ratio

def timeline(preds):
    window = max(10, len(preds)//20)
    return [
        sum(preds[i:i+window]) / len(preds[i:i+window])
        for i in range(0, len(preds), window)
        if len(preds[i:i+window]) > 0
    ]

def guess_attack_type(preds):
    ratio = sum(preds) / len(preds)

    if ratio > 0.7:
        return "DDoS / Flood Attack"
    elif ratio > 0.4:
        return "Botnet Communication"
    else:
        return "Scanning Activity"

# ---------------- MAIN ----------------
if input_path:
    st.info("Processing PCAP file...")

    df = extract_features(input_path, packet_limit, sampling_rate)

    if df.empty:
        st.error("No packets extracted ❌ (Reduce sampling rate)")
    else:
        model = joblib.load(MODEL_PATH)

        X = prepare_features(df, model)
        preds = model.predict(X)

        score, level, ratio = calculate_risk(preds)

        # State Machine
        sm = StateMachine()
        for p in preds:
            state, _ = sm.update(p)

        attack_type = guess_attack_type(preds)

        # ---------------- METRICS ----------------
        st.subheader("📊 Analysis Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📦 Packets", len(preds))
        col2.metric("🚨 Attacks", int(np.sum(preds)))
        col3.metric("📉 Attack %", f"{ratio*100:.2f}%")
        col4.metric("⚠️ Risk", level)

        st.progress(min(score / 100, 1.0))

        # ---------------- DISTRIBUTION ----------------
        st.subheader("📊 Traffic Distribution")

        attack_count = int(np.sum(preds))
        benign_count = len(preds) - attack_count

        chart_data = pd.DataFrame({
            "Type": ["Benign", "Attack"],
            "Count": [benign_count, attack_count]
        })

        st.bar_chart(chart_data.set_index("Type"))

        # ---------------- TIMELINE ----------------
        st.subheader("📈 Traffic Timeline")

        timeline_data = timeline(preds)

        if len(timeline_data) > 1:
            st.line_chart(timeline_data)
        else:
            st.warning("Not enough data for timeline")

        # ---------------- HEATMAP ----------------
        st.subheader("🔥 Attack Intensity Map")

        if len(preds) >= 100:
            heatmap = np.array(preds[:100]).reshape(10, 10)
            st.dataframe(heatmap)
        else:
            st.warning("Need at least 100 packets for heatmap")

        # ---------------- RESULT ----------------
        st.subheader("🚨 Detection Result")

        if state == "ATTACK":
            st.error("🚨 BOTNET DETECTED")
        elif state == "SUSPICIOUS":
            st.warning("⚠️ Suspicious Traffic")
        else:
            st.success("✅ Normal Traffic")

        # ---------------- INSIGHTS ----------------
        st.subheader("🧠 Insights")

        st.write(f"**Attack Type:** {attack_type}")
        st.write(f"**Final State:** {state}")
        st.write(f"**Confidence Score:** {score:.2f}")

        if ratio > 0.6:
            st.error("High probability of coordinated attack traffic.")
        elif ratio > 0.3:
            st.warning("Traffic shows anomaly patterns.")
        else:
            st.success("Traffic appears normal.")

else:
    st.info("Please select a PCAP file to start analysis.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("🚀 Final Year Project | IoT Botnet Detection System")
