from scapy.all import PcapReader
import pandas as pd
import numpy as np
import joblib
from llm_module import generate_explanation

# Paths
PCAP_FILE = "pcaps/Friday-WorkingHours.pcap"
MODEL_PATH = "models/cic_model.pkl"

# Limit packets for speed
PACKET_LIMIT = 10000


# ---------------- STATE MACHINE ----------------
class StateMachine:
    def __init__(self):
        self.state = "NORMAL"
        self.attack_count = 0
        self.total_count = 0

    def update(self, prediction):
        self.total_count += 1

        if prediction == 1:
            self.attack_count += 1

        attack_ratio = self.attack_count / self.total_count

        if attack_ratio > 0.5:
            self.state = "ATTACK"
        elif attack_ratio > 0.2:
            self.state = "SUSPICIOUS"
        else:
            self.state = "NORMAL"

        return self.state, attack_ratio


# ---------------- PCAP FEATURE EXTRACTION ----------------
def extract_features(pcap_file):
    print("[INFO] Reading PCAP file...")

    data = []

    with PcapReader(pcap_file) as packets:
        for i, pkt in enumerate(packets):

            if i >= PACKET_LIMIT:
                break

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

            except:
                continue

    df = pd.DataFrame(data, columns=["packet_length", "protocol"])

    print(f"[INFO] Extracted {len(df)} packets")

    return df


# ---------------- FEATURE ALIGNMENT ----------------
def prepare_features(df, model):
    expected_features = model.n_features_in_

    X = df.select_dtypes(include=[np.number])
    X_array = X.values

    if X_array.shape[1] != expected_features:
        X_array = np.resize(X_array, (X_array.shape[0], expected_features))

    return X_array


# ---------------- PREDICTION ----------------
def predict(df):
    print("[INFO] Loading model...")

    model = joblib.load(MODEL_PATH)

    X_array = prepare_features(df, model)

    print("[INFO] Running predictions...")
    preds = model.predict(X_array)

    return preds

def calculate_risk(preds):
    total = len(preds)
    attacks = int(np.sum(preds))

    ratio = attacks / total if total > 0 else 0

    # Weighted scoring
    score = (ratio * 70) + (attacks / 1000 * 30)

    if score > 70:
        level = "HIGH"
    elif score > 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    return score, level

def packet_timeline(preds):
    timeline = []
    window = 100

    for i in range(0, len(preds), window):
        chunk = preds[i:i+window]
        ratio = sum(chunk) / len(chunk)
        timeline.append(ratio)

    return timeline
def guess_attack_type(preds):
    attack_ratio = sum(preds) / len(preds)

    if attack_ratio > 0.7:
        return "DDoS / Flood Attack"
    elif attack_ratio > 0.4:
        return "Botnet Communication"
    else:
        return "Scanning / Suspicious Activity"

# ---------------- SUMMARY ----------------
def summarize(preds):
    total = len(preds)
    attacks = int(np.sum(preds))
    benign = total - attacks

    print("\n===== PCAP RESULT =====")
    print(f"Total: {total}")
    print(f"Benign: {benign}")
    print(f"Attack: {attacks}")


# ---------------- STATE MACHINE RUN ----------------
def run_state_machine(preds):
    sm = StateMachine()

    print("\n===== STATE MACHINE =====")

    for i, pred in enumerate(preds):
        state, ratio = sm.update(pred)

        if i % 500 == 0:
            print(f"[STEP {i}] {state} | Ratio: {ratio:.2f}")

    print("\nFINAL STATE:", sm.state)
    explanation = generate_explanation(sm.state, ratio, sm.total_count)
    print(explanation)

    if sm.state == "ATTACK":
        print("🚨 BOTNET DETECTED")
    elif sm.state == "SUSPICIOUS":
        print("⚠️ SUSPICIOUS TRAFFIC")
    else:
        print("✅ NORMAL")


# ---------------- MAIN ----------------
def main():
    df = extract_features(PCAP_FILE)

    if df.empty:
        print("[ERROR] No packets found")
        return

    preds = predict(df)

    summarize(preds)

    run_state_machine(preds)


if __name__ == "__main__":
    main()
