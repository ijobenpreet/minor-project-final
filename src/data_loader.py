import os
import pandas as pd

def load_nbaiot_dataset(base_path):
    all_data = []

    # Loop through each device folder
    for device in os.listdir(base_path):
        device_path = os.path.join(base_path, device)

        if not os.path.isdir(device_path):
            continue

        print(f"[INFO] Processing device: {device}")

        # Loop through CSV files inside device
        for file in os.listdir(device_path):
            if file.endswith(".csv"):
                file_path = os.path.join(device_path, file)

                try:
                    df = pd.read_csv(file_path)

                    # Label creation
                    if "benign" in file.lower():
                        df["label"] = 0
                    else:
                        df["label"] = 1

                    df["device"] = device

                    all_data.append(df)

                except Exception as e:
                    print(f"[ERROR] Skipping {file}: {e}")

    # Combine all
    final_df = pd.concat(all_data, ignore_index=True)

    return final_df


if __name__ == "__main__":
    base_path = "data/raw/N_BaIoT"

    df = load_nbaiot_dataset(base_path)

    print("[INFO] Dataset shape:", df.shape)

    df.to_csv("data/processed/nbaiot_combined.csv", index=False)

    print("[INFO] Saved to data/processed/nbaiot_combined.csv")
