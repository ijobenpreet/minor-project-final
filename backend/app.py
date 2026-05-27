from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import joblib
import numpy as np
import pandas as pd
from scapy.all import PcapReader
import traceback

app = FastAPI()

# ✅ CORS FIX (IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/cic_model.pkl"


@app.post("/analyze-path")
def analyze_path(path: str):
    try:
        path = path.strip().replace('"', '')

        data = []
        max_packets = 500   # 🔥 keep stable
        sampling = 5

        packets = PcapReader(path)

        for i, pkt in enumerate(packets):
            if i >= max_packets:
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

            except:
                continue

        if len(data) == 0:
            return JSONResponse(content={"error": "No packets extracted"}, status_code=400)

        df = pd.DataFrame(data, columns=["packet_length", "protocol"])

        model = joblib.load(MODEL_PATH)

        X = df.select_dtypes(include=[np.number]).values
        expected = model.n_features_in_

        if X.shape[1] != expected:
            X = np.resize(X, (X.shape[0], expected))

        preds = model.predict(X)

        total = len(preds)
        attacks = int(np.sum(preds))
        ratio = attacks / total

        return JSONResponse(content={
            "total_packets": total,
            "attack_count": attacks,
            "attack_ratio": ratio
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

model = joblib.load(MODEL_PATH)
print("MODEL TYPE:", type(model))
