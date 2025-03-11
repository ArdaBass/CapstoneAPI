import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from scipy.signal import butter, filtfilt, find_peaks
import nest_asyncio
from pyngrok import ngrok

# ✅ Start FastAPI App
app = FastAPI()

# ✅ Butterworth Bandpass Filter
def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=512, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

@app.post("/process_ecg/")
async def process_ecg(ecg_data: dict):
    time = np.array(ecg_data["time"])
    voltage = np.array(ecg_data["voltage"])

    # ✅ Apply Filtering
    filtered_voltage = butter_bandpass_filter(voltage, lowcut=0.5, highcut=40.0, fs=512)

    # ✅ Detect Peaks with Adaptive Threshold
    peak_threshold = np.mean(filtered_voltage) + (1.8 * np.std(filtered_voltage))
    peaks, _ = find_peaks(filtered_voltage, height=peak_threshold, distance=60)

    # ✅ Keep only real R-peaks using RR-interval filtering
    true_peaks = [peaks[0]]
    for i in range(1, len(peaks)):
        rr_interval = time[peaks[i]] - time[true_peaks[-1]]
        if 0.3 < rr_interval < 1.5:  # Valid R-R intervals
            true_peaks.append(peaks[i])

    return {"time": time.tolist(), "filtered_voltage": filtered_voltage.tolist(), "peaks": [time[i] for i in true_peaks]}

# ✅ Run API in Colab
nest_asyncio.apply()
ngrok.set_auth_token("2u1N6QrfR5w5WL0VIigieM38PNt_64c3NLWN5s4v5Ck3eSxTV")  # Add your token here
ngrok_tunnel = ngrok.connect(8000)
print(f"Public URL: {ngrok_tunnel.public_url}")

uvicorn.run(app, host="0.0.0.0", port=8000)
