import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from scipy.signal import butter, filtfilt, find_peaks

# ✅ Start FastAPI App
app = FastAPI()

# ✅ Root endpoint to verify API is running
@app.get("/")
def home():
    return {"message": "FastAPI ECG Processing Server is Running!"}

# ✅ Butterworth Bandpass Filter
def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=512, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

@app.post("/process_ecg/")
async def process_ecg(ecg_data: dict):
    try:
        time = np.array(ecg_data["time"])
        voltage = np.array(ecg_data["voltage"])

        if len(time) != len(voltage):
            raise ValueError("Time and voltage arrays must have the same length.")

        # ✅ Apply Filtering
        filtered_voltage = butter_bandpass_filter(voltage, lowcut=0.5, highcut=40.0, fs=512)

        # ✅ Detect Peaks with Adaptive Threshold
        peak_threshold = np.mean(filtered_voltage) + (1.8 * np.std(filtered_voltage))
        peaks, _ = find_peaks(filtered_voltage, height=peak_threshold, distance=60)

        # ✅ Keep only real R-peaks using improved RR-interval filtering
        true_peaks = []
        for i in range(len(peaks)):
            if i == 0 or (0.3 < (time[peaks[i]] - time[peaks[i - 1]]) < 1.5):
                true_peaks.append(peaks[i])

        return {
            "time": time.tolist(),
            "filtered_voltage": filtered_voltage.tolist(),
            "peaks": [time[i] for i in true_peaks]
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
