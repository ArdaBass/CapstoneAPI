from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fftpack import fft
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can change to your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=512, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_hrv_metrics(rr_intervals):
    rr = np.array(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else None
    pnn50 = np.sum(np.abs(np.diff(rr)) > 0.05) / len(rr) * 100 if len(rr) > 1 else None
    sd1 = np.std(rr) / np.sqrt(2) if len(rr) > 1 else None
    lf_power = hf_power = None
    if len(rr) > 1:
        rr_fft = np.abs(fft(rr - np.mean(rr)))[:len(rr)//2]
        freqs = np.fft.fftfreq(len(rr), d=np.mean(rr))[:len(rr)//2]
        lf_power = np.sum(rr_fft[(freqs >= 0.04) & (freqs < 0.15)])
        hf_power = np.sum(rr_fft[(freqs >= 0.15) & (freqs < 0.4)])
    dfa_alpha1 = np.std(np.log(rr)) if len(rr) > 1 else None

    return {
        "RMSSD": rmssd,
        "pNN50": pnn50,
        "SD1": sd1,
        "LF Power": lf_power,
        "HF Power": hf_power,
        "DFA-α1": dfa_alpha1
    }

def analyze_ecg(file_bytes, start_index: int = 0):
    df = pd.read_csv(io.BytesIO(file_bytes), delimiter=";", decimal=",", skiprows=[1])
    df.columns = ["Time (s)", "Voltage (mV)"]
    df = df.astype(float)

    time = df["Time (s)"].values
    voltage = df["Voltage (mV)"].values * 1000
    filtered = butter_bandpass_filter(voltage)
    filtered = np.clip(filtered, -600, 600)

    threshold = np.mean(filtered) + 1.8 * np.std(filtered)
    peaks, _ = find_peaks(filtered, height=threshold, distance=60)

    true_peaks = []
    true_peak_times = []
    if len(peaks):
        true_peaks.append(peaks[0])
        true_peak_times.append(time[peaks[0]])
        for i in range(1, len(peaks)):
            rr = time[peaks[i]] - time[true_peaks[-1]]
            if 0.3 < rr < 1.5:
                true_peaks.append(peaks[i])
                true_peak_times.append(time[peaks[i]])

    if start_index >= len(true_peaks):
        raise ValueError("Invalid start index!")

    start_time = time[true_peaks[start_index]]
    mask = time >= start_time
    trimmed_time = time[mask] - start_time
    trimmed_voltage = filtered[mask]

    t_peaks, _ = find_peaks(trimmed_voltage, height=np.mean(trimmed_voltage) + 1.8 * np.std(trimmed_voltage), distance=60)
    true_peaks_trimmed, rr_intervals_trimmed, true_peak_times_trimmed = [], [], []

    if len(t_peaks):
        true_peaks_trimmed.append(t_peaks[0])
        true_peak_times_trimmed.append(trimmed_time[t_peaks[0]])
        for i in range(1, len(t_peaks)):
            rr = trimmed_time[t_peaks[i]] - trimmed_time[true_peaks_trimmed[-1]]
            if 0.3 < rr < 1.5:
                true_peaks_trimmed.append(t_peaks[i])
                true_peak_times_trimmed.append(trimmed_time[t_peaks[i]])
                rr_intervals_trimmed.append(rr)

    hrv = calculate_hrv_metrics(rr_intervals_trimmed)

    # Plot image to buffer
    buf = io.BytesIO()
    plt.figure(figsize=(12, 5))
    plt.plot(trimmed_time, trimmed_voltage, color='blue', label="Trimmed ECG")
    plt.scatter(trimmed_time[true_peaks_trimmed], trimmed_voltage[true_peaks_trimmed], color='red', label="Trimmed Peaks", zorder=3)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (µV)")
    plt.title(f"Trimmed ECG from Peak #{start_index}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return buf, {
        "hrvMetrics": hrv,
        "rrTable": [
            {"timestamp": true_peak_times_trimmed[i], "rr": None if i == 0 else rr_intervals_trimmed[i - 1]}
            for i in range(len(true_peak_times_trimmed))
        ]
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), start_index: int = Form(0)):
    try:
        content = await file.read()
        plot_buf, analysis = analyze_ecg(content, start_index)

        return StreamingResponse(
            plot_buf,
            media_type="image/png",
            headers={"X-Metrics": JSONResponse(content=analysis).body.decode()}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
