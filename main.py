from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fftpack import fft
import base64
import io
import pymssql
import os
import uuid
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- pymssql DB Connection ----------------
def get_db_connection():
    return pymssql.connect(
        server="aktekworkers.database.windows.net",
        user="sqladmin",
        password="nd1W594.",
        database="capstone"
    )

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- Models ----------------
class FolderCreate(BaseModel):
    name: str
    parent_id: Optional[int] = None

# ---------------- Folder Routes ----------------
@app.post("/folders")
def create_folder(folder: FolderCreate):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Folders (Name, ParentId) VALUES (%s, %s)", (folder.name, folder.parent_id))
        conn.commit()
        cursor.close()
        conn.close()
        return {"message": "Folder created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/folders")
def get_top_level_folders():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT Id, Name, ParentId FROM Folders WHERE ParentId IS NULL")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [
            {"id": row[0], "name": row[1], "parent_id": row[2]} for row in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/folders/{parent_id}/subfolders")
def get_subfolders(parent_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT Id, Name, ParentId FROM Folders WHERE ParentId = %s", (parent_id,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [
            {"id": row[0], "name": row[1], "parent_id": row[2]} for row in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{folder_id}")
def list_files_in_folder(folder_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT Id, FileName FROM Files WHERE FolderId = %s", (folder_id,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [{"id": row[0], "name": row[1]} for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{file_id}")
def download_file(file_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT FilePath, FileName FROM Files WHERE Id = %s", (file_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if not row or not os.path.exists(row[0]):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path=row[0], filename=row[1], media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/folders/{folder_id}")
def delete_folder(folder_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Folders WHERE Id = %s", (folder_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return {"message": "Folder deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/folders/{folder_id}")
def rename_folder(folder_id: int, new_name: str = Form(...)):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE Folders SET Name = %s WHERE Id = %s", (new_name, folder_id))
        conn.commit()
        cursor.close()
        conn.close()
        return {"message": "Folder renamed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- ECG Utils ----------------
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
        "LF_Power": lf_power,
        "HF_Power": hf_power,
        "DFA_alpha1": dfa_alpha1,
    }

# ---------------- Analyze Route ----------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...), start_index: int = Form(0)):
    try:
        df = pd.read_csv(file.file, delimiter=";", decimal=",", skiprows=[1])
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
            raise ValueError(f"Start index {start_index} is out of range.")

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

        buf = io.BytesIO()
        plt.figure(figsize=(12, 5))
        plt.plot(trimmed_time, trimmed_voltage, color='blue')
        plt.scatter(trimmed_time[true_peaks_trimmed], trimmed_voltage[true_peaks_trimmed], color='red')
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "image": encoded_image,
            "hrvMetrics": hrv,
            "rrTable": [
                {"timestamp": true_peak_times_trimmed[i], "rr": None if i == 0 else rr_intervals_trimmed[i - 1]}
                for i in range(len(true_peak_times_trimmed))
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------------- Upload & File Operations ----------------
@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...), folder_id: int = Form(...)):
    try:
        file_ext = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4().hex}{file_ext}"
        saved_path = os.path.join(UPLOAD_DIR, unique_name)

        with open(saved_path, "wb") as f:
            f.write(await file.read())

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Files (FolderId, FileName, FilePath, UploadedAt) VALUES (%s, %s, %s, %s)",
            (folder_id, file.filename, saved_path, datetime.utcnow())
        )
        conn.commit()
        cursor.close()
        conn.close()

        return {"message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/files/{file_id}/move")
async def move_file(file_id: int, new_folder_id: int = Form(...)):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE Files SET FolderId = %s WHERE Id = %s", (new_folder_id, file_id))
        conn.commit()
        cursor.close()
        conn.close()
        return {"message": "File moved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/files/{file_id}/rename")
def rename_file(file_id: int, new_name: str = Form(...)):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE Files SET FileName = %s WHERE Id = %s", (new_name, file_id))
        conn.commit()
        cursor.close()
        conn.close()
        return {"message": "File renamed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{file_id}")
def delete_file(file_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT FilePath FROM Files WHERE Id = %s", (file_id,))
        row = cursor.fetchone()
        if row and os.path.exists(row[0]):
            os.remove(row[0])
        cursor.execute("DELETE FROM Files WHERE Id = %s", (file_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return {"message": "File deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
