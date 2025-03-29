from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fftpack import fft
import base64
import io
import sqlalchemy
from sqlalchemy import create_engine, text
import os
import uuid
from datetime import datetime
import urllib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- SQLAlchemy Engine with Pooling ----------------
conn_str = urllib.parse.quote_plus(
    "DRIVER={ODBC Driver 17 for SQL Server};SERVER=aktekworkers.database.windows.net;UID=sqladmin;PWD=nd1W594.;DATABASE=capstone"
)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={conn_str}", pool_pre_ping=True, pool_size=5, max_overflow=10)

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
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO Folders (Name, ParentId) VALUES (:name, :parent_id)"), folder.dict())
        return {"message": "Folder created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/folders")
def get_folders():
    try:
        with engine.connect() as conn:
            folder_rows = conn.execute(text("SELECT Id, Name, ParentId FROM Folders")).fetchall()
            file_rows = conn.execute(text("SELECT Id, FolderId, FileName FROM Files")).fetchall()

        folders = []
        for folder in folder_rows:
            files = [
                {"id": f.Id, "name": f.FileName} for f in file_rows if f.FolderId == folder.Id
            ]
            folders.append({
                "id": folder.Id,
                "name": folder.Name,
                "parent_id": folder.ParentId,
                "files": files
            })
        return folders
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/{folder_id}")
def list_files_in_folder(folder_id: int):
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT Id, FileName FROM Files WHERE FolderId = :fid"), {"fid": folder_id}).fetchall()
        return [{"id": row.Id, "name": row.FileName} for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{file_id}")
def download_file(file_id: int):
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT FilePath, FileName FROM Files WHERE Id = :fid"), {"fid": file_id}).fetchone()
        if not row or not os.path.exists(row.FilePath):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path=row.FilePath, filename=row.FileName, media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/folders/{folder_id}")
def delete_folder(folder_id: int):
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM Folders WHERE Id = :fid"), {"fid": folder_id})
        return {"message": "Folder deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/folders/{folder_id}")
def rename_folder(folder_id: int, new_name: str = Form(...)):
    try:
        with engine.begin() as conn:
            conn.execute(text("UPDATE Folders SET Name = :name WHERE Id = :fid"), {"name": new_name, "fid": folder_id})
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

        with engine.begin() as conn:
            conn.execute(text("INSERT INTO Files (FolderId, FileName, FilePath, UploadedAt) VALUES (:fid, :fn, :fp, :up)"), {
                "fid": folder_id,
                "fn": file.filename,
                "fp": saved_path,
                "up": datetime.utcnow()
            })

        return {"message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/files/{file_id}/move")
async def move_file(file_id: int, new_folder_id: int = Form(...)):
    try:
        with engine.begin() as conn:
            conn.execute(text("UPDATE Files SET FolderId = :fid WHERE Id = :id"), {"fid": new_folder_id, "id": file_id})
        return {"message": "File moved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/files/{file_id}/rename")
def rename_file(file_id: int, new_name: str = Form(...)):
    try:
        with engine.begin() as conn:
            conn.execute(text("UPDATE Files SET FileName = :name WHERE Id = :id"), {"name": new_name, "id": file_id})
        return {"message": "File renamed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{file_id}")
def delete_file(file_id: int):
    try:
        with engine.begin() as conn:
            row = conn.execute(text("SELECT FilePath FROM Files WHERE Id = :id"), {"id": file_id}).fetchone()
            if row and os.path.exists(row.FilePath):
                os.remove(row.FilePath)
            conn.execute(text("DELETE FROM Files WHERE Id = :id"), {"id": file_id})
        return {"message": "File deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
