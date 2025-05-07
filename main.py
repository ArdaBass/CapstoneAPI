from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
from azure.storage.blob import BlobServiceClient
from pydantic import BaseModel
import traceback





# ---------------- Azure Setup ----------------
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=hrvstoragearda;AccountKey=PC3HHRI4bnsph1dHH96K4t8UyE6Z6nM7Uvgw1AiNVmsQ76DxDuMC+/tkz88nWq1xXmVt2BN+hRjP+AStzuAmEQ==;EndpointSuffix=core.windows.net"
AZURE_CONTAINER = "capstone-files"
blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service.get_container_client(AZURE_CONTAINER)
try:
    container_client.create_container()
except Exception:
    pass

# ---------------- FastAPI Setup ----------------
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

# ---------------- Models ----------------
class FolderCreate(BaseModel):
    name: str
    parent_id: Optional[int] = None

# ---------------- Folder Endpoints ----------------


@app.head("/ping")
def ping_head():
    return {"status": "ok"}


@app.post("/import-participants")
async def import_participants(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")

        # Normalize headers
        df.columns = [col.strip().lower().replace("\u00a0", " ") for col in df.columns]

        conn = get_db_connection()
        cursor = conn.cursor()
        added = 0

        for i, row in df.iterrows():
            person_id = int(i + 1)
            try:
                name = str(row.get("full name", "")).strip()
                if not name:
                    continue

                cursor.execute("SELECT COUNT(*) FROM Participants WHERE Id = %s OR Name = %s", (person_id, name))
                if cursor.fetchone()[0] > 0:
                    continue

                def safe_get(col):
                    val = row.get(col, "")
                    if pd.isna(val):
                        return ""
                    return str(val).strip()

                def parse_float(value):
                    try:
                        cleaned = ''.join(c for c in value if c.isdigit() or c in ['.', ',']).replace(',', '.')
                        return float(cleaned)
                    except:
                        return None

                def parse_int(value):
                    try:
                        return int(''.join(filter(str.isdigit, value)))
                    except:
                        return None

                # Insert into Participants table
                cursor.execute("""
                    INSERT INTO Participants (
                        Id, Name, Age, Stress, SleepHours, SmokingStatus,
                        CaffeineToday, CaffeineDetails, Alcohol, PhysicalActivity,
                        ActivityDetails, Medication, MedicationName,
                        CardioIssues, CardioIssuesName, Rested5Min,
                        RecentIllness, IllnessExplanation
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    person_id,
                    name,
                    parse_int(safe_get("age")),
                    parse_int(safe_get("current stress level (1–10):")),
                    parse_float(safe_get("sleep duration last night (hours):")),
                    safe_get("smoking status:"),
                    safe_get("caffeine intake today"),
                    safe_get("amount and time"),
                    safe_get("alcohol intake (last 24h):"),
                    safe_get("physical activity before test:"),
                    safe_get("type and time"),
                    safe_get("medication"),
                    safe_get("medication name"),
                    safe_get("known cardiovascular issues"),
                    safe_get("cardiovascular issues name"),
                    safe_get("resting for 5 minutes before test:"),
                    safe_get("recent illnesses (past 2 weeks):"),
                    safe_get("please explain")
                ))
                added += 1

                # ✅ Create folder with participant name if not exists
                cursor.execute("SELECT COUNT(*) FROM Folders WHERE Name = %s AND ParentId IS NULL", (name,))
                if cursor.fetchone()[0] == 0:
                    cursor.execute("INSERT INTO Folders (Name, ParentId) VALUES (%s, NULL)", (name,))



            except Exception as row_err:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error on row {i+2}: {row_err}"
                )

        conn.commit()
        cursor.close()
        conn.close()

        return {"message": f"{added} new participants added and folders created."}

    except Exception as e:
        print("Traceback:", traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Import failed: {e}")




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
def get_folders():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT Id, Name, ParentId FROM Folders")
        folder_rows = cursor.fetchall()

        cursor.execute("SELECT Id, FolderId, FileName FROM Files")
        file_rows = cursor.fetchall()

        folders = []
        for folder in folder_rows:
            folder_id = folder[0]
            files = [{"id": f[0], "name": f[2]} for f in file_rows if f[1] == folder_id]
            folders.append({
                "id": folder[0],
                "name": folder[1],
                "parent_id": folder[2],
                "files": files
            })

        cursor.close()
        conn.close()
        return folders
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- File Upload ----------------
@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...), folder_id: int = Form(...)):
    try:
        file_ext = os.path.splitext(file.filename)[1]
        unique_blob_name = f"{uuid.uuid4().hex}{file_ext}"

        blob_client = container_client.get_blob_client(unique_blob_name)
        blob_client.upload_blob(await file.read(), overwrite=True)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Files (FolderId, FileName, FilePath, UploadedAt) VALUES (%s, %s, %s, %s)",
            (folder_id, file.filename, unique_blob_name, datetime.utcnow())
        )
        conn.commit()
        cursor.close()
        conn.close()

        return {"message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- File Download ----------------
@app.get("/download/{file_id}")
def download_file(file_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT FilePath, FileName FROM Files WHERE Id = %s", (file_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="File not found")

        blob_path, filename = row
        blob_client = container_client.get_blob_client(blob_path)
        stream = blob_client.download_blob()
        return StreamingResponse(stream.chunks(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})
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

# ---------------- ECG & HRV Analysis ----------------
def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=512, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_hrv_metrics(rr_intervals):
    rr = np.array(rr_intervals)
    
    if len(rr) < 2:
        return {
            "RMSSD": None,
            "pNN50": None,
            "SD1": None,
            "LF_Power": None,
            "HF_Power": None,
            "DFA_alpha1": None,
        }

    # Time-domain
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    pnn50 = np.sum(np.abs(np.diff(rr)) > 0.05) / (len(rr) - 1) * 100
    sd1 = np.sqrt(np.var(np.diff(rr)) / 2)
    
    # Frequency-domain (approximate)
    rr_detrended = rr - np.mean(rr)
    rr_fft = np.abs(fft(rr_detrended))[:len(rr)//2]
    freqs = np.fft.fftfreq(len(rr), d=np.mean(rr))[:len(rr)//2]
    lf_power = np.sum(rr_fft[(freqs >= 0.04) & (freqs < 0.15)])
    hf_power = np.sum(rr_fft[(freqs >= 0.15) & (freqs < 0.4)])
    
    # Nonlinear
    dfa_alpha1 = np.std(np.log(rr))  # Simple proxy

    return {
        "RMSSD": rmssd,
        "pNN50": pnn50,
        "SD1": sd1,
        "LF_Power": lf_power,
        "HF_Power": hf_power,
        "DFA_alpha1": dfa_alpha1,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), start_index: int = Form(0)):
    try:
        # Read all lines and process manually to handle multiple header rows
        raw = await file.read()
        text = raw.decode("utf-8").replace(",", ".")
        lines = text.splitlines()

        # Remove lines like 'sec;mV' and blank lines
        cleaned_lines = [line for line in lines if not any(x in line.lower() for x in ["sec", "mv"]) and line.strip()]
        if not cleaned_lines or not cleaned_lines[0].lower().startswith("time"):
            raise HTTPException(status_code=400, detail="CSV missing valid headers.")

        df = pd.read_csv(io.StringIO("\n".join(cleaned_lines)), delimiter=";", skip_blank_lines=True)
        df.columns = [c.strip().lower() for c in df.columns]

        # Normalize headers
        if "time (s)" in df.columns and "voltage (mv)" in df.columns:
            df.rename(columns={"time (s)": "time", "voltage (mv)": "voltage"}, inplace=True)
        elif "time" in df.columns and "voltage" in df.columns:
            pass
        else:
            raise HTTPException(status_code=400, detail=f"CSV missing required columns. Found: {df.columns.tolist()}")

        # Drop any non-numeric rows
        df = df[pd.to_numeric(df["time"], errors="coerce").notnull()]
        df = df[pd.to_numeric(df["voltage"], errors="coerce").notnull()]
        df = df.astype(float)

        # Proceed with your existing logic...
        time = df["time"].values
        voltage = df["voltage"].values * 1000  # Convert to µV
        fs = round(1 / np.mean(np.diff(time)))
        filtered = butter_bandpass_filter(voltage, fs=fs)
        filtered = np.clip(filtered, -600, 600)

        threshold = np.mean(filtered) + 1.8 * np.std(filtered)
        min_distance = int(0.3 * fs)
        peaks, _ = find_peaks(filtered, height=threshold, distance=min_distance, prominence=150)

        true_peaks = []
        if len(peaks):
            true_peaks.append(peaks[0])
            for i in range(1, len(peaks)):
                rr = time[peaks[i]] - time[true_peaks[-1]]
                if 0.3 < rr < 1.5:
                    true_peaks.append(peaks[i])

        if start_index >= len(true_peaks):
            raise ValueError(f"Start index {start_index} is out of range.")

        start_time = time[true_peaks[start_index]]
        mask = time >= start_time
        trimmed_time = time[mask] - start_time
        trimmed_voltage = voltage[mask]
        trimmed_filtered = filtered[mask]

        t_peaks, _ = find_peaks(trimmed_filtered,
                                height=np.mean(trimmed_filtered) + 1.8 * np.std(trimmed_filtered),
                                distance=min_distance, prominence=150)

        final_peaks, peak_times, rr_intervals = [], [], []
        window = int(0.03 * fs)

        if len(t_peaks):
            pk_idx = t_peaks[0]
            s, e = max(pk_idx - window, 0), min(pk_idx + window, len(trimmed_voltage))
            true_idx = s + np.argmax(trimmed_voltage[s:e])
            final_peaks.append(true_idx)
            peak_times.append(trimmed_time[true_idx])

            for i in range(1, len(t_peaks)):
                rr = trimmed_time[t_peaks[i]] - trimmed_time[final_peaks[-1]]
                if 0.3 < rr < 1.5:
                    pk_idx = t_peaks[i]
                    s, e = max(pk_idx - window, 0), min(pk_idx + window, len(trimmed_voltage))
                    true_idx = s + np.argmax(trimmed_voltage[s:e])
                    final_peaks.append(true_idx)
                    peak_times.append(trimmed_time[true_idx])
                    rr_intervals.append(rr)

        hrv = calculate_hrv_metrics(rr_intervals)

        # Plot ECG
        buf = io.BytesIO()
        plt.figure(figsize=(12, 5))
        plt.plot(trimmed_time, trimmed_voltage, color='blue')
        plt.scatter([trimmed_time[p] for p in final_peaks], [trimmed_voltage[p] for p in final_peaks], color='red')
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "image": encoded_image,
            "hrvMetrics": hrv,
            "rrTable": [
                {
                    "timestamp": float(peak_times[i]),
                    "rr": None if i == 0 else float(rr_intervals[i - 1])
                }
                for i in range(len(peak_times))
            ],
            "trimmedTime": trimmed_time.tolist(),
            "trimmedVoltage": trimmed_voltage.tolist(),
            "rawVoltage": trimmed_voltage.tolist(),
            "truePeaks": [int(i) for i in final_peaks]
        }

    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Analyze failed: {str(e)}")







@app.post("/trim-and-save")
async def trim_and_save(file: UploadFile = File(...), start_index: int = Form(0), folder_id: int = Form(...)):
    try:
        # Read and parse the uploaded ECG CSV
        df = pd.read_csv(file.file, delimiter=";", decimal=",", skiprows=[1])
        df.columns = ["Time (s)", "Voltage (mV)"]
        df = df.astype(float)

        time = df["Time (s)"].values
        voltage = df["Voltage (mV)"].values * 1000  # Convert to µV
        fs = round(1 / np.mean(np.diff(time)))  # Detect actual sampling rate
        filtered = butter_bandpass_filter(voltage, fs=fs)
        filtered = np.clip(filtered, -600, 600)

        threshold = np.mean(filtered) + 1.8 * np.std(filtered)
        min_distance = int(0.3 * fs)
        peaks, _ = find_peaks(filtered, height=threshold, distance=min_distance, prominence=150)

        true_peaks = []
        if len(peaks):
            true_peaks.append(peaks[0])
            for i in range(1, len(peaks)):
                rr = time[peaks[i]] - time[true_peaks[-1]]
                if 0.3 < rr < 1.5:
                    true_peaks.append(peaks[i])

        if start_index >= len(true_peaks):
            raise ValueError(f"Start index {start_index} is out of range.")

        # Refine starting point to local max in ±30ms window
        start_peak_index = true_peaks[start_index]
        window = int(0.03 * fs)
        s, e = max(start_peak_index - window, 0), min(start_peak_index + window, len(voltage))
        refined_start = s + np.argmax(voltage[s:e])
        start_time = time[refined_start]

        # Trim the signal from refined peak time
        mask = time >= start_time
        trimmed_time = time[mask] - start_time
        trimmed_voltage = voltage[mask]

        # Create CSV content in original format (mV, ; delimiter, , decimal)
        csv_buf = io.StringIO()
        csv_buf.write("Time;Voltage\n")
        for t, v in zip(trimmed_time, trimmed_voltage):
            csv_buf.write(f"{t:.6f};{v/1000:.9f}\n")  # Convert back to mV

        csv_bytes = csv_buf.getvalue().encode("utf-8")

        # Construct filename
        original_name = os.path.splitext(file.filename)[0]
        trimmed_name = f"trimmed_{original_name}.csv"

        # Upload to Azure
        blob_client = container_client.get_blob_client(trimmed_name)
        blob_client.upload_blob(csv_bytes, overwrite=True)

        # Save metadata to DB
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Files (FolderId, FileName, FilePath, UploadedAt) VALUES (%s, %s, %s, %s)",
            (folder_id, trimmed_name, trimmed_name, datetime.utcnow())
        )
        conn.commit()
        cursor.close()
        conn.close()

        return {"message": "Trimmed file saved successfully", "filename": trimmed_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/files/{file_id}")
async def delete_file(file_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT FilePath FROM Files WHERE Id = %s", (file_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path = row[0]

        # Delete from Azure Blob
        blob_client = container_client.get_blob_client(file_path)
        blob_client.delete_blob()

        # Delete from DB
        cursor.execute("DELETE FROM Files WHERE Id = %s", (file_id,))
        conn.commit()
        cursor.close()
        conn.close()

        return {"message": "File deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/files/{file_id}/rename")
async def rename_file(file_id: int, new_name: str = Form(...)):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT FilePath FROM Files WHERE Id = %s", (file_id,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="File not found")

        old_blob_path = row[0]
        file_ext = os.path.splitext(old_blob_path)[1]
        new_blob_path = f"{uuid.uuid4().hex}{file_ext}"

        # Copy blob to new name
        old_blob = container_client.get_blob_client(old_blob_path)
        new_blob = container_client.get_blob_client(new_blob_path)
        new_blob.start_copy_from_url(old_blob.url)
        old_blob.delete_blob()

        # Update DB
        cursor.execute(
            "UPDATE Files SET FileName = %s, FilePath = %s WHERE Id = %s",
            (new_name, new_blob_path, file_id)
        )
        conn.commit()
        cursor.close()
        conn.close()

        return {"message": "File renamed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/folders/{folder_id}")
async def rename_folder(folder_id: int, new_name: str = Form(...)):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE Folders SET Name = %s WHERE Id = %s", (new_name, folder_id))
        conn.commit()
        cursor.close()
        conn.close()
        return {"message": "Folder renamed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/folders/{folder_id}")
async def delete_folder(folder_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if folder exists
        cursor.execute("SELECT COUNT(*) FROM Folders WHERE Id = %s", (folder_id,))
        if cursor.fetchone()[0] == 0:
            raise HTTPException(status_code=404, detail="Folder not found")

        # Optional: Check and delete subfolders and files recursively if needed
        # For now, we'll assume the frontend prevents deletion of non-empty folders

        # Delete folder
        cursor.execute("DELETE FROM Folders WHERE Id = %s", (folder_id,))
        conn.commit()
        cursor.close()
        conn.close()

        return {"message": "Folder deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MergeRequest(BaseModel):
    folder_id: int
    file_ids: List[int]

@app.post("/merge-files")
async def merge_files(data: MergeRequest):
    try:
        folder_id = data.folder_id
        file_ids = data.file_ids

        # Fetch folder name
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT Name FROM Folders WHERE Id = %s", (folder_id,))
        folder_row = cursor.fetchone()
        if not folder_row:
            raise HTTPException(status_code=404, detail="Folder not found")
        folder_name = folder_row[0].replace(" ", "_").lower()

        merged_df = None
        total_time_offset = 0.0

        for file_id in file_ids:
            cursor.execute("SELECT FilePath FROM Files WHERE Id = %s AND FolderId = %s", (file_id, folder_id))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"File ID {file_id} not found in folder {folder_id}")
            blob_path = row[0]
            blob_client = container_client.get_blob_client(blob_path)

            try:
                content = blob_client.download_blob().readall()
                decoded_text = content.decode("utf-8").replace(",", ".")
                df = pd.read_csv(io.StringIO(decoded_text), delimiter=";", skiprows=1)
                df.columns = ["Time", "Voltage"]
                df = df.astype(float)
            except Exception as err:
                raise HTTPException(status_code=500, detail=f"Error reading or parsing {blob_path}: {err}")

            if merged_df is None:
                merged_df = df
                total_time_offset = df["Time"].iloc[-1] + (df["Time"].iloc[1] - df["Time"].iloc[0])
            else:
                df["Time"] += total_time_offset
                total_time_offset = df["Time"].iloc[-1] + (df["Time"].iloc[1] - df["Time"].iloc[0])
                merged_df = pd.concat([merged_df, df], ignore_index=True)

        # Write merged CSV
        output_csv = io.StringIO()
        output_csv.write("Time;Voltage\n")
        for _, row in merged_df.iterrows():
            output_csv.write(f"{row['Time']:.6f};{row['Voltage']:.9f}\n")
        output_bytes = output_csv.getvalue().encode("utf-8")

        # Save to Azure
        merged_filename = f"merged_{folder_name}.csv"
        blob_client = container_client.get_blob_client(merged_filename)
        blob_client.upload_blob(output_bytes, overwrite=True)

        # Save metadata to DB
        cursor.execute(
            "INSERT INTO Files (FolderId, FileName, FilePath, UploadedAt) VALUES (%s, %s, %s, %s)",
            (folder_id, merged_filename, merged_filename, datetime.utcnow())
        )
        conn.commit()
        cursor.close()
        conn.close()

        return {
            "message": "Merged file created and saved successfully.",
            "filename": merged_filename
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")
