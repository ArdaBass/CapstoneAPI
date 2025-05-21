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


def normalize_folder_name(name):
    tr_map = str.maketrans({
        'ç': 'c', 'Ç': 'C',
        'ğ': 'g', 'Ğ': 'G',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S',
        'ü': 'u', 'Ü': 'U',
    })
    return name.translate(tr_map).replace(" ", "_")



@app.post("/import-participants")
async def import_participants(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")
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

                # Insert participant
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

                # 🔠 Normalize folder name
                normalized_name = normalize_folder_name(name)

                # 👤 Check or create main folder
                cursor.execute("SELECT Id FROM Folders WHERE Name = %s AND ParentId IS NULL", (normalized_name,))
                result = cursor.fetchone()
                if result:
                    parent_id = result[0]
                else:
                    cursor.execute("INSERT INTO Folders (Name, ParentId) VALUES (%s, NULL)", (normalized_name,))
                    parent_id = cursor.lastrowid

                # 📁 Create Biopac, Watch, ML subfolders
                for subfolder in ["Biopac", "Watch", "ML"]:
                    cursor.execute(
                        "SELECT COUNT(*) FROM Folders WHERE Name = %s AND ParentId = %s",
                        (subfolder, parent_id)
                    )
                    if cursor.fetchone()[0] == 0:
                        cursor.execute("INSERT INTO Folders (Name, ParentId) VALUES (%s, %s)", (subfolder, parent_id))

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
        raw = await file.read()
        text = raw.decode("utf-8").replace(",", ".")
        lines = text.splitlines()

        cleaned_lines = [line for line in lines if not any(x in line.lower() for x in ["sec", "mv"]) and line.strip()]
        if not cleaned_lines or not cleaned_lines[0].lower().startswith("time"):
            raise HTTPException(status_code=400, detail="CSV missing valid headers.")

        df = pd.read_csv(io.StringIO("\n".join(cleaned_lines)), delimiter=";", skip_blank_lines=True)
        df.columns = [c.strip().lower() for c in df.columns]

        if "time" in df.columns and "voltage" in df.columns:
            pass
        else:
            raise HTTPException(status_code=400, detail=f"CSV missing required columns. Found: {df.columns.tolist()}")

        df = df[pd.to_numeric(df["time"], errors="coerce").notnull()]
        df = df[pd.to_numeric(df["voltage"], errors="coerce").notnull()]
        df = df.astype(float)

        time = df["time"].values
        voltage = df["voltage"].values
        fs = round(1 / np.mean(np.diff(time)))

        from scipy.signal import savgol_filter

        voltage_uv = voltage * 1000
        filtered = butter_bandpass_filter(voltage_uv, fs=fs)
        smoothed = savgol_filter(filtered, window_length=19, polyorder=2)
        smoothed = np.clip(smoothed, -600, 600)
        derivative = np.gradient(smoothed)
        sharpness = derivative ** 2

        min_distance = int(0.3 * fs)
        peak_candidates, _ = find_peaks(sharpness, distance=min_distance, prominence=np.std(sharpness))

        final_peaks, peak_times, rr_intervals = [], [], []
        search_window = int(0.03 * fs)

        for idx in peak_candidates:
            s = max(0, idx - search_window)
            e = min(len(smoothed), idx + search_window)
            true_idx_smoothed = s + np.argmax(smoothed[s:e])
            true_idx_voltage = s + np.argmax(voltage_uv[s:e])
            true_idx = true_idx_voltage if voltage_uv[true_idx_voltage] >= voltage_uv[true_idx_smoothed] else true_idx_smoothed

            check_window = int(0.04 * fs)
            ws = max(0, true_idx - check_window)
            we = min(len(voltage_uv), true_idx + check_window)
            if voltage_uv[true_idx] < max(voltage_uv[ws:we]):
                continue


            if final_peaks and (time[true_idx] - time[final_peaks[-1]]) < 0.3:
                continue

            final_peaks.append(true_idx)
            peak_times.append(time[true_idx])
            if len(peak_times) > 1:
                rr_intervals.append(peak_times[-1] - peak_times[-2])

        if start_index >= len(final_peaks):
            raise ValueError(f"Start index {start_index} is out of range. Found {len(final_peaks)} peaks.")

        start_time = time[final_peaks[start_index]]
        mask = time >= start_time
        trimmed_time = time[mask] - start_time
        trimmed_voltage = voltage_uv[mask]

        trimmed_peak_indices = [i for i in final_peaks if time[i] >= start_time]
        trimmed_peak_indices = [i - np.where(mask)[0][0] for i in trimmed_peak_indices]

        hrv = calculate_hrv_metrics(rr_intervals)

        # Plot
        buf = io.BytesIO()
        plt.figure(figsize=(12, 5))
        plt.plot(trimmed_time, trimmed_voltage, color='blue')
        plt.scatter([trimmed_time[i] for i in trimmed_peak_indices], [trimmed_voltage[i] for i in trimmed_peak_indices], color='red')
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
            "truePeaks": [int(i) for i in trimmed_peak_indices]
        }

    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Analyze failed: {str(e)}")





@app.post("/trim-and-save")
async def trim_and_save(file: UploadFile = File(...), start_index: int = Form(0), folder_id: int = Form(...)):
    try:
        # 📥 Read file and handle optional unit line
        content = await file.read()
        text = content.decode("utf-8").replace(",", ".")
        lines = text.strip().splitlines()

        # Skip second row if it contains units
        if len(lines) > 1 and ("sec" in lines[1].lower() or "mv" in lines[1].lower()):
            data_lines = lines[2:]
        else:
            data_lines = lines[1:]

        csv_text = "Time;Voltage\n" + "\n".join(data_lines)
        df = pd.read_csv(io.StringIO(csv_text), delimiter=";", skip_blank_lines=True)
        df.columns = [c.strip().lower() for c in df.columns]

        df = df[pd.to_numeric(df["time"], errors="coerce").notnull()]
        df = df[pd.to_numeric(df["voltage"], errors="coerce").notnull()]
        df = df.astype(float)

        time = df["time"].values
        voltage = df["voltage"].values
        fs = round(1 / np.mean(np.diff(time)))

        from scipy.signal import savgol_filter

        # Match analyze logic exactly
        voltage_uv = voltage * 1000
        filtered = butter_bandpass_filter(voltage_uv, fs=fs)
        smoothed = savgol_filter(filtered, window_length=19, polyorder=2)
        smoothed = np.clip(smoothed, -600, 600)
        derivative = np.gradient(smoothed)
        sharpness = derivative ** 2

        min_distance = int(0.3 * fs)
        peak_candidates, _ = find_peaks(sharpness, distance=min_distance, prominence=np.std(sharpness))

        final_peaks = []
        search_window = int(0.03 * fs)

        for idx in peak_candidates:
            s = max(0, idx - search_window)
            e = min(len(smoothed), idx + search_window)
            true_idx_smoothed = s + np.argmax(smoothed[s:e])
            true_idx_voltage = s + np.argmax(voltage_uv[s:e])
            true_idx = true_idx_voltage if voltage_uv[true_idx_voltage] >= voltage_uv[true_idx_smoothed] else true_idx_smoothed

            check_window = int(0.04 * fs)
            ws = max(0, true_idx - check_window)
            we = min(len(voltage_uv), true_idx + check_window)
            if voltage_uv[true_idx] < max(voltage_uv[ws:we]):
                continue

            if final_peaks and (time[true_idx] - time[final_peaks[-1]]) < 0.3:
                continue

            final_peaks.append(true_idx)

        if start_index >= len(final_peaks):
            raise ValueError(f"Start index {start_index} is out of range. Found only {len(final_peaks)} peaks.")

        # 🕒 Trim from selected peak
        start_time = time[final_peaks[start_index]]
        mask = time >= start_time
        trimmed_time = time[mask] - start_time
        trimmed_voltage = voltage_uv[mask]

        # 💾 Save as CSV in mV
        csv_buf = io.StringIO()
        csv_buf.write("Time;Voltage\n")
        for t, v in zip(trimmed_time, trimmed_voltage):
            csv_buf.write(f"{t:.6f};{v/1000:.9f}\n")

        csv_bytes = csv_buf.getvalue().encode("utf-8")
        original_name = os.path.splitext(file.filename)[0]
        trimmed_name = f"trimmed_{original_name}.csv"

        # ☁️ Upload to Azure
        blob_client = container_client.get_blob_client(trimmed_name)
        blob_client.upload_blob(csv_bytes, overwrite=True)

        # 🗂️ Save metadata
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
        print("TRACEBACK:\n", traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Trim and save failed: {e}")




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
        folder_name = normalize_folder_name(folder_row[0]).lower()


        merged_df = None
        last_time = 0.0
        last_voltage = 0.0
        sampling_interval = None

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

            if df.empty:
                continue

            # Calculate sampling interval only once
            if sampling_interval is None and len(df) > 1:
                sampling_interval = df["Time"].iloc[1] - df["Time"].iloc[0]

            if merged_df is None:
                merged_df = df
            else:
                # Align both time and voltage to previous file's end
                time_shift = last_time - df["Time"].iloc[0]
                voltage_shift = last_voltage - df["Voltage"].iloc[0]
                df["Time"] += time_shift
                df["Voltage"] += voltage_shift
                merged_df = pd.concat([merged_df, df], ignore_index=True)

            # Update last time and voltage for next file alignment
            last_time = merged_df["Time"].iloc[-1] + (sampling_interval if sampling_interval else 0.001)
            last_voltage = merged_df["Voltage"].iloc[-1]

        if merged_df is None or merged_df.empty:
            raise HTTPException(status_code=400, detail="No valid files to merge.")

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
