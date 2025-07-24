# 🧠 CapstoneHRV Backend API

This is the backend service for **CapstoneHRV**, a heart rate variability (HRV) analysis platform. Built with **FastAPI**, it handles:
- File upload & storage on **Azure Blob**
- Folder and file management
- Participant import
- ECG trimming & peak detection
- HRV metric calculation
- CSV merging for longitudinal ECG analysis

---

## 🚀 Features

### ✅ ECG/HRV Analysis
- Filters raw ECG signals
- Detects peaks and calculates RR intervals
- Computes HRV metrics (RMSSD, SDNN, pNN50, LF/HF, etc.)
- Visualizes ECG with peak annotations (base64 PNG)

### 📁 File & Folder Management
- Create, rename, delete folders
- Upload, download, rename, move, delete CSV files
- Merge ECG CSVs across files with voltage/time alignment

### 👤 Participant Import
- Bulk import participants from Excel
- Automatically create per-user folders with `Biopac`, `Watch`, and `ML` subfolders

### ☁️ Cloud Integration
- Azure Blob Storage for scalable file handling
- Microsoft SQL Server via `pymssql` for metadata and participant storage

---

## 🧱 Tech Stack

| Layer           | Tech                             |
|----------------|----------------------------------|
| Backend        | FastAPI                          |
| File Storage   | Azure Blob Storage               |
| Database       | Microsoft SQL Server (via pymssql) |
| Data Analysis  | Pandas, NumPy, SciPy, Matplotlib |
| Deployment     | Render / Docker (optional)       |

---

## ⚙️ Requirements

- Python 3.9+
- Azure Blob Storage account
- Azure SQL Server connection string
- Dependencies listed in `requirements.txt`

---

## 🛠 Setup

```bash
# Clone the repository
git clone https://github.com/ArdaBass/CapstoneHRV-Backend.git
cd CapstoneHRV-Backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn main:app --reload
