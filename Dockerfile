# 1. Gunakan Python sebagai base image
FROM python:3.10-slim

# 2. Set direktori kerja di dalam container
WORKDIR /app

# 3. Copy file requirements.txt ke dalam container
COPY requirements.txt .

# 4. Install dependencies
RUN pip install -r requirements.txt

# 5. Copy semua file project ke dalam container
COPY . .

# 6. Expose port 8000 (untuk FastAPI)
EXPOSE 8000

# 7. Jalankan server FastAPI menggunakan Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
