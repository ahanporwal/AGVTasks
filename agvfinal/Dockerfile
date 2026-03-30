FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libxrender1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .

# Install normal deps first
RUN pip install --no-cache-dir -r requirements.txt

# Install pybullet with custom flags
RUN CFLAGS="-fno-define-target-os-macros" pip install pybullet

COPY . .

CMD ["python", "t1s1/main.py"]