# 🐳 DocuMind AI — Enterprise Docker & GPU Deployment Guide

This guide provides step-by-step instructions for containerizing, orchestrating, and deploying the **DocuMind AI Document Intelligence System** using Docker, Docker Compose, NVIDIA GPU acceleration, and cloud infrastructure.

---

## 📋 1. Prerequisites

Before starting, ensure your host environment meets the following requirements:

* **Docker Engine**: Version `20.10.0` or higher.
* **Docker Compose**: Version `2.0.0` or higher (`docker compose` CLI plugin).
* **NVIDIA GPU Acceleration (Optional but Recommended)**:
  * NVIDIA GPU Driver installed on host.
  * `nvidia-container-toolkit` installed on host (for GPU passthrough to Docker containers).

---

## 📁 2. Directory & Volume Mapping Architecture

The containerized system relies on four host-mounted persistent storage directories to ensure data durability across container restarts:

| Host Path | Container Path | Description |
| :--- | :--- | :--- |
| `./data` | `/app/data` | Raw uploaded files (PDFs, TXT, DOCX, Images, audio, etc.) |
| `./logs` | `/app/logs` | SQLite databases (`chat_history.db`), auth credentials, and evaluation metrics |
| `./faiss_index` | `/app/faiss_index` | FAISS vector store index & chunk embeddings persistence |
| `./bm25_index` | `/app/bm25_index` | BM25 keyword index token state persistence |

---

## 🚀 3. Quickstart Local Deployment

### Step 1: Environment Configuration
Copy the template configuration file to `.env` and set your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your **Groq API Key**:
```ini
GROQ_API_KEY=gsk_your_real_groq_api_key_here
JWT_SECRET_KEY=your_secure_random_jwt_secret_key_2026
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CONFIDENCE_THRESHOLD=0.15
NVIDIA_VISIBLE_DEVICES=all
```

### Step 2: Build and Launch Container Service
Run Docker Compose in detached mode:

```bash
docker compose up -d --build
```

### Step 3: Access Application
Open your web browser and navigate to:
👉 **[http://localhost:8501](http://localhost:8501)**

Default Login Credentials for testing:
* **Admin Account**: `username: admin` | `password: admin123`
* **User Account**: `username: user` | `password: user123`

---

## ⚡ 4. NVIDIA GPU Acceleration Setup

To enable GPU acceleration for PyTorch embedding generation and CrossEncoder re-ranking:

### Step 1: Install NVIDIA Container Toolkit (Ubuntu / Debian)
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb [^ ]* #deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] #g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Step 2: Verify GPU Passthrough in Docker
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If `nvidia-smi` displays your GPU hardware specs, `docker-compose up` will automatically utilize GPU acceleration via the `deploy.resources.reservations.devices` block in `docker-compose.yml`.

---

## 🌐 5. Production Cloud Deployment

### Option A: AWS EC2 / AWS ECS (GPU Instance)
1. Launch an AWS EC2 instance (e.g. `g4dn.xlarge` with NVIDIA T4 GPU or `t3.xlarge` CPU-only) running Ubuntu 22.04.
2. Install Docker, Docker Compose, and NVIDIA Container Toolkit.
3. Clone repository and run:
   ```bash
   docker compose up -d --build
   ```
4. Configure Security Group to allow inbound traffic on TCP port `8501` (or set up an Nginx reverse proxy with SSL/HTTPS on port `443`).

### Option B: Nginx Reverse Proxy with SSL (HTTPS)
For production enterprise setups, route domain traffic through Nginx with Let's Encrypt SSL:

```nginx
server {
    listen 80;
    server_name documind.yourcompany.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name documind.yourcompany.com;

    ssl_certificate /etc/letsencrypt/live/documind.yourcompany.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/documind.yourcompany.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 🛠️ 6. Useful Container Management Commands

| Action | Command |
| :--- | :--- |
| **View Live Logs** | `docker compose logs -f documind-app` |
| **Check Container Health** | `docker inspect --format='{{json .State.Health}}' documind-container` |
| **Restart Application** | `docker compose restart` |
| **Stop Service** | `docker compose down` |
| **Rebuild Without Cache** | `docker compose build --no-cache` |
| **Shell Access inside Container** | `docker exec -it documind-container /bin/bash` |
