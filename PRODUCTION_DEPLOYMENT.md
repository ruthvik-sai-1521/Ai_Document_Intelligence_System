# 🚀 DocuMind AI — Production Deployment & Cloud Handbook

This handbook provides comprehensive instructions for deploying the **DocuMind AI Enterprise Document Intelligence System** into production across Linux servers, AWS, Azure, GCP, Render, Railway, Nginx SSL reverse proxies, CI/CD automation, monitoring, and logging.

---

## 📑 Table of Contents

1. [Linux Host Hardening & System Setup](#1-linux-host-hardening--system-setup)
2. [Cloud Platform Deployments](#2-cloud-platform-deployments)
   - [AWS (EC2 GPU / ECS / Route53)](#aws-ec2-gpu--ecs--route53)
   - [Google Cloud Platform (GCP Compute Engine / Cloud Run)](#google-cloud-platform-gcp)
   - [Microsoft Azure (Azure VM / Container Apps)](#microsoft-azure)
   - [Render (PaaS)](#render-paas)
   - [Railway (PaaS)](#railway-paas)
3. [Nginx Reverse Proxy & SSL (HTTPS) Setup](#3-nginx-reverse-proxy--ssl-https-setup)
4. [Monitoring & Centralized Logging](#4-monitoring--centralized-logging)
5. [CI/CD Automated Deployment (GitHub Actions)](#5-cicd-automated-deployment-github-actions)
6. [Environment Variables Reference](#6-environment-variables-reference)

---

## 🐧 1. Linux Host Hardening & System Setup

For bare-metal or Cloud Linux instances (Ubuntu 22.04 LTS / Debian 12 / RHEL 9):

### System Preparation & Firewall Rules
```bash
# Update OS packages
sudo apt-get update && sudo apt-get upgrade -y

# Configure Uncomplicated Firewall (UFW)
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### Install Docker, Docker Compose & NVIDIA Drivers
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit (for GPU passthrough)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb [^ ]* #deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] #g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## ☁️ 2. Cloud Platform Deployments

### AWS (EC2 GPU / ECS / Route53)
1. **Instance Provisioning**: Launch an EC2 `g4dn.xlarge` instance (1 NVIDIA T4 GPU, 4 vCPUs, 16GB RAM) running Ubuntu 22.04 LTS.
2. **EBS Volume**: Attach a 50GB General Purpose SSD (`gp3`) for document and FAISS vector index storage.
3. **Security Group**:
   - Inbound: Port 22 (SSH), Port 80 (HTTP), Port 443 (HTTPS).
4. **Deploy Application**:
   ```bash
   git clone https://github.com/your-org/Ai_Document_Intelligence_System.git
   cd Ai_Document_Intelligence_System
   cp .env.example .env
   # Set GROQ_API_KEY and JWT_SECRET_KEY in .env
   docker compose -f docker-compose.prod.yml up -d --build
   ```

### Google Cloud Platform (GCP)
1. **Compute Engine**: Provision a `n1-standard-4` instance with 1 NVIDIA T4 GPU.
2. **Cloud DNS**: Point your domain `documind.yourcompany.com` to the GCP external static IP address.
3. **Launch Container Stack**: Use `docker-compose.prod.yml` to start the app and Nginx SSL proxy.

### Microsoft Azure
1. **Virtual Machine**: Provision an Azure `NC6s_v3` GPU instance running Ubuntu 22.04.
2. **Network Security Group**: Open inbound HTTP (80) and HTTPS (443) rules.
3. **Launch Application**: `docker compose -f docker-compose.prod.yml up -d --build`.

### Render (PaaS)
1. Connect your GitHub repository to [Render](https://render.com).
2. Render will automatically detect the [render.yaml](file:///d:/Projects/Ai_Document_Intelligence_System/render.yaml) blueprint.
3. Configure secret environment variables (`GROQ_API_KEY`, `JWT_SECRET_KEY`) in the Render Dashboard.
4. Render automatically manages SSL certificates for custom domains!

### Railway (PaaS)
1. Connect your repository to [Railway](https://railway.app).
2. Railway auto-detects the [railway.json](file:///d:/Projects/Ai_Document_Intelligence_System/railway.json) manifest and builds using the `Dockerfile`.
3. Add environment variables under **Variables** (`GROQ_API_KEY`, `JWT_SECRET_KEY`).
4. Generate a public domain under **Settings -> Networking**.

---

## 🔒 3. Nginx Reverse Proxy & SSL (HTTPS) Setup

To issue free SSL certificates via Let's Encrypt / Certbot:

### Step 1: Initial Certificate Request
Run Certbot once to obtain initial certificates for your domain:
```bash
docker run --rm -v $(pwd)/certbot/conf:/etc/letsencrypt -v $(pwd)/certbot/www:/var/www/certbot \
  certbot/certbot certonly --webroot -w /var/www/certbot \
  -d documind.yourcompany.com --email admin@yourcompany.com --agree-tos --no-eff-email
```

### Step 2: Start Production Stack with Auto-Renewal
```bash
docker compose -f docker-compose.prod.yml up -d
```
The included `certbot` container automatically checks and renews certificates every 12 hours!

---

## 📊 4. Monitoring & Centralized Logging

### Healthcheck Monitoring
- The application exposes a health endpoint: `http://localhost:8501/_stcore/health`
- Probed by Docker Compose every 30 seconds.

### Structured Logging
All system logs are written to `/app/logs/chat_history.db` and `/app/logs/production.log`.

To inspect live application logs:
```bash
docker compose -f docker-compose.prod.yml logs -f documind-app
```

---

## 🔄 5. CI/CD Automated Deployment (GitHub Actions)

The repository includes a GitHub Actions workflow [.github/workflows/ci-cd.yml](file:///d:/Projects/Ai_Document_Intelligence_System/.github/workflows/ci-cd.yml) that:
1. Executes automated syntax and model compilation tests.
2. Runs authentication and evaluation benchmark tests.
3. Builds the Docker image.
4. Triggers PaaS deployment hooks.

### Required GitHub Secrets:
Add the following secrets in **GitHub Repo Settings -> Secrets and variables -> Actions**:
- `GROQ_API_KEY`: Groq API key for benchmark tests.
- `RENDER_DEPLOY_HOOK`: Render automatic deploy URL (optional).

---

## ⚙️ 6. Environment Variables Reference

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | *Required* | API key for LLM answer generation and query rewriting |
| `JWT_SECRET_KEY` | *Required* | Secret key for signing JWT authentication tokens |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | HuggingFace Sentence-Transformers model name |
| `CONFIDENCE_THRESHOLD` | `0.15` | Minimum cosine similarity required to answer queries |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU passthrough specification for CUDA |
