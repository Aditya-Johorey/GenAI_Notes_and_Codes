# 🐳 Beginner Friendly Guide: Install Docker and Run n8n Locally

This guide will help you install **Docker** and run **n8n automation platform** on your computer.

No advanced technical knowledge required — just follow step by step 🙂

---

# 📌 What You Will Achieve

After completing this guide, you will be able to:

✅ Install Docker on your system
✅ Verify Docker is working
✅ Run n8n using Docker
✅ Open n8n in your browser
✅ Start creating workflows

---

# 📦 What is Docker?

Docker is a tool that lets you run applications in **containers**.

Think of containers as:
👉 Small packaged environments that already contain everything needed to run software.

This means:
✔ No complicated setup
✔ No dependency issues
✔ Works the same on every computer

---

# ⚙️ Step 1 — Install Docker

Choose your operating system below.

---

## 🪟 Windows Installation

### Step 1 — Download Docker Desktop

Go to:

👉 https://www.docker.com/products/docker-desktop/

Click:

```
Download for Windows
```

---

### Step 2 — Run Installer

Double-click downloaded file:

```
Docker Desktop Installer.exe
```

Click:

```
OK → Install → Restart computer
```

---

### Step 3 — Start Docker Desktop

After restart:

Open Start Menu → Search:

```
Docker Desktop
```

When Docker starts successfully, you will see:

```
Docker Desktop is running
```

---

### ✅ Sample Output (System Tray)

You should see the Docker whale icon:

```
🐳 Docker Desktop running
```

---

## 🍎 macOS Installation

### Step 1 — Download

https://www.docker.com/products/docker-desktop/

Choose:

```
Mac with Apple Chip
OR
Mac with Intel Chip
```

---

### Step 2 — Install

Open downloaded `.dmg`

Drag Docker into Applications folder.

---

### Step 3 — Start Docker

Open Applications → Docker

You will see:

```
Docker Desktop is running
```

---

## 🐧 Linux (Ubuntu Recommended)

Run in terminal:

```bash
sudo apt update
sudo apt install docker.io -y
```

Start Docker:

```bash
sudo systemctl start docker
```

Enable auto start:

```bash
sudo systemctl enable docker
```

---

### ✅ Sample Output

```bash
Created symlink /etc/systemd/system/multi-user.target.wants/docker.service
```

---

# 🧪 Step 2 — Verify Docker Installation

Open terminal / command prompt and run:

```bash
docker --version
```

---

### ✅ Expected Output

Example:

```
Docker version 25.0.3, build 4debf41
```

If you see version → Docker is installed successfully ✅

---

# 🧪 Step 3 — Test Docker With Hello World

Run:

```bash
docker run hello-world
```

---

### ✅ Sample Output

You will see something like:

```
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

🎉 Docker is working!

---

# 🚀 Step 4 — Run n8n Using Docker

Now we will start n8n.

---

## ▶ Run This Command

```bash
docker run -it --rm \
-p 5678:5678 \
n8nio/n8n
```

---

### 💡 What This Means

| Part           | Meaning               |
| -------------- | --------------------- |
| `docker run`   | start container       |
| `-p 5678:5678` | open port for browser |
| `n8nio/n8n`    | n8n software          |

---

### ✅ Sample Output

First time you run this, Docker downloads n8n:

```
Unable to find image 'n8nio/n8n:latest' locally
Downloading...
```

Then:

```
n8n ready on 0.0.0.0, port 5678
```

---

# 🌐 Step 5 — Open n8n in Browser

Open browser and go to:

```
http://localhost:5678
```

---

### ✅ What You Will See

First time setup screen:

```
Create Owner Account
Email
Password
```

After login:

```
Welcome to n8n workflow editor
```

🎉 SUCCESS — n8n is running!

---

# 🧰 Step 6 — Stop n8n

In terminal press:

```
CTRL + C
```

Container stops.

---

# 💾 Step 7 — Run n8n With Data Saved (Recommended)

Without this, workflows disappear after restart.

Use this command:

```bash
docker run -it --rm \
-p 5678:5678 \
-v n8n_data:/home/node/.n8n \
n8nio/n8n
```

---

### ✅ Sample Output

```
Using existing volume n8n_data
n8n ready on port 5678
```

Now your workflows are permanent.

---

# ⭐ Step 8 — Run n8n in Background (Advanced but Useful)

```bash
docker run -d \
--name n8n \
-p 5678:5678 \
-v n8n_data:/home/node/.n8n \
n8nio/n8n
```

---

### ✅ Output

```
f4a9c1a9b0a8d21e...
```

That long code = container ID.

---

## Check running containers

```bash
docker ps
```

Example:

```
CONTAINER ID   IMAGE       PORTS
f4a9c1a9b0a8   n8nio/n8n   0.0.0.0:5678->5678/tcp
```

---

# 🛑 Stop Background Container

```bash
docker stop n8n
```

---

# ❗ Common Problems & Fixes

---

## Docker not starting

Restart computer.

---

## Port already in use

Change port:

```bash
-p 8080:5678
```

Open:

```
http://localhost:8080
```

---

## Permission denied (Linux)

Run:

```bash
sudo docker run hello-world
```

Or add user to docker group.

---

# 🎉 Congratulations!

You now know how to:

✔ Install Docker
✔ Test Docker
✔ Run n8n
✔ Access workflow editor
✔ Save workflows

You are ready to automate anything 🚀

---

# ❤️ End of Guide
