# 🐳 Super Beginner Guide: Install Docker & Run n8n (No Coding Needed)

This guide is designed for **non-technical students**. We will use **buttons, clicks, and simple steps** instead of complex commands.

👉 Goal: Install n8n in a way that:
- Your data is saved in a **real folder on your computer**
- You can **start/stop with one click**
- You don’t have to type commands every time

---

# 🎯 Final Result (What You’ll Get)

✔ n8n installed on your computer  
✔ Data saved safely in a folder  
✔ Start/Stop using Docker Desktop (GUI)  
✔ Opens in browser anytime  

---

# ⚙️ Step 1 — Install Docker Desktop (One-Time Setup)

### 👉 Download
Go here:
https://www.docker.com/products/docker-desktop/

Click:
```
Download for Windows
```

---

### 👉 Install

1. Double click:
```
Docker Desktop Installer.exe
```
2. Click:
```
Install
```
3. Restart computer

---

### 👉 Open Docker

After restart:

- Press **Start Menu**
- Search: `Docker Desktop`

You should see:
```
Docker Desktop is running
```

✔ Also check bottom right corner:
```
🐳 icon visible
```

---

# 📁 Step 2 — Create a Folder for Your Data (IMPORTANT)

We will store your n8n data here so nothing is lost.

### 👉 Do this:

1. Open File Explorer
2. Go to:
```
Documents
```
3. Create a new folder:
```
n8n-data
```

Final path will look like:
```
C:\Users\YourName\Documents\n8n-data
```

✔ This is where all your workflows will be saved safely.

---

# 🧩 Step 3 — Use Docker Desktop (No Terminal Method)

Now we will **avoid coding** and use buttons.

---

## 👉 Open Docker Desktop

You will see sections like:

- Containers
- Images
- Volumes

---

## 👉 Pull n8n Image

1. Click **Images** tab
2. Click **Search** bar
3. Type:
```
n8nio/n8n
```
4. Click **Pull**

---

### ✅ Sample Output

You will see:
```
Status: Downloaded newer image for n8nio/n8n
```

---

# 🚀 Step 4 — Run n8n (Using GUI)

Now the important part 👇

---

## 👉 Click Run

1. Go to **Images tab**
2. Find:
```
n8nio/n8n
```
3. Click:
```
Run
```

---

## 👉 Fill Settings (VERY IMPORTANT)

A settings window will open.

### 🧠 Set these values:

### 1. Container Name
```
n8n
```

---

### 2. Ports

Set:
```
Host Port: 5678
Container Port: 5678
```

---

### 3. Volume (THIS SAVES YOUR DATA)

Click:
```
Add Folder
```

Select:
```
C:\Users\YourName\Documents\n8n-data
```

Set mount path:
```
/home/node/.n8n
```

---

### 4. Restart Policy (IMPORTANT)

Set:
```
Always
```

👉 This ensures:
- n8n auto starts when PC starts
- No need to run again manually

---

### 5. Run Container

Click:
```
Run
```

---

# ✅ Step 5 — Check if n8n is Running

Go to:

👉 Docker Desktop → Containers

You should see:
```
n8n (running)
```

---

### ✅ Sample Output (Logs)

Click container → Logs:

```
n8n ready on 0.0.0.0, port 5678
```

---

# 🌐 Step 6 — Open n8n in Browser

Open:
```
http://localhost:5678
```

---

### First Screen

```
Create Owner Account
Email
Password
```

After login:
```
Welcome to n8n
```

🎉 DONE!

---

# 🔁 Step 7 — How to Start Next Time (Very Easy)

No commands needed.

### 👉 Just do this:

1. Open **Docker Desktop**
2. Go to **Containers**
3. Click:
```
Start ▶
```

---

# 🛑 Step 8 — Stop n8n

In Docker Desktop:

Click:
```
Stop ⏹
```

---

# 💾 Where Your Data is Stored

All workflows are saved here:

```
Documents → n8n-data
```

Even if:
- Docker crashes ❌
- PC restarts ❌

👉 Your data is SAFE ✅

---

# ❗ Common Beginner Issues

---

## ❌ n8n not opening

Check:
- Docker is running
- Container is running

---

## ❌ Port already in use

Change port to:
```
8080
```

Open:
```
http://localhost:8080
```

---

## ❌ Forgot password

Delete contents of:
```
n8n-data folder
```

Then restart container.

---

# 🎉 You’re Done!

You now have:

✔ One-click start system  
✔ Safe local storage  
✔ Beginner-friendly setup  
✔ No coding required  

---

# 💡 Bonus Tip (For Students)

👉 Always:
- Start Docker first
- Then start container
- Then open browser

That’s it 👍

---

# ❤️ End of Guide

