# How to Fix Slow Gemini & LLM Node Performance in n8n

If your n8n workflows feel incredibly slow, freeze, or take a long time to load elements when working with Google Gemini or other AI nodes, it is usually because n8n is trying to run invisible background tracking "telemetry" scripts. 

When n8n is running inside a private network, behind a firewall, or on a local machine, these tracking scripts get stuck in an endless loop trying to connect to the internet, creating massive lag.

This guide will help you safely remove that tracking loop in less than 5 minutes. **Your existing workflows, credentials, and data will remain 100% safe.**

---

## 🛠️ Step-by-Step Fix

Follow these steps in the exact folder where your n8n configuration files are saved.

### Step 1: Create a Patch File
We need to create a small instruction sheet (called a `Dockerfile`) that tells your system how to clean up n8n's internal code.

1. Open your code editor (like VS Code) or a simple text editor (like Notepad).
2. Create a new file and name it exactly: `Dockerfile`
   * *Note for Windows users:* Make sure your computer doesn't accidentally save it as `Dockerfile.txt`. It must have **no file extension** at all.
3. Copy and paste the exact text below into that file, then save it:

```dockerfile
FROM n8nio/n8n:latest

USER root

# Clean out the slow tracking loops from the engine room
RUN find /usr/local/lib/node_modules/n8n/ -name "*.js" \
    -exec sed -i 's/new [a-zA-Z0-9_]*LlmTracing(this)/ /g' {} +

USER node
```

---

### Step 2: Update Your Setup File (`docker-compose.yml`)
Now we need to tell your n8n container to use this patch file on startup instead of pulling the stock setup from the internet.

1. Open your existing `docker-compose.yml` file.
2. Find the line that says `image: n8nio/n8n` or `image: n8nio/n8n:latest`.
3. **Delete that line**, and replace it with: `build: .`
4. *(Optional but recommended)* Under the `environment:` section, add this line to give n8n more breathing room for AI data: `- NODE_OPTIONS=--max-old-space-size=4096`

Your file's core section should look similar to this:

```yaml
services:
  n8n:
    build: .             # <--- Add this line.  This is what tells it to use our patch
    ports:
      - "5678:5678"
    volumes:
      - ./local_files:/home/node/local_files
      - n8n_data:/home/node/.n8n
    environment:
      - N8N_DIAGNOSTICS_ENABLED=false                  # <--Add this line to your environment
      - NODE_OPTIONS=--max-old-space-size=4096         # <--- Add this line to your environment. Bumps up memory
    restart: unless-stopped
```

---

### Step 3: Run the Refresh Command
Lastly, we need to tell the computer to apply these changes.

1. Open your terminal or command prompt window.
2. Make sure you are in the directory containing these files. For this go to the folder where your n8n files are, right click on your mouse and then click on **Open in Terminal** option.
3. Type or paste the following command and press **Enter**:

```bash
docker compose up -d --build
```

### What to expect next:
* Your screen will show some downloading and processing lines for about 30 to 60 seconds. This is completely normal!
* Once it finishes, it will say `Started`. 
* Open your n8n browser tab. Your AI dropdown menus and Gemini elements should now load almost instantly.
