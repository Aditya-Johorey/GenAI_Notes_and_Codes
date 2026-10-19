# Sample prompts by feature

Some features are triggered by your words, and others are toggles or setup steps in the app. I've marked which is which.

---

## Core abilities (prompt-triggered)

**Writing**
- "Rewrite this email to sound warmer but still professional: [paste]"
- "Write a 150-word product description for a handmade candle, playful tone."

**Analysis and reasoning**
- "Compare renting vs. buying a home for someone staying 4 years. Lay out the key tradeoffs."
- "Think step by step: is this argument valid? [paste]"

**Coding**
- "Write a Python script that renames all files in a folder by date."
- "Here's my error message and code. Find the bug and explain it."

**Vision** (attach an image first)
- "What's in this photo, and what's the text on the sign?"
- "Read this chart and tell me the three main takeaways."
- "Here's a screenshot of my app. What's wrong with the layout?"

**File analysis** (attach a file first)
- "Summarize this PDF in 5 bullets, then list any deadlines."
- "Extract every date and dollar amount from this contract into a table."

**Languages**
- "Translate this to Spanish, Mexican style, casual."
- "Explain the difference between ser and estar with examples."

---

## Finding current information

**Web search** (toggle on in settings, or just ask for current info)
- "What's the latest news on [topic]? Use search."
- "Who is the current CEO of [company]?"

**Research mode** (toggle it on, then prompt)
- "Brief me on how mid-size SaaS companies are using AI agents in customer support, with three vendor examples and the risks."
- "Research the evidence on intermittent fasting, with sources."

**Images, maps, sports** (apps only)
- "Show me what a pangolin looks like."
- "Find good ramen places near me and show them on a map." (needs location permission)
- "What was the latest score for [team]?"

---

## Organizing your work

**Projects** (a setup feature, so create one in the sidebar)
- Instructions to paste in: "You are my writing coach for my novel. Always check against the style guide in the project files."
- Then prompt: "Using the project files, draft chapter 3's opening."

**Memory** (toggle in Settings → Capabilities)
- "Remember that I'm vegetarian and cook for two."
- "What do you remember about me?"
- "Forget what I told you about my job."
- "Look back at our earlier chat about my budget and continue it." (needs chat search on)

**Incognito** is a toggle or button in the app, not a prompt.

**Styles**
- Pick a preset such as Concise, Explanatory, or Formal, or create a custom one by uploading your writing.
- In-chat: "From now on, keep answers under 100 words and skip bullet points."

---

## Building things

**Artifacts**
- "Build me an interactive mortgage calculator."
- "Make a simple Snake game I can play here."
- "Create a one-page HTML landing page for a dog-walking business, with a dark mode."
- "Make a flowchart of how a bill becomes law."
- "Build a habit tracker that remembers my entries between sessions." (triggers persistent storage)
- "Build a small app where I paste text and Claude rewrites it in three tones." (triggers an AI-powered artifact)

**Charts and diagrams**
- "Chart these numbers as a bar graph: Q1 40, Q2 55, Q3 48."
- "Diagram how HTTPS works."

**File creation**
- "Make a Word doc of this meeting summary."
- "Create a 6-slide PowerPoint on renewable energy for high schoolers."
- "Build an Excel budget tracker with monthly totals and formulas."
- "Turn this into a PDF I can email."
- "Clean this messy CSV: remove duplicates and fix the date format." (attach the file)

**Docs and designs** (where available)
- "Write a project proposal doc for a community garden."
- "Design a flyer for a bake sale, with a few options."
- "Mock up a mobile app screen for a plant-care app."

---

## Connecting your tools

**Connectors** (set up first under Settings → Connectors, then prompt)
- Gmail: "Find the latest email from my landlord and summarize it."
- Drive: "Find my Q3 sales deck and list the main points."
- Slack, Notion, Asana: "Create a task for reviewing the budget, due Friday."
- "What connectors can help me manage tasks?" (Claude can suggest options)

**Skills** (they load automatically when relevant)
- "Make a presentation about [topic]." (the PowerPoint skill is used behind the scenes)
- To create your own: "Help me create a skill that formats our weekly status reports in our standard layout."

**Phone and device tools** (in the mobile app, where permitted)
- "Set an alarm for 6:45 AM."
- "Set a 10-minute timer for the pasta."
- "Add dinner with Sam to my calendar Friday at 7."
- "Draft a text to my manager saying I'll be late." (shows an editable draft card)
- "Show me a recipe for pad thai that I can scale to 6 servings."

---

## Agentic products

These run outside the regular chat, so you open each one separately.

- **Claude Code** (terminal or desktop): "Read this repo and add input validation to the signup form, then run the tests."
- **Cowork** (desktop): "Organize the files in my Downloads folder by type and date, then list what you moved."
- **Claude in Chrome**: "Find the cheapest nonstop flight to Denver next Friday on this site."
- **Claude in Excel**: "Add a column that calculates profit margin and highlight the lowest 5."
- **Claude in PowerPoint**: "Tighten these slides and make the fonts consistent."
- **Google Workspace sidebar**: "Summarize this doc and suggest edits as comments."
- **Claude Tag** (Slack): "@Claude summarize this thread and list action items."

---

## Developer topics

- "Show me how to call the Claude API in Python with a system prompt."
- "How do I use tool calling to let Claude query a database?"
- "Explain the difference between streaming and batch requests." (Claude should check current docs for specifics)

---

## Prompting tips that boost any feature

- **Be specific**: state the audience, goal, length, and format.
- **Show examples**: "Match the tone of this paragraph: [paste]."
- **Ask for reasoning**: "Walk me through it step by step."
- **Name the format**: "as a table," "as a Word doc," "in 3 bullets."
- **Iterate**: "Make it shorter," "more formal," "add a section on risks."

Want these arranged as a printable cheat sheet in a Word or PDF file?
