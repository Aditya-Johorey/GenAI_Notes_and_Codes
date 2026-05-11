# AI Connectors Curriculum
### Understanding How Connectors Work — and How to Use Them in Daily Life

**For:** University students, working professionals, and retirees  
**Prerequisite:** Basic familiarity with prompt engineering  
**Structure:** 4 Modules · 14 Lessons  

---

## How to Use This Document

Each lesson includes:
- A **core concept or activity**
- An **analogy or real-world example** to make it tangible
- A **prompt you can try right now** in Claude

Expand lessons in any order, but Module 1 → 4 is the recommended sequence for first-time learners.

---

## Module 1 — What Are Connectors?
*The bridge between AI and your tools*

> **Module goal:** Understand what connectors are, why they exist, and how they extend what AI can do — using analogies and real-world examples.

---

### Lesson 1.1 — AI Without Connectors: The Walled Garden
**Type:** Concept

AI assistants like Claude are trained on vast knowledge, but by default they can't *do* things in the world — they can't check your email, see your calendar, or fetch a live document. They only know what you type to them.

> **Analogy:** Imagine a brilliant advisor who sits in a room with no windows, no phone, and no internet. You can ask them anything — but everything you tell them must come from you first.

Connectors are the equivalent of giving that advisor a phone, a window, and access to your filing cabinet.

**Try this prompt:**
```
Can you give me more analogies to explain what AI without connectors feels like?
```

---

### Lesson 1.2 — What a Connector Actually Is
**Type:** Concept + Definition

A connector (also called an integration or MCP — Model Context Protocol) is a bridge that lets an AI securely talk to an external app or service on your behalf.

- It works in the background — you don't see the wiring
- You give permission once; the AI uses it as needed
- Examples: Gmail, Google Drive, Slack, Notion, Asana

> **Everyday parallel:** When a hotel app asks for your credit card to "connect" to a payment service — you don't do the bank transfer yourself. The app handles it securely. Connectors work the same way.

**Try this prompt:**
```
Show me a simple diagram of how a connector links Claude to an external app like Gmail.
```

---

### Lesson 1.3 — How Connectors Relate to Prompt Engineering
**Type:** Concept — Bridging Prior Knowledge

You already know that a well-crafted prompt changes the quality of AI output. Connectors extend this further — they give AI access to *real data* so your prompts can reference things that actually exist in your world.

| Without a Connector | With a Connector |
|---|---|
| "Summarise the main points of this project update." *(You paste the text)* | "Summarise the Google Doc I shared with my team last Monday." *(Claude fetches it)* |
| "Draft a reply to this email." *(You paste the email)* | "Draft a reply to the last email from my manager." *(Claude reads it)* |

**Key insight:** Better prompts + relevant connectors = significantly more powerful results.

**Try this prompt:**
```
Help me understand when to use a connector versus when a well-crafted prompt alone is enough.
```

---

## Module 2 — How Connectors Work
*Permissions, privacy, and the flow of information*

> **Module goal:** Understand the mechanics — how connectors are turned on, what they can and can't access, and how to think about privacy and trust.

---

### Lesson 2.1 — Turning Connectors On: The Settings Menu
**Type:** Hands-on / Practical

In Claude, connectors are managed in your settings. Here's the general flow:

1. Go to **Settings → Connected apps** (or Integrations)
2. Choose an app — e.g. Gmail or Google Drive
3. Click **Connect** and log in with that app's account
4. Review and grant the permissions shown
5. You can disconnect anytime — your data stays in the original app

> **Tip:** You don't need to connect everything at once. Start with one app you use daily and explore from there.

**Try this prompt:**
```
Walk me through connecting Google Drive to Claude step by step.
```

---

### Lesson 2.2 — What Connectors Can (and Cannot) Access
**Type:** Concept — Scope and Boundaries

Each connector has a defined *scope* — a list of things it's allowed to see or do. This is shown at the moment you connect it.

| Permission Type | What It Means | Example |
|---|---|---|
| **Read-only** | Can view, but not change | Read your email threads |
| **Read + Write** | Can view and modify | Create a calendar event |
| **Action-based** | Can trigger things | Send a Slack message |

> **Real example:** A Gmail connector might ask to "read your email threads." It does not get your password, and it can only do what you authorised. You can revoke this anytime from your Google account settings.

**Try this prompt:**
```
What questions should I ask before connecting an app to my AI assistant?
```

---

### Lesson 2.3 — Privacy and Trust: What You Should Know
**Type:** Concept — Staying Safe and Informed

Connectors follow privacy standards, but staying informed is always wise:

- Only connect apps from providers you trust
- Review what permissions each connector requests before approving
- Avoid connecting highly sensitive systems (e.g. banking) unless offered by the institution directly
- Disconnecting a connector does not delete your data — it just removes AI access

> **Good habit:** Periodically review your connected apps — just like you'd check which apps are installed on your phone.

**Try this prompt:**
```
What are the best practices for staying safe when using AI connectors?
```

---

## Module 3 — Connectors in Daily Life
*Students, professionals, and beyond*

> **Module goal:** Discover how different types of users can use connectors to simplify real tasks they face every day.

---

### Lesson 3.1 — For Students: Research, Notes & Organisation
**Type:** Activity

Students can save hours each week by connecting AI to tools they already use:

| Connector | What You Can Do |
|---|---|
| Google Drive | Summarise lecture notes, help structure essay drafts |
| Gmail | Draft replies to professors, follow up on applications |
| Calendar | Plan study schedules around deadlines and exams |

> **Try this prompt:**
> ```
> Look at my Google Drive folder called 'ECON201' and give me a revision 
> plan for this week based on what's in there.
> ```

**Discussion question for class:** What's one task you do every week that involves searching through files or emails? Could a connector help?

---

### Lesson 3.2 — For Professionals: Emails, Tasks & Reports
**Type:** Activity

Working professionals can use connectors to reduce admin and focus on high-value work:

| Connector | What You Can Do |
|---|---|
| Gmail | Summarise inboxes, draft professional replies, find old threads |
| Google Drive | Find relevant documents, generate reports from existing data |
| Slack / Teams | Catch up on missed conversations, prepare for meetings |
| Asana / Notion | Review tasks, create new ones, update project statuses |

> **Try this prompt:**
> ```
> Find all emails from my client XYZ Ltd in the last two weeks and give 
> me a summary of open issues and any requests I haven't replied to.
> ```

**Discussion question for class:** Think of the most repetitive admin task in your week. How many steps does it involve — and which ones could a connector handle?

---

### Lesson 3.3 — For Retirees: Managing Life Admin With Ease
**Type:** Activity

Connectors can make everyday admin less stressful — no technical expertise required:

| Connector | What You Can Do |
|---|---|
| Gmail | Get help drafting replies to family, banks, or service providers |
| Google Drive | Keep important documents organised and easy to find |
| Calendar | Track appointments, social events, and medication reminders |

> **Try this prompt:**
> ```
> I have a GP appointment on Thursday. Check my calendar and remind me what 
> I need to prepare, then draft a quick note to my daughter about the timing.
> ```

**Key insight for this group:** You describe what you need in plain language. The connectors do the fetching — you don't need to know how it works under the hood.

---

### Lesson 3.4 — Combining Connectors for Multi-Step Tasks
**Type:** Hands-on — The Real Power Unlocked

The greatest value comes when connectors work together in a single, well-crafted request:

**Student example:**
```
Check my calendar for the essay deadline, find the draft in Google Drive, 
and send my tutor an email saying I'll submit it on time.
```

**Professional example:**
```
Find last week's meeting notes in Drive, pull out the action items assigned 
to me, and create tasks in Asana for each one with today's date.
```

**Retiree example:**
```
Check my Gmail for any letter from the NHS, summarise what it says, 
and add the appointment date to my calendar.
```

> This is precisely where your prompt engineering skills become especially valuable — orchestrating multiple steps clearly and deliberately, in the right order.

**Try this prompt:**
```
Give me a multi-step connector task I can try today that would save me real time.
```

---

## Module 4 — Prompting With Connectors
*Making your prompts connector-aware*

> **Module goal:** Apply your existing prompt engineering skills in a world where AI has access to real tools and data.

---

### Lesson 4.1 — Anatomy of a Connector-Aware Prompt
**Type:** Hands-on

Good connector prompts tend to include four ingredients:

| Ingredient | What It Looks Like |
|---|---|
| **Where to look** | "In my Gmail…" / "In the Drive folder called…" |
| **Time range** | "…from the last 7 days" / "…sent in March" |
| **What to do** | "Summarise / Draft / Create / Find / Add" |
| **Output format** | "…in bullet points" / "…as a short email" |

**Weak prompt:**
```
Help with my email.
```

**Strong connector-aware prompt:**
```
In my Gmail, find all unread messages from this week from senders outside 
my organisation. Summarise each one in two sentences and suggest a reply 
priority: urgent, normal, or low.
```

**Try this prompt:**
```
Help me write a connector-aware prompt for managing my weekly tasks.
```

---

### Lesson 4.2 — When Connectors Aren't Needed
**Type:** Concept — Knowing the Difference

Not every task needs a connector. Here's a simple guide:

**Use a connector when:**
- The data lives in an external app (email, drive, calendar)
- The task requires fetching current or real-time information
- You need to create or update something in another tool

**Skip the connector when:**
- You can paste the content directly — it's often faster
- The task is fully self-contained (e.g. "explain this concept to me")
- You're not sure the connector has access to what you need

> **Rule of thumb:** If you'd normally open another app to get the information, a connector can probably fetch it for you.

**Try this prompt:**
```
Give me 3 examples of tasks where I should use a connector and 3 where I shouldn't.
```

---

### Lesson 4.3 — Iterating on Connector Results
**Type:** Hands-on — Refining Like an Expert

Just like with regular prompts, your first attempt isn't always perfect. Iteration is normal:

| Problem | How to Fix It |
|---|---|
| Results too broad | Add constraints — "only from this sender", "only this folder" |
| Something missing | Tell Claude what was wrong and rephrase |
| Action didn't happen | Check if the connector has write permission, then retry |

**Iteration example:**

First attempt — too broad:
```
Find my project files in Drive.
```

Refined prompt:
```
Find files in the Drive folder 'Q2 Project' modified in the last 14 days, 
and list them with the last editor's name next to each one.
```

**Try this prompt:**
```
What are the most common mistakes people make when prompting with connectors, 
and how do I fix them?
```

---

### Lesson 4.4 — Your First Week With Connectors: A Practice Plan
**Type:** Activity — Putting It All Together

A gentle plan to build confidence over your first week:

| Day | Task |
|---|---|
| **Day 1** | Connect one app you use daily (Gmail or Google Drive) |
| **Day 2** | Ask a simple read-only question: "Summarise my last 5 emails" |
| **Day 3** | Try a write task: "Draft a reply to this email" |
| **Day 4** | Combine two data sources in one prompt |
| **Day 5–7** | Identify one repetitive task in your week and try to handle it with a connector |

**Reflection prompt for end of week:**
```
Based on my activities this week, what recurring tasks could I offload 
to AI using my connected apps?
```

---

## Quick Reference

### The Connector Prompt Formula
```
[Where to look] + [Time range] + [What to do] + [Output format]
```

### Use a Connector When...
- Data lives in another app
- You need real-time or current information
- You want to create or update something in another tool

### Stay Safe by...
- Only connecting apps from providers you trust
- Reviewing permissions before approving
- Disconnecting apps you no longer use
- Remembering: disconnecting doesn't delete your data

---

## Glossary

| Term | Plain-language definition |
|---|---|
| **Connector** | A bridge that lets AI securely access an external app on your behalf |
| **MCP (Model Context Protocol)** | The technical standard that connectors are built on |
| **Permission / Scope** | The list of things a connector is allowed to see or do |
| **Read-only** | Can view data, but cannot change anything |
| **Read + Write** | Can both view and modify data |
| **Integration** | Another word for connector — same idea, different label |

---

*Curriculum designed for mixed learner groups — students, professionals, and retirees. Build on prompt engineering foundations. No technical background assumed.*
