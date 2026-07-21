> **By the end of 4 weeks, every student can analyze a business problem, design an AI-powered workflow, integrate multiple applications, and deploy a production-ready automation in n8n without writing code.**

I would structure the curriculum around increasing levels of capability rather than around n8n features.

---

# Week 1 — Thinking in Workflows

**Goal:** Learn how to break real-world problems into automation steps.

## Session 1 — Automation Mindset

Students usually think:

> "How do I automate this?"

Teach them to think:

```
Trigger
↓

Collect Data

↓

Decision

↓

AI

↓

Action

↓

Store Result

↓

Notify User
```

Topics

* What is workflow automation?
* AI vs Automation
* When should AI make decisions?
* Event-driven workflows
* Sequential vs parallel workflows
* Error handling basics

Exercise:
Convert five everyday tasks into workflow diagrams before opening n8n.

---

## Session 2 — Mastering n8n

Deep dive into:

* Nodes
* Credentials
* Variables
* Expressions
* JSON
* Execution data
* Data transformation
* Loops
* Merge node
* IF node
* Switch node

Mini Projects:

* Auto email responder
* AI text summarizer
* File organizer

---

## Session 3 — APIs Without Code

Students often fear APIs.

Teach:

* Request
* Response
* Headers
* Authentication
* GET
* POST
* Webhooks

Tools

* Postman (for understanding APIs)
* ReqBin (browser-based API testing)
* Swagger/OpenAPI documentation

Project

Connect a free Weather API and generate an AI weather report.

---

## Session 4 — Prompt Design Inside Workflows

Difference between

```
Prompt Engineering
```

and

```
Automation Prompt Design
```

Topics

* Dynamic prompts
* Variables
* Structured outputs (JSON)
* Prompt templates
* Error-resistant prompting
* Cost optimization

---

# Week 2 — Building Intelligent Workflows

Goal:

Students stop building simple automations and begin building AI-powered systems.

---

## Session 1 — AI Nodes

Explore

* OpenAI
* Gemini
* Claude (via API)
* OpenRouter

Teach:

* Model selection
* Temperature
* Tokens
* Structured outputs
* Function calling (conceptually)

Exercise

Compare the same workflow using three different models.

---

## Session 2 — Multi-Step AI Workflows

Instead of

```
User

↓

AI

↓

Done
```

Teach

```
Receive Request

↓

Classify

↓

Research

↓

Summarize

↓

Format

↓

Send
```

Projects

* Email assistant
* Meeting summarizer
* AI note generator

---

## Session 3 — Document Automation

Tools

* Google Drive
* Google Docs
* Google Sheets
* PDFs

Projects

Invoice reader

↓

Extract information

↓

AI verification

↓

Store in Sheets

↓

Email report

---

## Session 4 — Human-in-the-Loop

Teach

Not everything should be automated.

Examples

AI drafts

↓

Manager approves

↓

Workflow continues

Use

* Slack
* Gmail
* Telegram
* Discord

---

# Week 3 — AI Agent Foundations in n8n

Goal:

Students build workflows that behave like agents.

---

## Session 1 — What Makes an Agent?

Difference between

Chatbot

Automation

AI Workflow

AI Agent

Teach concepts:

* Goal
* Planning
* Memory
* Tool use
* Decision making

Project

Personal AI Assistant

---

## Session 2 — Tool-Using Agents

Give workflows access to:

* Google Search
* Tavily
* Firecrawl
* Calculator
* Calendar
* Gmail
* Google Sheets
* Notion

Teach:

An AI becomes useful when it can **take actions**, not just generate text.

---

## Session 3 — Knowledge & Retrieval

Introduce Retrieval-Augmented Generation (RAG):

```
Documents

↓

Embeddings

↓

Vector Search

↓

Relevant Context

↓

LLM Response
```

Tools

* Dify Knowledge Base
* Flowise RAG
* Pinecone (conceptual)
* Qdrant (conceptual)

Projects

* Company FAQ assistant
* Resume Q&A
* Research assistant

---

## Session 4 — Robust Workflows

Topics

* Error handling
* Retries
* Rate limits
* Logging
* Monitoring
* Cost optimization
* Versioning workflows

Students intentionally break workflows and learn to recover gracefully.

---

# Week 4 — Production Projects

## Session 1 — Solution Design

Students define:

* User
* Problem
* Trigger
* AI tasks
* External tools
* Outputs
* Success criteria

---

## Session 2 — Build

Each student develops a complete workflow.

Examples:

* AI CRM assistant
* Lead qualification workflow
* Social media content generator
* Customer support automation
* HR resume screening
* Research assistant
* Meeting minutes generator
* Invoice processing system
* AI travel planner

---

## Session 3 — Testing & Optimization

Teach:

* Happy path
* Edge cases
* Performance
* Cost analysis
* Prompt refinement
* User feedback

---

## Session 4 — Demo Day

Each student presents:

* Problem statement
* Workflow architecture
* Live demo
* Failure handling
* Future improvements

---

# Essential Tools to Introduce

| Category        | Recommended Tools                           | Why They Matter                     |
| --------------- | ------------------------------------------- | ----------------------------------- |
| Automation      | **n8n**                                     | Core workflow builder               |
| AI Models       | Gemini, OpenAI, Claude, OpenRouter          | Compare reasoning, cost, and speed  |
| API Testing     | Postman, ReqBin                             | Understand and debug APIs           |
| Knowledge Bases | Google Drive, Notion, PDFs                  | Build document-aware workflows      |
| Search          | Tavily, Exa                                 | Reliable web search for AI          |
| Web Extraction  | Firecrawl, Jina AI Reader                   | Clean website content for workflows |
| Databases       | Google Sheets, Airtable, Supabase           | Store workflow data                 |
| Notifications   | Gmail, Slack, Telegram, Discord             | Deliver workflow results            |
| Storage         | Google Drive, Dropbox                       | Manage files and documents          |
| Scheduling      | Google Calendar                             | Time-based automations              |
| Forms           | Tally, Google Forms, Typeform               | Collect user input                  |
| Monitoring      | n8n execution logs, Better Stack (optional) | Debug and observe workflows         |

---

# Recommended Progression of Projects

1. AI Email Summarizer
2. PDF Information Extractor
3. AI Blog Generator
4. Customer Support Ticket Classifier
5. AI Research Assistant
6. Meeting Notes Generator
7. Resume Screening Workflow
8. Invoice Processing Automation
9. AI Social Media Content Pipeline
10. Multi-step Business Assistant (Capstone)

---

## Final Capstone Challenge

Give students a real business scenario instead of a predefined tutorial. For example:

> "A small e-commerce company receives customer emails, order issues, invoices, and product questions every day. Build an AI-powered workflow that classifies incoming requests, retrieves relevant information, drafts responses, updates a spreadsheet, and notifies the support team when human intervention is required."

This forces students to combine everything they've learned—workflow design, AI prompting, APIs, external integrations, document processing, decision logic, and deployment—into a production-style automation. It's the closest experience to what they'll encounter in industry while still remaining entirely no-code.
