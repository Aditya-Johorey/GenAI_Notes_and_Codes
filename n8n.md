# n8n Complete Teaching Guide (Beginner → Advanced)

**Author:** ChatGPT (Customized for Aditya Johorey)
**Audience:** Students with no prior automation or backend experience
**Goal:** Take students from zero → production-ready n8n automation builders, including AI workflows

---

## 📌 How to Use This Document

This guide is written as:

* A **teacher’s handbook**
* A **student reference book**
* A **step-by-step curriculum**

You can:

* Teach module-by-module
* Convert directly into slides
* Use labs as assignments
* Run projects as capstone work

---

# TABLE OF CONTENTS

1. What is n8n & Automation Thinking
2. Core Architecture of n8n
3. Workflows & Execution Model
4. Nodes (Deep Dive)
5. Data & JSON in n8n
6. Expressions System
7. Triggers (Deep Dive)
8. Core Processing Nodes
9. Logic & Routing Nodes
10. Function Node & Custom Code
11. Merge Node & Data Joining
12. Workflow Design Patterns
13. Credentials, Secrets & Auth
14. Error Handling & Reliability
15. Debugging & Testing
16. Scheduling & Background Jobs
17. Webhooks & API Design
18. Databases & Storage
19. Deployment & Production
20. AI Automation with n8n
21. Teaching Roadmap
22. Labs & Projects
23. Assessment Framework
24. Real-World Use Cases
25. Glossary

---

# 1. WHAT IS n8n & AUTOMATION THINKING

## 1.1 What is n8n?

n8n ("node-n") is a **workflow automation platform** that allows you to connect software systems, APIs, databases, and AI models visually — without writing full backend applications.

Instead of coding entire servers, you build **pipelines**.

### Traditional Backend

```
Request → Controller → Logic → Database → Response
```

### n8n Backend

```
Trigger → Node → Node → Node → Output
```

Each block = a **node**
Each connection = **data flow**

---

## 1.2 What Problem Does n8n Solve?

| Problem                          | n8n Solution          |
| -------------------------------- | --------------------- |
| Writing boilerplate backend code | Visual pipelines      |
| Integrating APIs                 | Built-in connectors   |
| Orchestrating AI agents          | Workflow logic        |
| Automation                       | Event-driven triggers |
| Scaling pipelines                | Queue execution       |

---

## 1.3 Automation Mindset

Students must learn to think in:

> **Events → Transformations → Decisions → Actions**

Instead of:

> "Write a program"

Think:

> "What happens when X occurs?"

---

## 1.4 Where n8n is Used

* AI agents
* Chatbots
* Data pipelines
* SaaS automation
* Backend APIs
* Business process automation
* Webhook servers
* ETL pipelines

---

# 2. CORE ARCHITECTURE OF n8n

## 2.1 Key Components

| Component  | Purpose                       |
| ---------- | ----------------------------- |
| Workflow   | Connected nodes forming logic |
| Node       | Single operation step         |
| Trigger    | Entry point                   |
| Execution  | One run of workflow           |
| Item       | One data object               |
| Credential | Stored secret                 |
| Expression | Dynamic data reference        |

---

## 2.2 High-Level System View

```
External Event/API
      ↓
   Trigger Node
      ↓
Processing Nodes
      ↓
Decision Nodes
      ↓
 Action Nodes
      ↓
 External System
```

---

## 2.3 Event-Driven Architecture

n8n workflows react to:

* HTTP calls
* Timers
* App events
* Manual runs

---

# 3. WORKFLOWS & EXECUTION MODEL

## 3.1 What is a Workflow?

A **workflow** is a directed graph of nodes that process data.

* Always flows left → right
* Each node transforms data

---

## 3.2 What is an Execution?

An **execution** is one complete run of a workflow.

Each execution has:

* Input data
* Node results
* Logs
* Output

---

## 3.3 What is an Item?

An **item** is one JSON object flowing through nodes.

Example:

```json
{
  "name": "Aditya",
  "email": "abc@gmail.com"
}
```

Multiple items:

```json
[
  {"name": "A"},
  {"name": "B"},
  {"name": "C"}
]
```

---

## 3.4 Node Execution Rule (Critical)

> **Each node runs once per item.**

Example:

* 5 items enter a node
* Node runs 5 times
* Outputs 5 items

---

## 3.5 Parallel Branching

Each branch processes items independently.

```
       → Branch A →
Trigger → Branch B → Merge → Output
       → Branch C →
```

---

# 4. NODES (DEEP DIVE)

## 4.1 What is a Node?

A **node** is a single operation in a workflow.

Examples:

* Receive webhook
* Call API
* Transform JSON
* Route logic
* Store data

---

## 4.2 Node Categories

| Category | Examples              |
| -------- | --------------------- |
| Trigger  | Webhook, Cron, Manual |
| Core     | Set, Function, Merge  |
| Logic    | If, Switch            |
| App      | Gmail, Slack, Notion  |
| Data     | MySQL, Postgres       |
| AI       | OpenAI, HuggingFace   |

---

## 4.3 Node Inputs & Outputs

Each node:

* Receives **items**
* Processes them
* Outputs new items

---

## 4.4 Node UI Anatomy

Each node has:

* Parameters panel
* Input/output pins
* Execution data
* Error output

---

# 5. DATA & JSON IN n8n

## 5.1 Everything is JSON

All data in n8n flows as JSON objects.

Example:

```json
{
  "user": "Aditya",
  "age": 23,
  "active": true
}
```

---

## 5.2 Nested JSON

```json
{
  "user": {
    "name": "Aditya",
    "skills": ["ML", "Robotics", "AI"]
  }
}
```

Access:

```
$json.user.name
$json.user.skills[0]
```

---

## 5.3 Arrays & Items

Each array element becomes a separate **item** in execution.

---

## 5.4 Data Transformation Philosophy

> Each node:
> Input JSON → Transform → Output JSON

---

# 6. EXPRESSIONS SYSTEM

## 6.1 What are Expressions?

Expressions allow dynamic values inside node fields.

They start with:

```
{{ ... }}
```

---

## 6.2 Basic Expression Examples

| Use Case             | Expression                      |
| -------------------- | ------------------------------- |
| Current field        | `{{ $json.name }}`              |
| Other node field     | `{{ $node["Set"].json.email }}` |
| Current time         | `{{ $now }}`                    |
| Environment variable | `{{ $env.API_KEY }}`            |

---

## 6.3 Math in Expressions

```js
{{ $json.price * $json.quantity }}
```

---

## 6.4 Conditional Expressions

```js
{{ $json.score > 50 ? "pass" : "fail" }}
```

---

## 6.5 Expression Debugging

Use the **Expression Editor** to test values.

---

# 7. TRIGGERS (DEEP DIVE)

## 7.1 What is a Trigger?

A **trigger** starts a workflow.

Without a trigger, workflow cannot run.

---

## 7.2 Common Trigger Types

| Trigger     | Use Case             |
| ----------- | -------------------- |
| Manual      | Testing              |
| Webhook     | API calls, bots      |
| Cron        | Scheduled jobs       |
| App Trigger | Gmail, Slack, GitHub |

---

## 7.3 Manual Trigger

Used during:

* Testing
* Development

---

## 7.4 Webhook Trigger (Very Important)

Allows external systems to call your workflow.

Example:

```
POST https://n8n.domain/webhook/chat
```

Payload:

```json
{
  "message": "Hello"
}
```

Webhook node outputs this JSON into workflow.

---

## 7.5 Webhook Methods

* GET
* POST
* PUT
* DELETE

---

## 7.6 Cron Trigger

Used for:

* Nightly jobs
* Scheduled syncs
* Reports

---

# 8. CORE PROCESSING NODES

## 8.1 Set Node

### Purpose

Used to:

* Rename fields
* Create new fields
* Remove unwanted fields

---

### Example

Input:

```json
{
  "first": "Aditya",
  "last": "Johorey"
}
```

Set Node Output:

```json
{
  "fullName": "Aditya Johorey"
}
```

---

## 8.2 HTTP Request Node (Most Important Node)

### Purpose

* Call REST APIs
* Send webhooks
* Interact with SaaS tools
* Call AI models

---

### Key Concepts

| Concept      | Meaning                |
| ------------ | ---------------------- |
| Method       | GET, POST, PUT, DELETE |
| URL          | Endpoint               |
| Headers      | Auth, content-type     |
| Body         | Payload                |
| Query params | URL parameters         |

---

### Example: POST Request

```json
POST https://api.example.com/user
Headers:
  Authorization: Bearer TOKEN
Body:
{
  "name": "Aditya"
}
```

---

## 8.3 Respond to Webhook

Used to send response back to caller.

---

# 9. LOGIC & ROUTING NODES

## 9.1 If Node

Routes items into:

* True branch
* False branch

---

### Example

Condition:

```
$json.score > 70
```

---

## 9.2 Switch Node

Routes items into **multiple paths**.

---

### Example

| Case             | Route            |
| ---------------- | ---------------- |
| intent = sales   | Sales pipeline   |
| intent = support | Support pipeline |
| intent = billing | Billing pipeline |

---

## 9.3 Compare Mode vs Expression Mode

Teach both for flexibility.

---

# 10. FUNCTION NODE & CUSTOM CODE

## 10.1 What is the Function Node?

Allows writing JavaScript to manipulate items.

Used when:

* Logic too complex for Set/IF
* Looping
* Validations
* Custom parsing

---

## 10.2 Basic Structure

```js
return items.map(item => {
  item.json.fullName = item.json.first + " " + item.json.last;
  return item;
});
```

---

## 10.3 Accessing Data

```js
item.json.field
items[0].json
```

---

## 10.4 Creating New Items

```js
return [
  { json: { value: 1 } },
  { json: { value: 2 } }
];
```

---

## 10.5 Filtering Items

```js
return items.filter(i => i.json.score > 50);
```

---

## 10.6 Error Throwing

```js
throw new Error("Invalid data");
```

---

# 11. MERGE NODE & DATA JOINING

## 11.1 Why Merge is Needed

When:

* Data splits into branches
* Must be recombined

---

## 11.2 Merge Modes

| Mode         | Purpose                |
| ------------ | ---------------------- |
| Append       | Combine lists          |
| Combine      | Join by key            |
| Wait         | Wait for both branches |
| Pass-through | Choose one branch      |

---

## 11.3 Example: Combine by Key

Branch A:

```json
{ "id": 1, "name": "Aditya" }
```

Branch B:

```json
{ "id": 1, "email": "abc@gmail.com" }
```

Merged:

```json
{ "id": 1, "name": "Aditya", "email": "abc@gmail.com" }
```

---

# 12. WORKFLOW DESIGN PATTERNS

## 12.1 Linear Pipeline

```
Trigger → Process → Output
```

Used for:

* Data transformation
* API forwarding

---

## 12.2 Branching Logic

```
Trigger → If/Switch → Multiple actions
```

---

## 12.3 Fan-Out → Fan-In

```
Trigger → Split → Parallel → Merge → Output
```

---

## 12.4 Event-Based Automation

```
Event → Filter → Action
```

---

## 12.5 Retry & Error Recovery

```
Action → Fail? → Retry / Notify / Log
```

---

# 13. CREDENTIALS, SECRETS & AUTH

## 13.1 Why Credentials Matter

Never hardcode secrets inside workflows.

---

## 13.2 Credential Types

| Type       | Used For      |
| ---------- | ------------- |
| API Key    | Simple auth   |
| OAuth2     | Gmail, Slack  |
| Basic Auth | Legacy APIs   |
| Token Auth | Bearer tokens |

---

## 13.3 Credential Storage

Stored encrypted inside n8n.

---

## 13.4 Using Credentials in Nodes

Attach credential in node config.

---

# 14. ERROR HANDLING & RELIABILITY

## 14.1 Why Error Handling Matters

Production systems must not silently fail.

---

## 14.2 Continue on Fail

Allows workflow to continue even if node errors.

---

## 14.3 Error Trigger Workflow

Special workflow triggered when any execution fails.

---

## 14.4 Try/Catch Pattern

```
Action → If error → Alternate flow
```

---

## 14.5 Retry Strategies

* Retry after delay
* Retry with exponential backoff

---

# 15. DEBUGGING & TESTING

## 15.1 Manual Execution

Run node-by-node during development.

---

## 15.2 Execution Logs

View:

* Input
* Output
* Errors

---

## 15.3 Pin Data

Pin node output for stable testing.

---

## 15.4 Testing Webhooks

Use:

* Postman
* curl
* Browser

---

# 16. SCHEDULING & BACKGROUND JOBS

## 16.1 Cron Node

Schedule workflows:

* Hourly
* Daily
* Weekly

---

## 16.2 Use Cases

* Daily reports
* Sync databases
* Data scraping
* Cleanup jobs

---

# 17. WEBHOOKS & API DESIGN

## 17.1 Webhooks as Backend APIs

n8n workflows can act as REST endpoints.

---

## 17.2 Request Lifecycle

```
Client → Webhook → Workflow → Respond → Client
```

---

## 17.3 Validation Layer

Validate inputs before processing.

---

## 17.4 Response Formatting

Use Respond to Webhook node.

---

# 18. DATABASES & STORAGE

## 18.1 Storage Options

| Type          | Examples                |
| ------------- | ----------------------- |
| Relational DB | MySQL, Postgres         |
| NoSQL         | MongoDB                 |
| SaaS          | Google Sheets, Airtable |
| Files         | S3, Local               |
| Memory        | Static data             |

---

## 18.2 Static Data

Persist small workflow state.

---

## 18.3 CRUD Operations

Create, Read, Update, Delete via nodes.

---

# 19. DEPLOYMENT & PRODUCTION

## 19.1 Development vs Production

| Environment | Purpose        |
| ----------- | -------------- |
| Local       | Testing        |
| Staging     | Validation     |
| Production  | Live workloads |

---

## 19.2 Workflow Activation

Inactive → Active

---

## 19.3 Version Control

Export workflows as JSON.

---

## 19.4 Scaling

* Queue mode
* Worker nodes

---

## 19.5 Security

* HTTPS
* Secrets management
* Role-based access

---

# 20. AI AUTOMATION WITH n8n

(Aligned with your background in ML, RAG, and agents)

---

## 20.1 Why n8n for AI?

Because AI systems need:

* Orchestration
* Routing
* Tool calling
* Memory
* API glue

n8n becomes the **AI backend brain**.

---

## 20.2 Basic LLM Pipeline

```
Webhook → Prompt Builder → LLM → Parser → Output
```

---

## 20.3 Calling LLMs

Via:

* HTTP Request
* OpenAI node
* Ollama
* HuggingFace

---

## 20.4 Prompt Engineering in n8n

Build prompts dynamically:

```js
"User question: " + $json.query
```

---

## 20.5 JSON Output Parsing

Use:

* Function node
* Structured output prompts

---

## 20.6 AI Router Pattern

```
Input → Classifier → Switch → Specialized agents
```

---

## 20.7 RAG Architecture in n8n

```
Upload Docs → Chunk → Embed → Store → Retrieve → LLM → Answer
```

---

## 20.8 Agent Tool-Calling

```
User → LLM → Tool Decision → Tool Node → Result → LLM
```

---

# 21. TEACHING ROADMAP (CLASSROOM-READY)

## Level 1 — Foundations (2 Days)

### Topics

1. What is automation
2. n8n UI tour
3. Workflows
4. Manual trigger
5. Set node
6. JSON basics
7. Expressions
8. HTTP Request

### Lab

Webhook → API forwarder

---

## Level 2 — Logic & Control (2 Days)

### Topics

1. If node
2. Switch node
3. Merge node
4. Execution model
5. Function node basics

### Lab

Email intent router

---

## Level 3 — Real Automation (2 Days)

### Topics

1. Credentials
2. OAuth
3. Error handling
4. Cron jobs
5. Multi-step pipelines

### Lab

Scheduled data sync pipeline

---

## Level 4 — AI Automation (Advanced)

### Topics

1. LLM calls
2. Prompt construction
3. JSON parsing
4. Agent routing
5. RAG pipelines

### Lab

AI customer support backend

---

# 22. LABS & PROJECTS

## Beginner Labs

1. Webhook → Google Sheets logger
2. Contact form → Email sender
3. API data fetch → JSON transform

---

## Intermediate Labs

1. CRM lead router
2. News scraper → summary → email
3. Payment webhook → invoice generator

---

## Advanced Labs

1. AI chatbot backend
2. RAG document assistant
3. Autonomous task agent

---

# 23. ASSESSMENT FRAMEWORK

| Skill              | How to Test              |
| ------------------ | ------------------------ |
| JSON understanding | Transformation exercises |
| Logic routing      | Multi-path workflows     |
| API skills         | HTTP Request labs        |
| Reliability        | Error handling tasks     |
| AI orchestration   | RAG project              |

---

# 24. REAL-WORLD USE CASES

* AI customer service bots
* Lead qualification pipelines
* Automated reporting systems
* Backend for SaaS MVPs
* Event-driven IoT automation
* Robotics telemetry pipelines

---

# 25. GLOSSARY

| Term       | Meaning                  |
| ---------- | ------------------------ |
| Workflow   | Connected nodes pipeline |
| Node       | Single processing step   |
| Item       | One JSON object          |
| Execution  | One workflow run         |
| Trigger    | Workflow entry point     |
| Expression | Dynamic field reference  |
| Credential | Secure secret storage    |
| Merge      | Combine branches         |
| Fan-out    | Splitting items          |
| Fan-in     | Recombining items        |

---

# 🎯 FINAL NOTE

This document gives students:

* Backend thinking
* Automation mindset
* API fluency
* AI orchestration skills

It positions them not just as **n8n users**, but as **systems engineers**.

---

If you want, I can:
✅ Convert this into **slides**
✅ Add **step-by-step screenshots**
✅ Create **hands-on lab sheets**
✅ Build **assessment quizzes**
✅ Add **AI-specific code labs**

Just tell me 😄
