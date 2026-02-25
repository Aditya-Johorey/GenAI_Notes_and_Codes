# Model Context Protocol (MCP) – Teaching Notes

## Audience

Non-technical professionals (business, finance, operations, healthcare, etc.)

## Level

Explained with conceptual clarity, but with depth expected from an AI Engineer.

---

# 1. What Problem Does MCP Solve?

Before MCP, connecting AI models to tools was messy.

Imagine:

* You have an AI model (like a chatbot)
* You have tools (calculator, database, Excel, CRM, API, etc.)

Each integration required:

* Custom code
* Custom format
* Custom communication logic

This created:

* High engineering effort
* Poor reusability
* Tight coupling between AI and tools

### MCP solves this by:

Providing a **standard communication protocol** between:

> AI Model  ↔  Tool Server

Just like HTTP standardised websites, MCP standardises AI-tool communication.

---

# 2. What is MCP (Simple Definition)?

**Model Context Protocol (MCP)** is a structured way for AI models to:

* Discover tools
* Understand how to use them
* Call them safely
* Receive structured responses

Think of MCP as:

> "USB port for AI tools"

You plug tools in, and the model knows how to use them.

---

# 3. Core Components of MCP

There are 3 main components:

## 3.1 MCP Server

This is where tools live.

It:

* Defines tools
* Describes inputs (schema)
* Describes outputs (schema)
* Executes logic

Example tools:

* add_numbers
* fetch_stock_price
* get_customer_data

## 3.2 MCP Client

This connects to the MCP server.

It:

* Asks the server what tools exist
* Sends tool execution requests
* Receives tool responses

In your code, this is:

* ClientSession
* stdio_client

## 3.3 AI Model

The intelligence layer.

It:

* Reads user message
* Decides whether tool is needed
* Generates tool call request
* Uses tool result to generate final answer

---

# 4. High-Level Architecture

User → AI Model → MCP Client → MCP Server → Tool
↑
Tool Result
↑
User ← AI Model ← MCP Client ← MCP Server

Important idea:

The model does NOT directly execute code.
It REQUESTS tools.

This separation improves:

* Safety
* Auditability
* Control
* Security

---

# 5. What is a Tool in MCP?

A tool is defined using:

1. Name
2. Description
3. Input Schema (JSON Schema)
4. Output Schema

Example tool schema:

Input:
{
"a": number,
"b": number
}

Output:
{
"result": string
}

This schema is VERY important.

Why?
Because the model reads the schema to understand:

* What arguments are required
* What type they should be
* What structure to return

Schema = Contract between AI and Tool.

---

# 6. Lifecycle of an MCP Interaction

Let’s break it down step-by-step.

## Step 1 – Initialize Session

Client connects to server.
Handshake happens.
Capabilities exchanged.

## Step 2 – Discover Tools

Client asks server:

"What tools do you have?"

Server responds with structured tool definitions.

## Step 3 – Convert Tools to Model Format

Different models require different tool formats.

Example:
Ollama expects:

{
"type": "function",
"function": {
"name": "tool_name",
"description": "...",
"parameters": {...}
}
}

## Step 4 – User Sends Message

User: "Add 5 and 10"

Model sees:

* It cannot compute directly (or is instructed not to)
* There is a tool called add_numbers

## Step 5 – Model Generates Tool Call

Instead of text answer, model returns:

{
tool_calls: [
{
function: {
name: "add_numbers",
arguments: {"a": 5, "b": 10}
}
}
]
}

## Step 6 – Client Executes Tool

Client receives tool call.
Client calls:

session.call_tool("add_numbers", {"a": 5, "b": 10})

Server executes logic.

Returns:

{"result": "15"}

## Step 7 – Tool Result Sent Back to Model

Now model receives:

Role: tool
Content: {"result": "15"}

Model now generates final natural language answer:

"The result is 15."

This is called:

Tool-Augmented Generation

---

# 7. Why MCP is Important in Real Systems

As an AI Engineer, here’s why MCP matters:

## 7.1 Scalability

Without MCP:
Each AI + Tool integration is custom.

With MCP:
Plug-and-play architecture.

## 7.2 Separation of Concerns

* AI decides WHAT to do.
* Tools decide HOW to do it.

## 7.3 Security

You can:

* Restrict tools
* Log tool usage
* Audit execution

The model never gets raw system access.

## 7.4 Enterprise Integration

In enterprise systems, you may connect:

* ERP
* CRM
* Databases
* Financial systems
* Analytics tools

MCP allows structured orchestration.

---

# 8. Teaching Analogy for Non-Tech Students

Use this analogy:

AI = Smart Manager
Tools = Employees
MCP = Standard Operating Procedure

Process:

* Manager receives request.
* Manager assigns correct employee.
* Employee returns structured report.
* Manager presents final response.

The manager does NOT do accounting or coding himself.

He delegates.

That delegation framework = MCP.

---

# 9. Important Technical Concepts (Explained Simply)

## 9.1 JSON Schema

A formal way to define:

* What fields are required
* What type they are

Why it matters:
Prevents wrong input.

## 9.2 Async Programming

Used because:

* Tool calls may take time
* We don’t want to block system

Think of async as:
"Start task, continue working, return when done."

## 9.3 Tool Calling vs Prompting

Prompting:
Model guesses answer.

Tool Calling:
Model executes verified logic.

Tool calling increases reliability.

---

# 10. Real-World Use Cases

1. AI Financial Advisor

   * Calls stock API
   * Calculates portfolio risk

2. AI HR Assistant

   * Fetches employee records
   * Updates leave balance

3. AI Data Analyst

   * Queries database
   * Runs statistics
   * Generates report

4. AI Operations Manager

   * Checks inventory
   * Places restock order

All of this using the same MCP standard.

---

# 11. Common Student Questions

### Q: Why not let model do everything?

Because:

* Models hallucinate
* Models are not deterministic
* Models should not access secure systems directly

### Q: Is MCP only for Python?

No.
MCP is language-agnostic.

### Q: Is MCP required for all AI apps?

No.
But it is critical when:

* Tools are involved
* Production systems are used
* Enterprise reliability is required

---

# 12. Final Summary for Teaching

MCP is:

* A standard
* A communication protocol
* A structured delegation framework

It enables AI to:

* Discover tools
* Call tools
* Use structured outputs
* Operate safely in real-world systems

If you want students to remember one sentence:

"MCP allows AI to safely and reliably use external tools in a standardised way."

---

End of Notes
