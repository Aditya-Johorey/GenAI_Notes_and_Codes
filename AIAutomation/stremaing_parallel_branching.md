# LangChain Workflow Components Summary (Ollama)

These four concepts are fundamental building blocks in LangChain workflows.

---

# 1. RunnableLambda

## What is it?

A **`RunnableLambda`** converts a normal Python function into a LangChain **Runnable**, allowing it to become part of a workflow.

Use it whenever you need **Python logic** inside your workflow instead of an LLM.

Examples:

* Transform text
* Validate data
* Format output
* Perform calculations
* Extract information

---

## Workflow

```text
Input
  │
  ▼
RunnableLambda
  │
  ▼
Output
```

---

## Simple Example

```python
from langchain_core.runnables import RunnableLambda

uppercase = RunnableLambda(
    lambda text: text.upper()
)

print(uppercase.invoke("hello world"))
```

Output

```
HELLO WORLD
```

---

## Inside a Workflow

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_template(
    "Explain {topic}"
)

uppercase = RunnableLambda(
    lambda text: text.upper()
)

chain = (
    prompt
    | llm
    | StrOutputParser()
    | uppercase
)

print(chain.invoke({"topic": "Artificial Intelligence"}))
```

---

# 2. RunnableParallel

## What is it?

`RunnableParallel` executes **multiple independent workflows simultaneously**.

Each workflow receives the **same input**.

The outputs are combined into a dictionary.

---

## Workflow

```text
               Input
                 │
      ┌──────────┼──────────┐
      ▼          ▼          ▼
 Summary      Keywords     Quiz
      │          │          │
      ▼          ▼          ▼
     LLM        LLM        LLM
      └──────────┼──────────┘
                 ▼
          Dictionary Output
```

---

## Example

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

llm = ChatOllama(model="llama3.2")

summary_chain = (
    ChatPromptTemplate.from_template(
        "Summarize {topic}"
    )
    | llm
    | StrOutputParser()
)

keywords_chain = (
    ChatPromptTemplate.from_template(
        "List five keywords about {topic}"
    )
    | llm
    | StrOutputParser()
)

parallel = RunnableParallel(
    summary=summary_chain,
    keywords=keywords_chain
)

result = parallel.invoke({
    "topic": "Machine Learning"
})

print(result["summary"])
print(result["keywords"])
```

Output

```python
{
    "summary": "...",
    "keywords": "..."
}
```

---

# 3. RunnableBranch (Branching)

## What is it?

`RunnableBranch` chooses **one workflow** from multiple possible workflows based on conditions.

Think of it as Python's

```python
if
elif
else
```

inside a LangChain workflow.

---

## Workflow

```text
              User Input
                   │
                   ▼
            RunnableBranch
             /     |      \
            /      |       \
      Weather?   Math?   Default
          │         │        │
          ▼         ▼        ▼
     Weather     Calculator   LLM
```

Only **one** branch executes.

---

## Example

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(

    (
        lambda x: "weather" in x.lower(),
        lambda x: "Use Weather Tool"
    ),

    (
        lambda x: "math" in x.lower(),
        lambda x: "Use Calculator"
    ),

    lambda x: "Use Chat Model"

)

print(branch.invoke("Do some math"))
```

Output

```
Use Calculator
```

---

# Using Real Chains

```python
weather_chain = weather_prompt | llm

calculator_chain = calculator

chat_chain = chat_prompt | llm

router = RunnableBranch(

    (
        lambda x: "weather" in x.lower(),
        weather_chain
    ),

    (
        lambda x: "math" in x.lower(),
        calculator_chain
    ),

    chat_chain

)
```

---

# 4. Streaming

## What is it?

Streaming returns the model's output **piece by piece** instead of waiting for the entire response.

Perfect for chatbots.

---

## Without Streaming

```text
User

↓

LLM

↓

(wait...)

↓

Entire Response
```

---

## With Streaming

```text
User

↓

LLM

↓

Artificial...

↓

Intelligence...

↓

is...

↓

...
```

---

## Normal Invocation

```python
response = chain.invoke({
    "topic": "AI"
})

print(response)
```

Waits for the complete answer.

---

## Streaming

```python
for chunk in chain.stream({
    "topic": "AI"
}):
    print(chunk, end="", flush=True)
```

---

## Complete Example

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_template(
    "Explain {topic}"
)

chain = (
    prompt
    | llm
    | StrOutputParser()
)

for chunk in chain.stream({
    "topic": "Artificial Intelligence"
}):
    print(chunk, end="", flush=True)
```

---

# Comparison Table

| Component            | Purpose                                   | Executes     | Returns                   |
| -------------------- | ----------------------------------------- | ------------ | ------------------------- |
| **RunnableLambda**   | Add custom Python logic                   | One function | Any Python object         |
| **RunnableParallel** | Execute multiple workflows simultaneously | All branches | Dictionary                |
| **RunnableBranch**   | Choose one workflow based on conditions   | One branch   | Output of selected branch |
| **Streaming**        | Return output progressively               | One workflow | Chunks of output          |

---

# When to Use Each

### RunnableLambda

✅ String manipulation

✅ Validation

✅ Calculations

✅ Formatting

---

### RunnableParallel

✅ Summary + Keywords

✅ Blog + Title + Tags

✅ Multiple independent LLM tasks

---

### RunnableBranch

✅ Math vs Chat

✅ Weather vs FAQ

✅ Student vs Admin routing

---

### Streaming

✅ Chatbots

✅ AI Assistants

✅ Long responses

✅ Better user experience

---

# Mental Models

| Component        | Think of it as...                                                           |
| ---------------- | --------------------------------------------------------------------------- |
| RunnableLambda   | A **Python function** inside the workflow                                   |
| RunnableParallel | Multiple **workers** doing different tasks at the same time                 |
| RunnableBranch   | An **if / elif / else** statement that selects one path                     |
| Streaming        | Reading a **live typing response** instead of waiting for the whole message |

## The Big Picture

A realistic workflow might combine all four concepts:

```text
                     User Input
                          │
                          ▼
                   RunnableBranch
                  /               \
             Math Query       General Query
                │                  │
                ▼                  ▼
           Calculator         Prompt → LLM
                                   │
                                   ▼
                           RunnableParallel
                        ┌─────────┴─────────┐
                        ▼                   ▼
                    Summary            Keywords
                        │                   │
                        └─────────┬─────────┘
                                  ▼
                           RunnableLambda
                       (Format the final output)
                                  │
                                  ▼
                             stream()
                                  │
                                  ▼
                          User sees response
```

This demonstrates how:

* **`RunnableBranch`** routes the request.
* **`RunnableParallel`** performs independent tasks concurrently.
* **`RunnableLambda`** applies custom Python processing.
* **`stream()`** delivers the final result incrementally to the user.

These four building blocks form the foundation for more advanced LangChain applications such as RAG systems, tool-calling agents, and LangGraph workflows.
