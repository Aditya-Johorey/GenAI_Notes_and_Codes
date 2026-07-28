# LangChain Workflows with Ollama

## Table of Contents
1. Introduction
2. Why LangChain?
3. LLM Fundamentals
4. LangChain Architecture
5. Core Concepts
6. Installing Ollama
7. First Chain
8. Output Parsers
9. RunnableLambda
10. Prompt Chaining
11. Parallel Workflows
12. Branching
13. Streaming
14. Structured Output
15. Tool Calling
16. Chat History
17. Mini Project
18. Exercises
19. Best Practices

# 1. Introduction

LangChain is a framework for building applications powered by Large Language Models (LLMs). It helps you connect prompts, models, tools, memory, and external data into reusable workflows.

Think of LangChain like Lego blocks:
- Prompt
- Model
- Parser
- Tool
- Memory

Each block does one job.

# 2. Why LangChain?

Without LangChain you write lots of glue code.

Typical application:

User -> Prompt -> Model -> Parse -> Tool -> Store -> Response

LangChain provides reusable building blocks for this pipeline.

# 3. LLM Fundamentals

## Tokens
LLMs read tokens, not characters.

## Context Window
The model only knows what is inside the current prompt.

## Prompt
Instructions given to the model.

## Temperature
Higher = more creative.
Lower = more deterministic.

# 4. LangChain Architecture

User
↓
Prompt
↓
Model
↓
Output Parser
↓
Application

Everything is a Runnable.

# 5. Core Concepts

## Chat Models

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)
```

## Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} simply."
)
```

## LCEL

The | operator connects workflow steps.

```python
chain = prompt | llm
```

# 6. Installing

```bash
pip install langchain langchain-core langchain-ollama
ollama pull llama3.2
```

# 7. First Workflow

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_template(
    "Explain {topic}"
)

chain = prompt | llm

response = chain.invoke({"topic":"Robotics"})
print(response.content)
```

Flow

Input
↓
Prompt
↓
Model
↓
Answer

# 8. Output Parser

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic":"AI"}))
```

# 9. RunnableLambda

```python
from langchain_core.runnables import RunnableLambda

uppercase = RunnableLambda(lambda x: x.upper())

print(uppercase.invoke("hello"))
```

Any Python function can become a workflow step.

# 10. Prompt Chaining

```python
idea = ChatPromptTemplate.from_template(
    "Generate an idea about {topic}"
)

improve = ChatPromptTemplate.from_template(
    "Improve this:\n{idea}"
)

chain = (
    {"idea": idea | llm | StrOutputParser()}
    | improve
    | llm
    | StrOutputParser()
)
```

# 11. Parallel Workflows

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=prompt | llm | StrOutputParser(),
    keywords=prompt | llm | StrOutputParser()
)
```

# 12. Branching

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "math" in x.lower(), lambda x: "Calculator"),
    (lambda x: "weather" in x.lower(), lambda x: "Weather Tool"),
    lambda x: "LLM"
)
```

# 13. Streaming

```python
for chunk in chain.stream({"topic":"Space"}):
    print(chunk, end="")
```

# 14. Structured Output

```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float
```

# 15. Tool Calling

```python
from langchain_core.tools import tool

@tool
def multiply(a:int,b:int)->int:
    \"\"\"Multiply two numbers.\"\"\"
    return a*b

llm_tools = llm.bind_tools([multiply])
```

# 16. Chat History

```python
from langchain_core.messages import HumanMessage, AIMessage

history=[
    HumanMessage(content="Hi"),
    AIMessage(content="Hello")
]

response=llm.invoke(history)
```

# 17. Mini Project

Build a Study Assistant:

User
↓
Prompt
↓
LLM
↓
Parser
↓
Response

Extension ideas:
- Add memory
- Add search
- Add PDF loader
- Add calculator tool

# 18. Exercises

1. Build a joke generator.
2. Build a recipe assistant.
3. Add a RunnableLambda.
4. Add parallel execution.
5. Add branching.

# 19. Best Practices

- Keep prompts focused.
- Separate workflow steps.
- Parse outputs.
- Test each runnable independently.
- Use low temperature for factual tasks.
- Prefer small reusable chains.

## Roadmap

Python
→ LangChain
→ LCEL
→ Tools
→ Memory
→ RAG
→ LangGraph
→ AI Agents
