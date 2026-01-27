# Full Mathematical Walkthrough of the Attention Mechanism

> A complete, step-by-step numerical and mathematical explanation of how self-attention works — from embeddings to final output — including **why** each step exists and **what breaks if removed**.

---

## 📌 Table of Contents

1. [Problem Setup](#problem-setup)
2. [Step 0 — Tokenization](#step-0--tokenization)
3. [Step 1 — Embeddings](#step-1--embeddings)
4. [Step 2 — Projection Matrices (W_Q, W_K, W_V)](#step-2--projection-matrices-w_q-w_k-w_v)
5. [Step 3 — Computing Q, K, and V](#step-3--computing-q-k-and-v)
6. [Step 4 — Attention Scores (QKᵀ)](#step-4--attention-scores-qk)
7. [Step 5 — Scaling by √dₖ](#step-5--scaling-by-dₖ)
8. [Step 6 — Softmax](#step-6--softmax)
9. [Step 7 — Weighted Sum of Values](#step-7--weighted-sum-of-values)
10. [Step 8 — Full Matrix Form](#step-8--full-matrix-form)
11. [Why Each Step Exists](#why-each-step-exists)
12. [What Breaks If a Step Is Removed](#what-breaks-if-a-step-is-removed)
13. [Connection to Multi-Head Attention](#connection-to-multi-head-attention)
14. [Summary](#summary)

---

## Problem Setup

We will compute **self-attention** on the real sentence:

> **"I love robotics"**

We will calculate the **contextual embedding for the token `"love"`**.

We assume:

* Vocabulary already tokenized
* Embedding dimension = **4**
* Attention projection dimension = **3**

---

## Step 0 — Tokenization

```
["I", "love", "robotics"]
```

🔍 **Why:** Neural networks operate on discrete units, not raw text.

---

## Step 1 — Embeddings

Each token is mapped to a vector in ℝ⁴.

```
x_I      = [1, 0, 1, 0]
x_love   = [1, 1, 0, 0]
x_robot  = [0, 1, 1, 0]
```

Let the input matrix be:

```
X = [
  [1, 0, 1, 0],
  [1, 1, 0, 0],
  [0, 1, 1, 0]
]
```

🔍 **Why:**

* Converts discrete tokens into continuous vectors
* Allows gradient-based learning

❌ Without embeddings → no numerical representation → no learning

---

## Step 2 — Projection Matrices (W_Q, W_K, W_V)

We introduce three learned matrices:

```
W_Q ∈ ℝ^{4×3}
W_K ∈ ℝ^{4×3}
W_V ∈ ℝ^{4×3}
```

Assume the learned values are:

### Query Projection

```
W_Q =
[[1, 0, 1],
 [0, 1, 0],
 [1, 0, 0],
 [0, 0, 1]]
```

### Key Projection

```
W_K =
[[1, 1, 0],
 [0, 1, 1],
 [1, 0, 1],
 [0, 1, 0]]
```

### Value Projection

```
W_V =
[[1, 0, 0],
 [0, 1, 0],
 [1, 0, 1],
 [0, 0, 1]]
```

🔍 **Why:**

* Same token must behave differently as:

  * a seeker (Q)
  * a matcher (K)
  * an information carrier (V)

❌ Without these projections → attention collapses into rigid similarity lookup

---

## Step 3 — Computing Q, K, and V

We multiply each embedding by each projection matrix.

### General Formula

```
Q = X W_Q
K = X W_K
V = X W_V
```

Each output vector is in ℝ³.

---

### 3.1 Query for "love"

```
q_love = x_love · W_Q
       = [1,1,0,0]
         [[1,0,1],
          [0,1,0],
          [1,0,0],
          [0,0,1]]
```

Compute each component:

```
q₁ = 1·1 + 1·0 + 0·1 + 0·0 = 1
q₂ = 1·0 + 1·1 + 0·0 + 0·0 = 1
q₃ = 1·1 + 1·0 + 0·0 + 0·1 = 1
```

```
q_love = [1, 1, 1]
```

---

### 3.2 Keys

#### For "I"

```
k_I = [1,0,1,0] · W_K
```

```
k₁ = 1·1 + 0·0 + 1·1 + 0·0 = 2
k₂ = 1·1 + 0·1 + 1·0 + 0·1 = 1
k₃ = 1·0 + 0·1 + 1·1 + 0·0 = 1
```

```
k_I = [2, 1, 1]
```

---

#### For "love"

```
k_love = [1,1,0,0] · W_K
```

```
k₁ = 1·1 + 1·0 + 0·1 + 0·0 = 1
k₂ = 1·1 + 1·1 + 0·0 + 0·1 = 2
k₃ = 1·0 + 1·1 + 0·1 + 0·0 = 1
```

```
k_love = [1, 2, 1]
```

---

#### For "robotics"

```
k_robot = [0,1,1,0] · W_K
```

```
k₁ = 0·1 + 1·0 + 1·1 + 0·0 = 1
k₂ = 0·1 + 1·1 + 1·0 + 0·1 = 1
k₃ = 0·0 + 1·1 + 1·1 + 0·0 = 2
```

```
k_robot = [1, 1, 2]
```

---

### 3.3 Values

#### For "I"

```
v_I = [1,0,1,0] · W_V
```

```
v₁ = 1·1 + 0·0 + 1·1 + 0·0 = 2
v₂ = 1·0 + 0·1 + 1·0 + 0·0 = 0
v₃ = 1·0 + 0·0 + 1·1 + 0·1 = 1
```

```
v_I = [2, 0, 1]
```

---

#### For "love"

```
v_love = [1,1,0,0] · W_V
```

```
v₁ = 1·1 + 1·0 + 0·1 + 0·0 = 1
v₂ = 1·0 + 1·1 + 0·0 + 0·0 = 1
v₃ = 1·0 + 1·0 + 0·1 + 0·1 = 0
```

```
v_love = [1, 1, 0]
```

---

#### For "robotics"

```
v_robot = [0,1,1,0] · W_V
```

```
v₁ = 0·1 + 1·0 + 1·1 + 0·0 = 1
v₂ = 0·0 + 1·1 + 1·0 + 0·0 = 1
v₃ = 0·0 + 1·0 + 1·1 + 0·1 = 1
```

```
v_robot = [1, 1, 1]
```

---

## Step 4 — Attention Scores (QKᵀ)

We compute dot products between `q_love` and each key.

```
q_love = [1,1,1]
```

### Against "I"

```
s_I = q_love · k_I
    = [1,1,1] · [2,1,1]
    = 2 + 1 + 1
    = 4
```

### Against "love"

```
s_love = [1,1,1] · [1,2,1]
       = 1 + 2 + 1
       = 4
```

### Against "robotics"

```
s_robot = [1,1,1] · [1,1,2]
        = 1 + 1 + 2
        = 4
```

Raw scores:

```
[4, 4, 4]
```

🔍 **Why:**

* Dot product measures alignment between query and keys
* High score → high relevance

❌ Without similarity → attention has no selection mechanism

---

## Step 5 — Scaling by √dₖ

Key dimension = 3 → √3 ≈ 1.732

```
scaled_scores = [4/1.732, 4/1.732, 4/1.732]
              ≈ [2.31, 2.31, 2.31]
```

🔍 **Why:**

* Prevents dot products from growing too large
* Keeps softmax gradients stable

❌ Without scaling → softmax saturation → vanishing gradients

---

## Step 6 — Softmax

```
softmax([2.31, 2.31, 2.31])
```

Since all values are equal:

```
= [1/3, 1/3, 1/3]
≈ [0.333, 0.333, 0.333]
```

🔍 **Why:**

* Converts scores into probabilities
* Normalizes weights to sum to 1
* Enables smooth interpolation

❌ Without softmax → unstable magnitudes and no probabilistic meaning

---

## Step 7 — Weighted Sum of Values

We combine values using attention weights:

```
output_love =
0.333·v_I + 0.333·v_love + 0.333·v_robot
```

Substitute:

```
= 0.333·[2,0,1]
+ 0.333·[1,1,0]
+ 0.333·[1,1,1]
```

```
= [0.666,0,0.333]
+ [0.333,0.333,0]
+ [0.333,0.333,0.333]
```

```
= [1.332, 0.666, 0.666]
```

✅ This is the **new contextual embedding for "love"**.

🔍 **Why:**

* Combines information from all tokens
* More relevant tokens contribute more
* Produces smooth, differentiable routing

❌ Without weighted sum → no information aggregation

---

## Step 8 — Full Matrix Form

Let:

```
Q = X W_Q
K = X W_K
V = X W_V
```

Then attention is:

```
Attention(X) = softmax(QKᵀ / √dₖ) V
```

This computes all token outputs simultaneously.

---

## Why Each Step Exists

| Step          | Purpose                             |
| ------------- | ----------------------------------- |
| Embeddings    | Convert tokens to vectors           |
| W_Q, W_K, W_V | Learn role-specific representations |
| Dot product   | Measure relevance                   |
| Scaling       | Stabilize gradients                 |
| Softmax       | Normalize importance                |
| Weighted sum  | Aggregate information               |

---

## What Breaks If a Step Is Removed

| Removed Component | Consequence              |
| ----------------- | ------------------------ |
| Embeddings        | No numerical learning    |
| Q/K/V projections | Attention becomes rigid  |
| Dot product       | No relevance computation |
| Scaling           | Gradient instability     |
| Softmax           | No normalization         |
| Weighted sum      | No information routing   |

---

## Connection to Multi-Head Attention

If we use `h` heads:

```
d_model = 512
d_k = d_v = 512 / h
```

Each head has its own:

```
W_Q^i, W_K^i, W_V^i
```

Outputs from all heads are concatenated and projected again.

This allows the model to attend to **multiple relationships simultaneously**.

---

## Summary

| Concept     | Meaning                            |
| ----------- | ---------------------------------- |
| Q           | What am I looking for?             |
| K           | What do I advertise?               |
| V           | What information do I send?        |
| Attention   | Differentiable information routing |
| Transformer | Stack of attention blocks          |

---

## 🧠 Final One-Line Insight

> **Attention works because tokens dynamically decide where to look and what to extract — using pure linear algebra.**

---

If you want, I can:

* Add multi-head numerical walkthrough
* Add causal masking math
* Add encoder-decoder attention
* Add backpropagation gradients
* Add PyTorch code mapping to math

Just ask 😄
