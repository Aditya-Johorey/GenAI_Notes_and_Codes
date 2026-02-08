## Prompt Engineering 📄
### Foundations of Prompt Engineering
**What is a prompt exactly?**
> A prompt is not just a command, it is instructions + context, wrapped with constraints and supported by examples that shape how the model predicts the next token.

LLMS actually dont:
* Understand the truth
* Read databases
* Reason properly

All they do is: **Predict the most probable next token given all previous tokens.**

Meaning:
* ❌ AI is not actually understanding your question via your prompt
* ✅ Your prompt is shaping the ***probability distribution*** of the next token.

**Setting up you first prompt (The proper way)**

The core fundamentals of a prompt generally involve:
  1. **Role:** It tells the AI who it should be. This shapes the:
     ```
     You are a Chartered Accountant.
     Explain mutual fund risk metrics to a beginner investor.

     ```
     Determines:
      * Tone of the model
      * Particular Vocabulary used by the model
      * Expertise of the model
     Parameters:
      * Risk tolerance
      * Depth of reasoning
      * Writing style
      * Use of caveats
  3. **Task:** It is the specific action you want the AI to perform. It needs to clear and direct.
  4. **Context:** Provides a background about why the task matters and who will be benefited by it. It gives the AI the necessary environment to understand the nuances of the task. Without context, the AI might hallucinate or provide generic advice that doesn't apply to your specific situation.
  5. **Constraints:** They tell the AI what **not** to do or specify strict limits it must follow. Constraints shrink the solution space. This saves you from having to edit out fluff or fix formatting later.

     ```
     Explain backpropagation in exactly 5 bullet points,
     each under 15 words, without formulas.

     ```

     Example Contraint Grounds:
     * Length constraints
     * Style constraints
     * Knowledge constraints
     * Ethical constraints
     * Format constraints

  7. **Output Format:** Defines how the information should look. This is the space where you define your output structure.
     ```
     ```
     Example Scenarios:
     * Tool calling
     * APIs
     * Automation pipelines (LangChain, n8n, agents)

Example Prompt:

***Role:*** You are an expert fitness coach. 
***Task:*** Create a 4-week workout routine. 
***Context:*** I am a beginner with no equipment who can only exercise for 20 minutes a day, three times a week. 
***Constraints:*** No high-impact movements (like jumping) because I have bad knees. 
***Output Format:*** Provide the routine in a bulleted list categorized by 'Week 1-2' and 'Week 3-4'.

---
### Categorising Prompts on the basis of Iterations:
  1. **Zero Shot Prompting:** Used for summarizing, translating, or general brainstorming where the "rules" are already well-understood by the AI.
  2. **One-Shot Prompting:** Exactly one exaple is given. Used for have a specific output style that the AI might not guess correctly on its own.
     
     ```
     "Convert the following sentence into a headline.

     Input: The local team won the championship after a 10-year drought. Output: CHAMPIONS AT LAST: 10-Year Wait Over for Local Heroes.

     Input: A new tech startup just raised $50 million in Series A funding. Output:"
     ```
     
  3. **Few Shot Prompting:** Providing multiple examples (usually between 2 and 5). By showing a pattern several times, you "ground" the AI in the logic or style you want it to emulate.

     ```
      Role: You are a Support Operations Specialist. Task: Classify the incoming customer ticket by Category and Priority (1-5).

      Context:
      
      Priority 5: System is down or users cannot log in.
      
      Priority 3: Feature isn't working as expected, but there is a workaround.
      
      Priority 1: General feedback or cosmetic UI suggestions.
      
      Examples:
      
      Input: "I love the app, but could you add a dark mode? My eyes hurt at night." Output: Category: Feature Request | Priority: 1 | Reason: Cosmetic improvement.
      
      Input: "The 'Export to PDF' button is grayed out for me today. I need this for a meeting in an hour, so I'm just copy-pasting for now." Output: Category: Functional Bug | Priority: 3 | Reason: Feature broken but workaround exists.
      
      Input: "Emergency! My entire team is getting a 404 error when trying to log into the dashboard. We can't do any work!" Output: Category: Critical Outage | Priority: 5 | Reason: Total loss of access for multiple users.
      
      New Task: Input: "Hi there, I’m trying to reset my password but the email link you sent me just leads to a blank white page. I've tried it three times and still can't get into my account." Output:
     ```

Summary: 

Method | Examples Provided | Best Use Case | Effort Level
---|---|---|---
Zero-shot | 0 | "General knowledge, simple instructions." | Low
One-shot | 1 | Setting a specific tone or basic format. | Medium
Few-shot | 2-5+ | "Complex patterns, logic, or niche formatting." | High

## Controlling Output Quality

1. Role Prompting (Persona Conditioning)
    Assigning an identity biases tone, vocabulary, and reasoning style.
   
    ```
    You are a Chartered Accountant.
    Explain mutual fund risk metrics to a beginner investor.
    ```
   
    This activates patterns associated with:
    * Finance language
    * Risk framing
    * Conservative advice
      
2. Constraints and Guardrails
   Constraints shrink the solution space.
   * Length constraints
   * Style constraints
   * Knowledge constraints
   * Ethical constraints
   * Format constraints
     
   ```
   Explain backpropagation in exactly 5 bullet points,
   each under 15 words, without formulas.
   ```
  
3. Output Formatting & Schema Control
   You can force machine-readable outputs.
   ```
   Return the answer in JSON with keys:
    "definition", "example", "common_mistake".
   ```

## Reasoning & Thinking Control
1. **Chain-of-Thought Prompting (Hidden vs Visible)**
   Complex problems benefit from intermediate reasoning steps.

   ```
   Solve step-by-step and explain your reasoning.
   ```

   This improves:
   * Accuracy
   * Logical consistency
   * Error detection

  In production, we often prefer:
  
  ```
  Solve internally but provide only the final answer.
  ```
  Because exposed chain-of-thought can:
  * Leak reasoning patterns
  * Be verbose
  * Be unreliable

2. **Task Decomposition & Stepwise Prompts**
   
   Instead of:
   ```Build a business plan for a robotics startup.```

   Use:
   ```
   Step 1: Ask me 5 clarifying questions.
   Step 2: Generate customer personas.
   Step 3: Create market analysis.
   Step 4: Draft business model canvas.
   ```

   Why this works
   * Reduces hallucinations
   * Improves structure
   * Allows human steering

3. **Self-Consistency Prompting**
   Ask the model to reason multiple ways and vote.

   ```
   Solve this problem using three different approaches,
   then select the most consistent result.
   ```

   This improves reasoning accuracy on:
   * Math
   * Logic
   * Multi-step planning

## Knowledge Reliability & Hallucination Control
1. **Hallucinations: Why They Happen**

   LLMs:
   * Optimize fluency, not truth
   * Prefer plausible completion over “I don’t know”
     
   Causes
   * Missing data
   * Ambiguous prompts
   * Overconfidence bias in training

2. **Reducing Hallucinations**

   * Source Grounding

     ```
     Answer only using the following document:
     <<<text>>>

     ```
   * Refusal Conditioning

     ```
     If information is missing, say "Insufficient data".
     ```

   * Citation Forcing

     ```
     Provide citations for each factual claim.
     ```

   * Confidence Calibration

     ```
     Estimate your confidence (0–100%) for each answer.
     ```
   
## Advanced Prompting Patterns

1. Retrieval-Augmented Prompting (RAG)
   Instead of relying on model memory, inject external documents into the prompt.

   ```
   System: Use only the context below.
   Context: [retrieved docs]
   User: Answer the question.
   ```

   This turns LLM into:
   > A reasoning engine over provided data
   
   Instead of:
   > A knowledge oracle
   
3. Tool Use Prompting (Function Calling / Agents)
   Model decides when to call tools.

   ```
   If calculation is required, call calculator().
   If web info is required, call search().
   Otherwise, answer directly.
   ```

   This enables:
   * Autonomous agents
   * Workflow automation
   * Code execution
   
5. Multi-Agent Prompting
   Different roles debate:

   ```
   You are three experts:
   1. Optimist
   2. Skeptic
   3. Engineer

   Debate the feasibility of humanoid robots in factories.
   Then produce a consensus report.
   ```

   This improves:
   * Coverage
   * Risk analysis
   * Creative solutions
    
7. Prompt Chaining & Memory
   Outputs from one prompt feed another:

   ```
   Prompt 1 → Outline
   Prompt 2 → Expand
   Prompt 3 → Critique
   Prompt 4 → Final polish
   ```

   This is how:
   * Long documents
   * Reports
   * Software specs
   * are generated reliably.
