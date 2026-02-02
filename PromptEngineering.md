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
