# Day 2 – Prompt Engineering + Editing Existing Images

## Goal: Go from "generate" to "control"

---

# Introduction (2 minutes)

Yesterday you learned how AI creates images from text.

Today you'll learn something even more useful:

> **Professionals don't keep generating new images.**
> They edit the existing one until it becomes exactly what they want.

Think of AI as a designer sitting beside you.

Instead of saying

> "Make another image."

You say

> "Move the camera."
> "Make it brighter."
> "Change the clothes."
> "Remove the chair."

AI keeps improving the same image.

This saves time and gives much better results.

---

# Part 1 – Iterative Prompting (5 minutes)

## What is Iterative Prompting?

Instead of writing one giant perfect prompt…

You generate something first.

Then improve it step by step.

---

### Bad Workflow

Prompt:

> A cyberpunk girl standing in Tokyo during rain at night wearing futuristic clothing with neon reflections cinematic lighting highly detailed 8k masterpiece.

Image isn't perfect.

User:

> Generate again.

AI creates a completely different image.

Everything changes.

---

### Better Workflow

Prompt 1

> A young woman standing on a rainy Tokyo street at night.

↓

Prompt 2

> Add neon reflections on the wet road.

↓

Prompt 3

> Make the lighting cinematic.

↓

Prompt 4

> Change her jacket to black leather.

↓

Prompt 5

> Add purple glowing signs.

↓

Prompt 6

> Increase realism.

Each prompt builds upon the previous image.

You stay in control.

---

## Rule

**Don't restart. Refine.**

Professional prompting is editing.

Not gambling.

---

# Part 2 – Anatomy of Good Editing Prompts

Suppose AI generated this:

A man sitting inside a café.

Now edit only what you need.

Instead of

> Make another image

Say

> Keep everything exactly the same. Change the weather outside to snow.

or

> Keep the composition identical. Make it sunset.

or

> Replace the coffee mug with a laptop.

Notice how specific these edits are.

---

# Part 3 – Negative Prompts (10 minutes)

Sometimes telling AI what NOT to do is equally important.

These are called **Negative Prompts**.

---

## Example

Prompt

> Portrait of a businessman

Negative Prompt

> No text
>
> No watermark
>
> No blurry face
>
> No extra fingers
>
> No cropped hands
>
> No low quality

Result becomes much cleaner.

---

## Common Negative Prompt Words

Blurry

Low quality

Extra fingers

Extra arms

Bad anatomy

Watermark

Logo

Text

Noise

Distorted face

Duplicate objects

Crooked eyes

Overexposed

Underexposed

Cartoon (if realism is desired)

---

### Example

Prompt

> Hyper realistic lion walking through a forest

Negative Prompt

> blurry, watermark, logo, text, low quality, cartoon, extra legs

---

# Part 4 – Style References

Instead of describing every visual detail…

You can simply tell AI the style.

Examples:

Studio Ghibli style

Pixar style

Watercolor painting

Oil painting

Comic book

Anime

Photorealistic

Clay animation

Low poly

Minimalist illustration

Vintage photography

Film noir

Cyberpunk

Steampunk

Pixel art

---

### Example

Prompt

> A cat sleeping on a windowsill in Studio Ghibli style.

Now compare

> A cat sleeping on a windowsill in photorealistic DSLR photography.

Same subject.

Different style.

---

# Part 5 – Aspect Ratio Control

Different platforms require different image sizes.

| Platform          | Aspect Ratio |
| ----------------- | ------------ |
| Instagram Post    | 1:1          |
| Instagram Story   | 9:16         |
| YouTube Thumbnail | 16:9         |
| Phone Wallpaper   | 9:16         |
| Desktop Wallpaper | 16:9         |
| LinkedIn Banner   | Wide         |
| Facebook Cover    | Wide         |

Always mention the desired format when needed.

Example

> Create a YouTube thumbnail in 16:9.

or

> Create a vertical Instagram story (9:16).

---

# Part 6 – Conversational Image Editing (10 minutes)

Modern AI understands natural conversation.

You don't need complicated prompts.

Imagine this conversation.

---

User

Create a portrait of a woman wearing a red dress.

AI creates image.

---

User

Change the background to Paris.

AI edits only the background.

---

User

Make it sunset.

AI changes lighting.

---

User

Add soft golden light.

AI edits lighting.

---

User

Change her dress to blue.

AI edits clothes.

---

User

Remove the handbag.

AI removes it.

---

User

Smile slightly.

AI edits expression.

---

User

Increase realism.

Done.

Everything happens naturally.

---

## Editing Commands You Should Know

Change the background

Remove the object

Replace the object

Change clothing

Change hairstyle

Change facial expression

Make it realistic

Make it cinematic

Increase contrast

Blur background

Add fog

Add rain

Make it daytime

Make it nighttime

Add reflections

Add snow

Sharpen image

Remove people

Remove shadows

Change camera angle

Zoom in

Zoom out

Increase saturation

---

# Live Exercise

Generate

> A dog sitting in a park.

Then ask AI:

Change the weather to winter.

↓

Now

Add snowfall.

↓

Now

Make it nighttime.

↓

Now

Add glowing street lamps.

↓

Now

Make the dog wear a red scarf.

Notice:

You never regenerated.

You edited.

---

# Part 7 – Background Removal (5 minutes)

Sometimes you only need the subject.

Not the background.

Canva provides an easy workflow.

---

## Example

Original

Person standing in a messy room.

↓

Remove Background

↓

Transparent image

↓

Place onto

Beach

Office

Mountains

Studio backdrop

Gradient

Marketing poster

Thumbnail

Presentation

---

## Quick Cleanup in Canva

Use **Magic Eraser** to remove unwanted objects.

Examples:

Remove a trash can

Remove tourists

Remove wires

Remove poles

Remove stains

Remove unwanted shadows

---

## Magic Edit

Highlight an object.

Type

> Replace with laptop

or

> Replace with flowers

AI changes only that object.

---

# Best Practices

Always edit the same image instead of generating a new one.

Change one thing at a time.

Be specific.

Preserve everything else.

Specify lighting.

Specify style.

Specify framing.

Specify aspect ratio.

Use negative prompts when quality matters.

---

# Mini Challenge (5 minutes)

Create an image of:

> A modern workspace.

Then make these edits one by one:

1. Change day to night.

2. Add warm lighting.

3. Replace the laptop with a tablet.

4. Remove the coffee cup.

5. Add a window overlooking mountains.

6. Make the room minimalist.

7. Convert to photorealistic.

8. Resize to YouTube thumbnail (16:9).

---

# Key Takeaways

* **Think like an editor, not a generator.** Small, specific changes give you more control than starting over.
* **Refine one element at a time.** This makes it easier to see what each change does.
* **Use negative prompts** to reduce common image issues like blur, watermarks, or unwanted artifacts (where the AI tool supports them).
* **Specify style and aspect ratio** early if they matter for your final use.
* **Have a conversation with the AI.** Modern image models respond well to natural editing requests such as "make it sunset," "remove the chair," or "change the outfit."
* **Finish with cleanup tools** like Canva's Magic Studio when you need quick background removal or object cleanup.

### Habit to build

> **Edit in conversation—don't regenerate from scratch each time.** This workflow is faster, more consistent, and gives you much finer creative control.
