# AI Video Generation: From Prompt to Production

A lecture for students of AI from mixed backgrounds. Level: basic to intermediate. Suggested length: 3 hours plus a lab.

## Learning Outcomes

By the end of this lecture, students will be able to:

- Explain in simple terms how AI video models work and why they sometimes fail.
- Choose the right generation mode (text-to-video, image-to-video, video-to-video) for a task.
- Write structured prompts that control subject, motion, camera, lighting, style and sound.
- Keep characters and style consistent across several shots.
- Plan a short multi-shot video and apply it to their own field.
- Use AI video responsibly, with attention to consent, copyright and disclosure.

A note on tools: products in this field change every few months. This lecture teaches principles and prompt patterns that transfer between tools. Example tools are named only as categories (commercial cloud tools, open-source models, editing suites).

## Module 1: How AI Video Generation Works (20 min)

**Core idea.** Most modern video generators are diffusion models (often with transformer architectures). They start from random noise and gradually remove it, guided by your text or image, until a sequence of frames appears. Unlike an image model, a video model must also keep things consistent over time.

**Key vocabulary**

- **Frame / FPS:** a video is a stack of images; 24 fps is the cinematic standard.
- **Temporal consistency:** objects keeping the same shape, colour and identity from frame to frame.
- **Latent space:** the compressed representation the model actually works in.
- **Seed:** the number that sets the starting noise; the same seed with the same settings gives similar results.
- **Conditioning:** the information that steers generation (text, image, reference video, audio).
- **Prompt adherence:** how closely the output follows your instructions.

**Why this matters for prompting.** The model does not understand physics or stories; it has learned statistical patterns of how the world looks when it moves. So clear, concrete, visual descriptions work better than abstract ideas.

**Discussion question:** Why is a 5-second clip much easier for a model than a 60-second one?

## Module 2: The Generation Modes (20 min)

| Mode | Input | Best for | Weakness |
|---|---|---|---|
| Text-to-video | Text prompt | Exploring ideas, b-roll, concepts | Least control over exact look |
| Image-to-video | Still image + motion prompt | Brand visuals, consistent characters, animating artwork | Motion limited by the image |
| First/last frame | Two images | Transitions, morphs, controlled endings | Needs two good frames |
| Video-to-video | Source clip + prompt | Restyling, changing a scene, VFX | Can flicker or drift |
| Reference-based | Character or object images | Consistency across shots | Tool-dependent support |
| Audio-enabled | Prompt includes sound or dialogue | Ads, social clips, quick demos | Lip-sync and voices vary |

**Rule of thumb:** the more control you need, the more you should start from an image rather than from text alone.

## Module 3: Anatomy of a Good Prompt (30 min)

Use this formula. You do not need every element every time, but thinking through each one avoids vague results.

**Subject + Action + Setting + Camera + Lighting + Style + Mood + Sound**

### Demonstration: from weak to strong

**Weak prompt**

> A dog running.

Problem: no breed, place, camera or look. The model fills every gap with a generic guess.

**Improved prompt**

> A golden retriever sprinting along a wet sandy beach at sunrise, ears flapping, water splashing from its paws. Low tracking shot at ground level moving alongside the dog. Warm golden-hour backlight, shallow depth of field, cinematic, realistic.

**Why it works:** subject (golden retriever), action (sprinting, splashing), setting (wet beach, sunrise), camera (low tracking shot), lighting (golden backlight), style (cinematic, realistic).

### Prompt-writing principles

1. **One main action per clip.** Short clips handle one clear event well.
2. **Be specific and visual.** Say 'a worn brown leather jacket', not 'a cool jacket'.
3. **Describe what you want, not what you do not want.** Phrases like 'no people' sometimes backfire; write 'an empty street' instead.
4. **Order matters.** Put the most important information first.
5. **Keep it to roughly 40 to 120 words.** Too short is vague; too long gets partly ignored.

**Exercise (5 min):** Rewrite 'A city at night' into a full six-element prompt.

## Module 4: Camera Language (25 min)

Camera direction is the cheapest way to make AI video look professional.

**Shot sizes:** extreme wide, wide, medium, close-up, extreme close-up.

**Camera movements:** static, pan (left/right), tilt (up/down), dolly in/out, tracking (follow the subject), orbit, crane or drone shot, handheld, push-in, whip pan.

**Lens and focus:** wide-angle (24mm) feels expansive; 85mm portrait lens compresses and flatters; shallow depth of field blurs the background; macro shows tiny detail.

### Demonstration: same scene, three cameras

**Prompt A (static wide)**

> Static wide shot of a lone lighthouse on a rocky cliff, stormy sea below, waves crashing, grey overcast sky. Tripod-steady, cinematic.

**Prompt B (slow push-in)**

> Slow dolly-in toward a lone lighthouse on a rocky cliff, stormy sea below, waves crashing, grey overcast sky. Steady, cinematic.

**Prompt C (drone orbit)**

> Aerial drone shot slowly orbiting a lone lighthouse on a rocky cliff at dusk, stormy sea below, waves crashing against the rocks, moody blue-grey light.

**Teaching point:** the scene is identical; only the camera sentence changes, and so does the emotion. Ask students to describe how each version feels.

### Deep Dive: Why Camera Language Works

Video models were trained on huge amounts of film, advertising, stock footage and online video, each labelled with descriptions. Words like 'close-up', 'dolly-in' or '85mm' appear in those descriptions, so the model has learned to link each word with a particular look and movement. When you use these terms you are speaking the model's native vocabulary. When you leave the camera out, the model picks the most average option, usually a medium shot with a slow, generic drift.

Think of yourself as three people at once: the **director** (what the shot should feel like), the **cinematographer** (where the camera is, how it moves, which lens) and the **editor** (how this shot will connect to the next). Every camera decision below belongs to the cinematographer's job.

### Shot Sizes in Detail

Shot size is how much of the subject and world fits in the frame. It controls intimacy and information.

| Shot size | What is in frame | What it communicates | Use it for | Prompt wording |
|---|---|---|---|---|
| Extreme wide / establishing | Whole landscape or city, subject tiny | Scale, isolation, place | Opening a scene, showing where we are | 'extreme wide establishing shot of a desert highway, a single car tiny in the frame' |
| Wide / full shot | Whole body plus surroundings | Context, relationship to the setting | Action, walking, dance | 'wide shot of a chef moving through a busy kitchen' |
| Medium shot | Waist up | Conversation, body language | Dialogue, presenters, most social clips | 'medium shot of a presenter speaking to camera' |
| Close-up | Face or one object | Emotion, importance | Reactions, products, key details | 'close-up of her eyes as a tear forms' |
| Extreme close-up / macro | One tiny detail | Intensity, texture, curiosity | Food, jewellery, nature, tension | 'extreme macro shot of a dewdrop sliding down a leaf' |

**Concept: shot size sets how hard the model has to work.** In a wide shot, faces are small and imperfections are hidden. In a close-up, every flaw on a face or hand is visible. If your results look uncanny, move the camera back.

**Demonstration: one subject, three sizes**

> Extreme wide shot of a lone astronaut walking across a red desert, huge dunes, small figure, long shadow, dusty orange sky.

> Medium shot of the astronaut walking toward the camera across the red desert, dust blowing around the boots, orange sky behind.

> Close-up of the astronaut's helmet visor reflecting the red dunes and a distant sun, soft dust particles drifting past the lens.

The story grows more personal with each step: first we see loneliness, then effort, then a human inside the suit.

### Camera Angles

Angle is the height and tilt of the camera relative to the subject.

| Angle | Effect | Typical use |
|---|---|---|
| Eye level | Neutral, natural, relatable | Interviews, everyday scenes |
| Low angle (camera looks up) | Subject feels powerful, tall, heroic | Heroes, buildings, products that should feel premium |
| High angle (camera looks down) | Subject feels small, vulnerable | Showing isolation, overview of action |
| Overhead / top-down | Pattern, layout, abstract look | Food prep, maps, dance formations |
| Dutch angle (tilted horizon) | Unease, tension, disorientation | Thrillers, dream scenes |
| Over-the-shoulder | Connection between two people | Conversations |
| POV (first person) | Immersion, we see what the character sees | Games, sports, horror |

**Demonstration**

> Low-angle shot looking up at a young chess champion standing at a table, dramatic overhead light, confident posture, dark background, cinematic.

> High-angle shot looking down at the same champion small at a huge table in an empty hall, cold light, quiet mood.

Same person, opposite feelings, because only the angle changed.

### Camera Movements in Detail

A movement tells the model how the viewpoint changes during the clip. Be clear about whether the **camera** is moving or the **subject** is moving. Beginners often mix these up.

| Movement | What the camera does | Feeling | Prompt wording | Watch out for |
|---|---|---|---|---|
| Static / locked-off | Does not move | Calm, observational, stable | 'static shot on a tripod' | Model may add small drift; say 'completely still' |
| Pan | Rotates left or right on a fixed spot | Revealing, following with the eyes | 'slow pan left across the skyline' | Fast pans cause blur and warping |
| Tilt | Rotates up or down on a fixed spot | Revealing height or a hidden detail | 'slow tilt up from the boots to the face' | Keep it slow |
| Dolly in / push-in | Camera physically moves toward the subject | Focus, tension, emotional emphasis | 'slow dolly-in toward the old man's face' | Background should shift naturally |
| Dolly out / pull-back | Camera moves away | Reveal, isolation, ending a scene | 'slow pull-back revealing the empty stadium' | Good for surprise reveals |
| Tracking / follow | Camera moves alongside or behind the subject | Energy, journey, engagement | 'tracking shot following a cyclist from the side' | Specify side, behind or in front |
| Truck / lateral move | Camera slides sideways | Exploring a scene, showing a line of items | 'camera trucks right past a row of shop windows' | Parallax makes it feel 3D |
| Orbit / arc | Camera circles the subject | Showcase, importance, product display | 'camera slowly orbits the sculpture, 180 degrees' | Models may change the subject's back side; keep orbits partial |
| Crane / jib | Camera rises or descends | Grand reveal, start or end of scene | 'crane shot rising above the crowd' | Combine with a wide shot |
| Drone / aerial | Flies through or above a landscape | Scale, beauty, travel | 'aerial drone shot flying low over a forest canopy' | Add direction: forward, backward, ascending |
| Handheld | Slight natural shake | Realism, urgency, documentary | 'handheld camera, subtle shake' | Too much shake becomes nausea; say 'subtle' |
| Whip pan | Very fast pan | Energy, transitions | 'whip pan to the right' | Often unreliable; use in editing instead |
| Zoom | Lens changes focal length; camera stays in place | Dramatic, retro or surveillance feel | 'slow zoom in on the window' | Different from a dolly: perspective does not change |

**Concept: dolly versus zoom.** In a dolly, the camera moves forward and the background slides past, so you feel depth. In a zoom, the camera stays put and the picture just gets bigger, which looks flatter. Use dolly for emotional pull and zoom for a more documentary or vintage look.

**Concept: camera motion versus subject motion.** Always state both separately.

> Camera is static. A fisherman casts a net from a wooden boat on a calm lake at dawn.

> Camera slowly tracks sideways. A fisherman stands still in a wooden boat on a calm lake at dawn.

In the first prompt the subject moves and the camera does not. In the second it is reversed.

### Lens, Aperture and Focus

Lens terms change how space looks, even though no real lens exists.

| Term | Look | Best for |
|---|---|---|
| Wide-angle (16 to 28mm) | Expansive, exaggerated depth, edges stretch | Landscapes, interiors, action in tight spaces |
| Standard (35 to 50mm) | Natural, human-eye view | Documentary, street scenes |
| Portrait / telephoto (85 to 200mm) | Compressed background, flattering faces, strong blur | Portraits, sports, product close-ups |
| Macro | Extreme detail of tiny things | Insects, food, textures |
| Fisheye | Strong bubble distortion | Skate videos, music videos |
| Anamorphic | Wide cinematic frame, oval light flares | Film-like look |

**Depth of field** is how much of the picture is sharp. 'Shallow depth of field' means the subject is sharp and the background is creamy and blurred, which looks professional and hides messy backgrounds. 'Deep focus' keeps everything sharp, which suits landscapes and architecture.

**Rack focus** shifts sharpness from one object to another during the shot, for example from a flower in the foreground to a person behind it. It can be unreliable, so test it.

**Demonstration: lens changes the mood**

> 24mm wide-angle shot of a girl running down a narrow alley, walls close on both sides, deep focus, handheld.

> 135mm telephoto shot of the same girl running down the alley, background compressed and blurred, shallow depth of field, tracking from the front.

The first feels chaotic and close; the second feels isolated and dreamy.

### Speed, Smoothness and Strength of Movement

Camera moves have a second layer: how fast and how steady.

- **Speed words:** slow, gentle, gradual, steady, quick, rapid. If you do not say 'slow', many models move too fast, which creates warping.
- **Stability words:** tripod, gimbal-smooth, steady, handheld, shaky.
- **Distance words:** 'slight push-in' (small), 'dramatic push-in' (large).
- **Direction words:** left, right, forward, backward, upward, downward, clockwise.

A reliable default for most clips is **'slow, smooth, steady [movement]'**.

### Combining Camera Moves

Tools differ in how many movements they follow at once. A safe rule: **one primary movement per clip**, plus at most one gentle secondary movement.

- Good: 'slow dolly-in while tilting slightly upward'.
- Risky: 'fast orbit, zoom in, tilt down and handheld shake' (the model will drop some or produce confusing motion).

If you need a complex move, split it into two shots and join them in editing.

### Worked Example: Building a Shot Step by Step

**Version 1: no direction**

> A chef cooking.

Result: generic cook, random angle, random kitchen.

**Version 2: add subject, action, setting**

> A chef in a white jacket flips vegetables in a flaming wok in a busy restaurant kitchen.

Better content, but camera still undefined.

**Version 3: add shot size and angle**

> Medium close-up, slightly low angle, of a chef in a white jacket flipping vegetables in a flaming wok in a busy restaurant kitchen.

Now the framing feels dynamic and the flames look bigger.

**Version 4: add movement, lens and light**

> Medium close-up, slightly low angle, of a chef in a white jacket flipping vegetables in a flaming wok in a busy restaurant kitchen. Camera slowly pushes in. 50mm lens, shallow depth of field, orange firelight on his face, steam drifting, cinematic.

This version controls everything the viewer feels: how close we are, how we are positioned, what moves and how the light looks. Each step added one layer, so students can see exactly which words did what.

### Common Camera Prompt Mistakes

| Mistake | Why it fails | Better version |
|---|---|---|
| 'Cool camera movement' | Too vague; model guesses | 'slow orbit around the subject' |
| Several fast moves at once | Models blend or drop moves | One main move, slow |
| Camera instruction at the end of a long prompt | Gets ignored | Put it in the first or second sentence |
| Confusing subject movement with camera movement | Wrong thing moves | 'Camera static; the subject walks left to right' |
| Using film jargon the model may not know ('Steadicam through a Dutch roll') | Unreliable | Use common terms |
| Full 360-degree orbit | Back of subject is invented and may change | Use a partial arc of 90 to 180 degrees |

### Practice for Module 4

1. Take the sentence 'A man sits on a bench in a park.' Write three versions: a wide static shot, a close-up with a slow push-in, and a high-angle tracking shot. Predict the mood of each before generating.
2. Write one prompt that should give a heroic feeling and one that should give a vulnerable feeling, using only camera angle and shot size changes.
3. Generate the same prompt twice, once with 'dolly-in' and once with 'zoom in', and describe the difference in background behaviour.

## Module 5: Motion, Speed and Physics (20 min)

Motion is where AI video most often breaks. Control it deliberately.

- **Name the motion verb:** drifting, sprinting, swirling, tumbling, flickering.
- **Name the speed:** slow motion, time-lapse, real-time.
- **Anchor secondary motion:** hair, cloth, smoke, water, leaves.
- **Keep physics simple:** complex collisions, many interacting objects and fast limb movement are the usual failure points.

### Demonstration: vague vs controlled motion

**Vague**

> A woman dancing in the rain.

**Controlled**

> A woman in a yellow raincoat slowly twirls once in an empty city plaza, rain falling steadily, her coat flaring outward. Slow motion, medium shot, camera gently circling her, wet pavement reflecting neon signs.

**Time-lapse example**

> Time-lapse of a cloudy sky over a mountain valley from dawn to sunrise, clouds rolling through, light slowly turning from deep blue to orange. Static wide shot.

### Deep Dive: Why Motion Is the Hardest Part

An image model only has to get one picture right. A video model has to get dozens of pictures right and also make them agree with each other. Three problems follow from this:

1. **Temporal coherence.** A person's face, a shirt pattern or a car's shape must stay the same from the first frame to the last. The longer and more complex the motion, the more small errors pile up and objects 'melt' or change.
2. **Learned, not simulated, physics.** The model has never been taught gravity or friction. It has seen millions of examples of how things usually move and imitates them. Common motions (walking, waves, clouds, smoke) look convincing; rare or tightly coordinated ones (juggling, gymnastics, hands tying a knot) often fail.
3. **A limited motion budget.** Within a short clip there is only so much change the model can keep stable. One clear action is easy; five simultaneous actions overload it.

**The practical lesson:** you are not asking the model to film reality, you are asking it to produce the most likely-looking version of the motion you describe. Your job is to describe motion it has seen often, clearly and in small amounts.

### The Three Layers of Motion

Every clip contains three kinds of movement. Describe each one on purpose, or the model will invent it.

| Layer | What it is | Examples | If you ignore it |
|---|---|---|---|
| Subject motion | What the main character or object does | Walking, turning, pouring, spinning | Subject stands still or does random gestures |
| Environment motion | What the world around the subject does | Rain, wind in trees, traffic, steam, flags, crowds | Scene looks frozen like a photograph |
| Camera motion | How the viewpoint moves (Module 4) | Push-in, pan, orbit | Default slow drift, or the camera moves unpredictably |

**Demonstration: layering motion**

**Subject only**

> A lone traveller in a long coat walks along a mountain ridge.

**Subject plus environment**

> A lone traveller in a long coat walks steadily along a mountain ridge. Strong wind pulls at the coat and scarf, low clouds roll past, loose snow blows across the path.

**All three layers**

> A lone traveller in a long coat walks steadily along a mountain ridge, away from the camera. Strong wind pulls at the coat and scarf, low clouds roll past, loose snow blows across the path. Camera tracks slowly behind the traveller at shoulder height. Wide shot, cold blue light, cinematic.

The third version feels alive because everything that could move has a reason and a direction.

### Choosing Strong Motion Verbs

The verb is the single most powerful word in a motion prompt. Generic verbs give generic movement; precise verbs tell the model how the body or object behaves.

| Weak verb | Stronger options | What changes |
|---|---|---|
| moves | glides, shuffles, strides, lurches | Speed, weight and attitude |
| walks | strolls, marches, tiptoes, trudges, limps | Mood and rhythm |
| runs | sprints, jogs, bolts, stumbles | Energy and pace |
| flies | soars, hovers, darts, circles | Path of motion |
| falls | tumbles, drifts down, plummets, flutters | Speed and style |
| shakes | trembles, vibrates, rattles, sways | Intensity |
| looks | glances, stares, scans, peers | Eye and head movement |
| talks | speaks calmly, whispers, argues, laughs while speaking | Facial animation |

**Demonstration: same character, four verbs**

> A woman strolls down a quiet street, relaxed, swinging a tote bag.

> A woman marches down a quiet street, jaw set, fast determined steps.

> A woman tiptoes down a quiet street, glancing over her shoulder.

> A woman stumbles down a quiet street, one shoe in her hand, laughing.

The scene is identical, but each verb gives a different story.

### Speed and Timing

**Named speeds**

- **Real-time:** default; say nothing or 'natural speed'.
- **Slow motion:** emphasises detail, drama and elegance; works very well for water, hair, fabric, impacts and dust. Say 'slow motion' or 'high-speed footage'.
- **Time-lapse:** compresses hours into seconds; ideal for clouds, flowers, city traffic, stars and construction. Say 'time-lapse' and name the starting and ending state.
- **Speed ramp:** speeds up then slows down within one shot. Harder; model support varies.

**The action arc inside a short clip.** A 5 to 8 second clip works best when it has a tiny beginning, middle and end:

- Start: the situation ('a cup stands on a table').
- Middle: the action ('a hand reaches in and lifts it').
- End: the result ('steam curls up as it is raised').

Words such as 'then', 'as', 'finally' and 'while' help the model order events. Keep to a maximum of two or three beats.

> A paper boat floats on a puddle. A raindrop lands beside it and ripples push it gently across the water. It drifts under a small leaf and slowly rotates.

### Weight, Force and Physics Cues

If you do not tell the model how heavy or fast something is, it often moves things as if they were weightless. Add physical clues:

- **Weight:** 'a heavy iron door swings slowly with a deep creak' versus 'a light curtain flutters'.
- **Inertia:** 'the car skids and slides sideways before stopping', 'the bike wobbles then steadies'.
- **Contact:** 'boots sink into deep snow', 'fingers press into soft dough'.
- **Liquids:** 'thick honey stretches slowly from the spoon', 'water splashes in a wide arc'.
- **Cloth and hair:** 'silk scarf ripples in the wind', 'long hair lifts and falls as she turns'.
- **Fire, smoke and fog:** 'thin smoke curls upward', 'fire flickers and sparks rise'. These are usually well handled because they are common in training data.

**Demonstration: weightless versus weighty**

> A big man steps out of a truck and walks to a door.

> A heavy-set man climbs down from a pickup truck, the suspension rocks as he steps off, gravel crunches under his boots, he walks slowly and steadily toward a wooden door.

### Secondary Motion and Loops

**Secondary motion** is the small movement that follows the main action: hair after a head turn, leaves after wind, ripples after a stone lands, a coat swinging after a stop. It is what makes video feel natural. Add one or two per prompt.

**Loops.** For web banners, social backgrounds and ambient scenes, choose motion that naturally repeats: waves, flickering candles, rain, neon signs, drifting clouds. Say 'seamless loop' or 'continuous gentle motion' if the tool supports it, and avoid actions with a clear beginning and end.

### What Models Handle Well and Badly

| Usually handled well | Often goes wrong |
|---|---|
| Slow, smooth camera moves | Fast whip pans and rapid cuts inside one clip |
| One person walking, turning or speaking | Several people interacting closely (hugging, fighting, dancing together) |
| Water, fire, smoke, clouds, fog | Fine hand actions (typing, playing piano, tying laces) |
| Wind in hair, cloth and trees | Counting objects, repeated precise actions |
| Animals walking or running at a distance | Sports with rules and equipment (tennis rallies, football passes) |
| Landscapes and time-lapse | Mechanisms with many linked parts (gears, engines) |
| Vehicles on roads in wide shots | Reflections and mirrors that must match the scene |
| Dreamlike or stylised movement | Exact physical cause and effect (a ball bouncing consistently) |

This list changes as models improve, so test rather than assume.

### Strategies for Tricky Motion

1. **Simplify.** Replace 'a drummer playing a fast solo' with 'a drummer sits at a drum kit, sticks moving, camera close on his focused face'.
2. **Increase distance.** A wide shot hides fine hand and face errors.
3. **Cut away.** Show the result or a reaction instead of the difficult action, for example the cup of coffee instead of the pouring hands.
4. **Split into shots.** One shot of the wind-up, a second shot of the result, joined in editing.
5. **Use image-to-video with a start and end frame.** Give the model the first and last picture and let it fill the movement between them.
6. **Slow it down.** Slow motion reduces the number of changes per second and lowers error.
7. **Use video-to-video.** Film a rough version of the movement yourself and let the model restyle it.
8. **Generate several takes.** Pick the best; even experts discard most outputs.

### Worked Example: Improving a Motion Prompt

**Version 1: vague**

> A boy plays football.

Problem: no ball path, no camera, likely results in strange legs and a floating ball.

**Version 2: add subject, setting, a single action**

> A boy in a red jersey dribbles a football across a grass field at sunset.

Better, but leg movement and the ball may still look wrong.

**Version 3: add speed, environment motion, camera**

> A boy in a red jersey dribbles a football slowly across a grass field at sunset, grass blowing in light wind, dust lifting from his boots. Wide tracking shot from the side, slow motion.

Slow motion and a wide, side-on view reduce the chance of leg errors and make the ball easier to follow.

**Version 4: add an action arc and a safe ending**

> A boy in a red jersey dribbles a football slowly across a grass field at sunset, grass blowing in light wind, dust lifting from his boots. He stops, looks up at the sunset and lets the ball settle at his feet. Wide tracking shot from the side, then slowing to a still frame. Slow motion, golden light.

The ending is calm, so there is nothing complex for the model to get wrong in the last seconds.

### Practice for Module 5

1. Write a three-layer motion prompt (subject, environment, camera) for 'a lighthouse keeper climbing the stairs'.
2. Take 'a man picks up a heavy box' and rewrite it with weight, contact and secondary motion cues.
3. Choose a motion from the 'often goes wrong' column and redesign it using two of the strategies above. Generate both versions and compare.
4. Create a seamless loop prompt for a website banner and explain why you chose that motion.

## Module 6: Image-to-Video Workflow (25 min)

This is the most useful professional workflow because it gives you control over the look before you spend time on motion.

**Steps**

1. Generate or choose a strong still image (image model, photograph or illustration).
2. Check it for flaws: hands, faces, text, odd objects.
3. Upload it as the first frame.
4. Write a motion-only prompt. The image already defines the subject and style, so describe only what moves and how the camera behaves.
5. Generate several variants and pick the best.

### Demonstration

**Step 1: image prompt (for an image model)**

> Product photo of a matte black ceramic coffee mug on a marble counter beside a sunlit window, steam rising, soft morning light, shallow depth of field, 3:2 composition.

**Step 2: motion prompt (for the video model)**

> Steam curls gently upward from the mug. Slow push-in toward the mug. Soft sunlight shifts subtly across the marble. Everything else stays still.

**Teaching point:** a short motion prompt such as 'everything else stays still' prevents unwanted changes to a good image.

## Module 7: Consistency Across Shots (25 min)

A single clip is easy. A story needs the same character, outfit and style repeatedly.

**Techniques**

- **Character sheet:** generate front, side and three-quarter views of the character first.
- **Reuse reference images** if the tool supports reference or subject features.
- **Repeat a fixed description block** word for word in every prompt.
- **Fix the style block:** keep lighting, colour palette and lens language identical.
- **Start shots from images** of the same character rather than from text.
- **Use the last frame of one clip as the first frame of the next** for continuity.

### Demonstration: reusable description blocks

**Character block**

> Mira, a 10-year-old girl with short curly black hair, round yellow glasses, a red knitted sweater and blue overalls.

**Style block**

> Soft watercolour storybook look, warm pastel palette, gentle diffused light, 35mm lens.

**Shot 1**

> [Character block] walks along a village lane lined with sunflowers, carrying a small wooden box. Medium tracking shot. [Style block]

**Shot 2**

> [Character block] kneels and opens the wooden box; a tiny glowing paper bird flies out. Close-up on her amazed face, then tilt up following the bird. [Style block]

Expect small drift anyway. Professionals generate many takes and choose the most consistent.

## Module 8: Common Failures and Fixes (20 min)

| Problem | Likely cause | Fix |
|---|---|---|
| Morphing faces or bodies | Too much motion, too many subjects | Simplify: one subject, slower action |
| Warped hands | Known weakness | Frame hands out of shot or use wide shots |
| Gibberish text on signs | Models render text poorly | Avoid readable text; add titles in editing |
| Flicker in video-to-video | Weak temporal guidance | Lower change strength, shorter clips |
| Ignoring part of the prompt | Prompt too long or contradictory | Trim to priority items; remove conflicts |
| Floaty or unreal movement | Vague motion description | Use specific verbs, add speed and weight cues |
| Camera does not move | Camera instruction too weak or buried | Move it earlier; use clear terms like 'slow dolly-in' |

### Demonstration: debugging a prompt

**Original (fails: people melt together)**

> A huge crowd at a festival dancing and celebrating, fireworks, confetti, people hugging and cheering.

**Debugged**

> Wide aerial shot of a festival crowd at night, lit by colourful fireworks overhead, confetti drifting down. Crowd is small in the frame and moves gently. Slow upward crane motion.

**Lesson:** reduce close interactions between many people; use distance and scale instead.

## Module 9: Planning a Short Film or Ad (25 min)

Treat AI video like filmmaking: plan first, generate second.

**Workflow**

1. Write a one-sentence idea and the goal (inform, sell, entertain).
2. Break it into 4 to 8 shots, each 3 to 8 seconds.
3. Write a shot list with shot size, action, camera and sound.
4. Generate each shot (several takes).
5. Edit in a normal video editor: trim, order, add music, voice-over, captions and titles.
6. Colour-match and, if needed, upscale or interpolate frames.

### Demonstration: a 20-second product ad shot list

Idea: a reusable water bottle for hikers.

| Shot | Prompt |
|---|---|
| 1 (4s) | Close-up of a hiker's boots stepping on a misty mountain trail at dawn, slow tracking shot, cool blue light. |
| 2 (4s) | Medium shot of the hiker unclipping a steel water bottle from a backpack, camera handheld, warm sunrise rim light. |
| 3 (4s) | Extreme close-up of cold water pouring from the bottle into a cup, droplets catching sunlight, slow motion, macro lens. |
| 4 (4s) | Wide drone shot rising above the hiker standing on a ridge, bottle in hand, golden valley below. |
| 5 (4s) | Product shot of the steel bottle on a rock, mountains blurred behind, slow orbit, clean commercial lighting. Leave empty space for text. |

The slogan, logo and voice-over are added in editing, not generated, because text rendering is unreliable.

## Module 10: Applications by Background (25 min)

Each student picks the track closest to their field and tests one prompt.

**Marketing and social media**

> Vertical 9:16 video: a bright café counter, barista pouring oat milk into an iced latte in slow motion, ice cubes clinking, soft natural window light, bright playful colours, shallow depth of field.

**Education and training**

> Clean 3D animation explaining how a plant cell works: camera slowly flies into a translucent green cell, highlighting the nucleus and chloroplasts one at a time. Soft studio lighting, pastel colours, no text.

**Film and storytelling**

> Noir detective in a trench coat stands under a flickering streetlamp in heavy rain, lights a cigarette-free pipe, slow push-in on his face, high-contrast black and white, 35mm film grain.

**Architecture and real estate**

> Smooth camera glide through a modern minimalist living room with floor-to-ceiling windows, afternoon sun casting long shadows on pale wood floors, a sofa and a plant, realistic, wide-angle 24mm lens, steady gimbal movement.

**Game design and concept art**

> Cinematic fly-through of a ruined fantasy city overgrown with glowing blue vines at twilight, floating lanterns drifting, slow forward drone shot, painterly concept-art style.

**Healthcare communication (non-diagnostic explainers)**

> Calm animated explainer: a stylised human silhouette with the heart softly glowing, camera slowly zooming in as a gentle pulse animation shows blood flow. Soft blue and white palette, minimal, no text.

**E-commerce and fashion**

> Model-free product clip: a pair of white sneakers on a pastel pink pedestal, slow 360-degree orbit, soft studio lighting, subtle shadows, glossy reflections on the floor.

**Journalism and data storytelling**

> Aerial time-lapse style shot of a river delta with water slowly rising across the land, muted colours, documentary look. (Label as illustrative, not footage of a real event.)

## Module 11: Intermediate Techniques (25 min)

- **Iteration loop:** generate, review, change one variable, regenerate. Changing five things at once teaches you nothing.
- **Seeds:** lock the seed to compare how prompt changes affect an otherwise similar result.
- **Prompt templates:** keep a personal library of tested camera, lighting and style blocks.
- **Negative or avoid fields:** some tools offer them (for example 'blurry, distorted, extra limbs'); use sparingly.
- **Aspect ratio and duration:** choose 16:9 for YouTube and film, 9:16 for Reels and Shorts, 1:1 for feeds, before generating.
- **Upscaling and frame interpolation:** raise resolution and smooth motion after choosing the best take.
- **Lip-sync and voice:** separate tools can sync speech to a face, or some video models generate audio directly.
- **Open-source route:** students with coding or GPU access can run open-weight video models in node-based tools such as ComfyUI for full control, at the cost of setup effort.
- **Evaluating output:** judge prompt adherence, motion quality, consistency, artefacts and usability for the actual purpose.

### Demonstration: single-variable testing

Base prompt:

> A red vintage car driving along a coastal road at sunset, tracking shot, cinematic.

Variant 1: change only the lens: add 'telephoto compression, background hills appear stacked close'.

Variant 2: change only the weather: replace 'sunset' with 'light fog at dawn'.

Variant 3: change only speed: add 'slow motion'.

Compare the three side by side and note what each change did.

## Module 12: Ethics, Law and Responsible Use (15 min)

- **Consent and likeness:** do not generate realistic videos of real people without their permission. Never create deceptive deepfakes.
- **Misinformation:** realistic fake news footage can harm people. Label synthetic content clearly.
- **Copyright and style:** avoid prompts that copy a named living artist, a studio's characters or branded IP for commercial use. Check each tool's terms and the licence for commercial output.
- **Disclosure:** many platforms and some jurisdictions require disclosing AI-generated media. Provenance tools such as content credentials and watermarks are becoming standard.
- **Bias:** generic prompts can produce stereotyped people and settings. Be specific and review outputs critically.
- **Safety:** do not use these tools for harassment, non-consensual intimate content or content involving minors in harmful ways.
- **Privacy and data:** do not upload confidential or personal footage to a tool without checking its data policy.

**Discussion:** A client asks for a testimonial video featuring a realistic AI person saying they used the product. What are the issues?

## Hands-On Lab (60 min)

**Task:** Produce a 15 to 20 second video (3 to 5 shots) for a real or imagined use case from your own field.

**Deliverables**

1. A one-sentence brief and target audience.
2. A shot list with a full prompt for each shot.
3. At least one shot made with image-to-video.
4. A log of three iterations showing what you changed and why.
5. Final edited video with music or voice-over, plus a disclosure line.

**Grading rubric (100 points)**

| Criterion | Points |
|---|---|
| Prompt structure and clarity | 25 |
| Camera and motion control | 20 |
| Consistency across shots | 15 |
| Iteration log and reflection | 15 |
| Fit to purpose and storytelling | 15 |
| Ethical and legal awareness | 10 |

## Quick Reference Cheat Sheet

**Prompt formula:** Subject + Action + Setting + Camera + Lighting + Style + Mood + Sound

**Camera words:** wide, medium, close-up, macro, pan, tilt, dolly-in, tracking, orbit, drone, handheld, crane

**Light words:** golden hour, soft diffused, hard noon sun, neon, candlelit, backlit, rim light, overcast, studio softbox

**Style words:** cinematic, documentary, 3D animation, watercolour, claymation, anime-inspired, vintage film grain, minimalist commercial

**Motion words:** slow motion, time-lapse, gentle drift, sudden burst, steady glide

**Golden rules**

1. One clip, one action.
2. Plan shots before generating.
3. Start from an image when you need control.
4. Change one variable at a time.
5. Add text, logos and final audio in editing.
6. Be honest about what is AI-generated.

## Further Practice for Students

- Rebuild one real ad or film scene you admire as a shot list and generate it with your own prompts.
- Take one prompt and produce it in three art styles; write down what changed.
- Swap prompts with a classmate and see whose wording gives the more accurate result.
