---
name: paper-framework-figure
description: Generate a CCF-A-grade paper framework figure via nano banana (google/gemini-2.5-flash-image on OpenRouter). The LLM autonomously summarises the project's background / problems / innovations / technical-route from prior stage outputs, fills a strict 4-section prompt template, calls OpenRouter, decodes the base64 image, and saves a PNG referenced from the stage output. Activate when Stage 4 (Methodology Design) or Stage 8 (Paper Generation) needs a publication-quality framework diagram.
allowed-tools: Read, Write, Bash
---

# Paper Framework Figure — Synthesize then Draw via nano banana

You are about to add a **framework figure** to a Stage 4 methodology
document or a Stage 8 paper draft. The figure must look like it came out
of a CCF-A venue (NeurIPS / ICML / CVPR / ACL), not a slide deck.

You do **not** sketch by hand or write SVG. You delegate the actual
drawing to **nano banana** (`google/gemini-2.5-flash-image` on OpenRouter),
which is an image-generation model that takes a long, specific prompt and
returns a PNG.

Your job is the **summarisation + prompt composition** part — the
half of the work that requires understanding what the paper actually
contributes. nano banana cannot read the prior stage markdowns; you must
distill them into the 4-section "work summary" that the prompt template
demands.

<HARD-GATE>
Do NOT call the API until you have:
  1. Read the relevant prior stage outputs (Stage 1 + 2 + 3 + 4 at
     minimum; for Stage 8 also include Stage 5/6/7).
  2. Produced the 4-section summary in EXACTLY the schema below
     (背景 / 问题和难点 / 创新点 / 具体的技术路线). No abbreviation,
     no skipping numbered points, no "...etc."
  3. Written a one-paragraph **figure contract** ABOVE the summary that
     states, in plain prose:
       - **Takeaway**: the single claim a reader should grasp from the
         figure in 5 seconds (one sentence).
       - **Central vs supporting**: which one component/flow is the visual
         centerpiece (the contribution), and which are context/baseline.
       - **Specificity hook**: 1-2 elements that make this figure
         unmistakably about THIS paper (named modules, the actual data
         flow, the novel step) and could not appear in a generic pipeline.
     The figure prompt must then realise this contract — decide the
     argument before the drawing. If the contract is generic, the figure
     will be too (and the critic auto-REJECTs it).
  4. Verified `OPENROUTER_API_KEY` is set (env var). If unset, STOP and
     report missing-credential — do NOT guess a key, do NOT hardcode one.

If the figure is missing or shows a generic flowchart unrelated to the
paper, the Stage 4/8 critic auto-REJECTs.
</HARD-GATE>

---

## When to Use This Skill

**USE for:**
- Stage 4 (Methodology Design) — embed a framework figure in
  `stage4_methodology_designer.md` that visualises the methodology's
  components and data flow.
- Stage 8 (Paper Generation) — render the headline framework figure
  (Figure 1 / Figure 2) for the introduction or methods section.

**SKIP for:**
- Stages 1, 2, 3, 5, 6, 7, 9 — none of these need a framework figure
  written by an image model.
- Retries after critic rejection where the critique was about the
  **text**, not the figure — the existing PNG is fine, fix the prose.
  Only re-call this skill if the critic explicitly flagged the figure.

---

## Phase 1 — Read Prior Context

Before composing the prompt, read **all** prior pipeline outputs that
inform the figure. Do not skip; the figure must reflect the actual
contributions, not a generic template.

For Stage 4:

```python
read("stage1_topic_refiner.md")        # research question
read("stage2_literature_surveyor.md")  # what's known, what's contested
read("stage3_idea_generator.md")       # the specific idea / claim
read("stage4_methodology_v1_draft.md") # this stage's draft (if exists)
```

For Stage 8 (paper writer), also:

```python
read("stage5_experiment_designer.md")  # methodology realised as procedure
read("stage6_experimentalist.md")      # the actual run + metrics
read("stage7_result_analyst.md")       # the confirmatory analysis
```

If a Stage's output file is missing, that information is unavailable —
do not invent it. Note the gap in **Open Questions** of the summary.

---

## Phase 2 — Produce the 4-Section Work Summary

Write the 4 sections **in this exact order with these exact headings**.
This is the structure the prompt template expects; deviation breaks the
figure layout.

```markdown
背景:
<Single paragraph (3-6 sentences). Why this research direction matters
right now. Concrete pain points, not vague intros. Reference the field
(e.g. "large-scale tool-using LLMs", "open-domain QA").>

问题和难点:
(1) <Numbered, concrete, ≤2 sentences each. Each one is a specific
    failure mode of existing approaches that your work addresses.>
(2) <...>
(3) <...>
(4) <...>

创新点:
(1) <Numbered. Each is a specific contribution with a named mechanism
    or formal property (e.g. "Stackelberg game between orchestrator
    and workers", not "novel framework"). Include the technical noun
    that names the contribution.>
(2) <...>
(3) <...>

具体的技术路线:
Stage 1: <Component name> — (对应组件: <list of named subcomponents>)
  Step 1: <action + input + output>
  Step 2: <...>
  Step 3: <...>
  Step 4: <...>

Stage 2: <Component name> — (对应组件: <list of named subcomponents>)
  Step 1: <...>
  Step 2: <...>
  Step 3: <...>
  Step 4: <...>

<Add more stages if the paper has them. Cap at 4 stages — diagrams
with >4 stages don't fit one figure.>
```

### Rules (hard, the critic checks these)

- **背景**: never just say "AI is important". State the specific
  capability gap and why the field cares.
- **问题和难点**: exactly 3-5 items, all numbered `(1)`, `(2)`, … and
  each ≤2 sentences. Each item must be a real research problem, not a
  generic complaint (no "scalability" without saying *what* scales and
  *why*).
- **创新点**: exactly 3-4 items. Each item must name the technical
  mechanism, not just label it "novel". If you cannot name the
  mechanism in ≤8 words, you have not understood it well enough to
  draw it.
- **具体的技术路线**: 2-4 stages, each with named components in
  parentheses and 3-5 numbered steps. The components are what get
  drawn as boxes; the steps are what get drawn as arrows. If you
  cannot list them, the figure cannot be drawn.
- **Language**: write the summary in the same language as the paper
  (default Chinese for the Memento-Team pipeline; switch to English
  if the upstream stages are English). nano banana renders whatever
  text you give it.

---

## Phase 3 — Compose the Final Prompt

Insert your 4-section summary into the template below **verbatim**.
Do not alter the constraint preamble or the icon URL / font line —
those are part of what makes the output look CCF-A.

```text
请你为我的论文绘制一幅符合CCF-A类会议的要求和taste的整体框架图。你绘制的整体框架图需要具有学术风格并且能表现出我论文的技术路线。我现在只画了一个简单的设计草图（这只是一个风格的参考，不一定需要完全在我的草图上进行进一步设计，但是最好是保留我的设计，如果你使用icon的话一定是这个网站上面的：https://www.flaticon.com/authors/special/lineal-color?author_id=1&type=standard，使用的字体必须是：Comic Sans MS）。

下面是论文的一些背景信息和详细的技术路线：

背景信息：
<<<INSERT 背景 paragraph from Phase 2>>>

问题和难点：
<<<INSERT 问题和难点 numbered list from Phase 2>>>

创新点：
<<<INSERT 创新点 numbered list from Phase 2>>>

具体的技术路线：
<<<INSERT Stage-by-Stage technical route from Phase 2>>>
```

Save the composed prompt to `paper_figure_prompt.md` in the project
workspace so the critic and future iterations can audit what was sent
to nano banana.

```python
write("paper_figure_prompt.md", composed_prompt)
```

---

## Phase 4 — Call nano banana via OpenRouter

Use the OpenRouter API (model `google/gemini-2.5-flash-image`).
**Authenticate with the env var `OPENROUTER_API_KEY`. Do NOT echo the
key into the chat or write it into any file.**

### Critical: prepend an image-only directive (or you get text back)

If you send the bare prompt template, the model often replies in
**chat mode** with text like "好的,我来帮你绘制..." and 0 image
tokens — wasted call. You MUST wrap the prompt with an explicit
"output one image, no text" directive. The Chinese paper-figure spec
goes inside the wrapper, not at the top:

```text
Generate ONE image (PNG, landscape ~1600×1000). No text explanation,
no markdown, no chat — just output the image. The image should be a
CCF-A-style academic paper framework figure following the spec below.

=== FIGURE SPEC (Chinese) ===
<<<full composed prompt from Phase 3 here>>>
```

If you see `image_tokens: 0` in the response usage, your wrapper was
missing — retry once with the wrapper. Don't retry > 2× on the same
prompt; if it still won't draw, the issue is the prompt content
(usually too vague), not the wrapper.

### Curl + Python decode

```bash
# Phase 3 wrote paper_figure_prompt.md — wrap it now
python3 - paper_figure_prompt.md > /tmp/nano_request.json << 'PYEOF'
import sys, json
spec = open(sys.argv[1], encoding="utf-8").read()
wrapped = (
    "Generate ONE image (PNG, landscape ~1600×1000). No text explanation, "
    "no markdown, no chat — just output the image. The image should be a "
    "CCF-A-style academic paper framework figure following the spec below.\n\n"
    "=== FIGURE SPEC (Chinese) ===\n" + spec
)
body = {
    "model": "google/gemini-2.5-flash-image",
    "messages": [{"role": "user", "content": wrapped}],
    "modalities": ["image", "text"],
}
print(json.dumps(body, ensure_ascii=False))
PYEOF

# POST
curl -s -X POST 'https://openrouter.ai/api/v1/chat/completions' \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/nano_request.json \
  -o /tmp/nano_response.json

# Decode the base64-encoded PNG and save under the project workspace
python3 - << 'PYEOF'
import json, base64, os, sys
with open("/tmp/nano_response.json", encoding="utf-8") as f:
    d = json.load(f)
if "error" in d:
    print(f"ERROR: {d['error']}", file=sys.stderr)
    sys.exit(1)
images = d.get("choices", [{}])[0].get("message", {}).get("images", [])
if not images:
    text = (d.get("choices",[{}])[0].get("message",{}).get("content") or "")[:200]
    print(f"ERROR: no image returned. Model replied in chat mode: {text!r}", file=sys.stderr)
    print("HINT: Ensure the prompt is wrapped with the 'Generate ONE image' directive", file=sys.stderr)
    sys.exit(1)
url = images[0].get("image_url", {}).get("url", "")
if not url.startswith("data:image/"):
    print(f"ERROR: unexpected image url prefix: {url[:60]}", file=sys.stderr)
    sys.exit(1)
mime, b64 = url.split(",", 1)
# Pick filename by stage — see Phase 5
out = "paper_framework_figure.png"   # overwrite per stage if needed
with open(out, "wb") as f:
    f.write(base64.b64decode(b64))
usage = d.get("usage", {})
img_tokens = usage.get("completion_tokens_details", {}).get("image_tokens", 0)
print(f"  ✓ wrote {out} ({os.path.getsize(out)} bytes, {img_tokens} image tokens)")
print(f"  cost: ${usage.get('cost', '?')}")
PYEOF
```

### Cost note

One call typically costs **~$0.04 USD** (~1290 image tokens at the
current Gemini 2.5 Flash Image rate). Budget accordingly — do not
re-run the skill multiple times "just to see variations" without an
explicit critic flag. If the figure is rejected, fix the prompt
(usually the technical-route section is wrong), not the model.

### Failure modes & retries

- **HTTP 401 / "User not found"** — `OPENROUTER_API_KEY` is unset or
  invalid. Stop. Do not retry. Surface to the user.
- **Empty `images` array** — the model returned only text. The prompt
  was probably too short or too vague. Add more technical detail to
  the 4-section summary (especially `具体的技术路线`) and retry once.
- **Image renders but is generic / unrelated** — the prompt was too
  abstract. Each numbered item in `问题和难点` and `创新点` must name
  concrete mechanisms; the LLM-image model cannot infer them.
- **Network timeout** — retry the curl up to 2 times with 5s sleep
  in between. If still failing, surface as infra issue.

---

## Phase 5 — Embed in the Stage Output

Rename the file to a stage-specific name and reference it from the
stage's markdown.

```bash
# Stage 4
mv paper_framework_figure.png stage4_framework_figure.png
```

Then add a Figure block to `stage4_methodology_designer.md` (or
`stage8_paper_writer.md`):

```markdown
![Figure 1. Memento-Team framework: a self-evolving bilevel game-theoretic
multi-agent system. The Orchestrator (leader) decomposes long-tail queries
into worker subtasks via a learned skill bank; Workers (followers) execute
in parallel against a shared Markdown workboard with file-lock arbitration.
The two layers co-evolve via a run-verify-reflect loop that promotes
recurring decomposition patterns into reusable skills.](stage4_framework_figure.png)
```

The figure caption is part of the deliverable. It must:

- Number the figure (`Figure 1.`, `Figure 2.`, …).
- Name every box and arrow shown in the rendered image, in one
  paragraph (CCF-A house style).
- Avoid "see above" / "the framework" / vague pronouns — name the
  components.

If the caption can't describe the figure because the figure is
unclear, the figure is wrong; rerun Phase 3/4 with a tighter
technical-route section.

---

## Submitting

After Phase 5 the stage owner's normal `submit_result` includes:

- The PNG file under the project workspace
- A captioned Figure block in the stage markdown
- `paper_figure_prompt.md` (the audit trail of what was sent to nano banana)

These three artifacts together make the figure reproducible — anyone
can edit `paper_figure_prompt.md` and rerun Phase 4 to iterate on the
figure without re-doing Phases 1-3.
