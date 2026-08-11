# Why Smaller Local AI Agents Fail in Strange Ways

**A beginner’s guide to quantization, tool-use failures, and a research plan for preserving the behavior that matters**  
**As of:** 2026-08-08  
**Origin:** [sparse-inference failure-mining thread](https://x.com/tokenbender/status/2085977125845352679)  
**Evidence base:** [`complaint_census.json`](complaint_census.json), 1,640 normalized reports  
**Focused set:** 101 reports involving tool calling  
**Research trail:** [`conversation_research_question_ledger.json`](conversation_research_question_ledger.json)

## Start with the problem a user can see

Imagine that you run an AI assistant on your own computer. It can search your files, look up weather, update a calendar, or work through a coding task. The original model is too large for your hardware, so you download a smaller “4-bit” version.

Ordinary chat still looks fine. The assistant answers questions and sounds like the same model. But longer jobs feel less dependable:

- a tool call contains the right idea in the wrong format;
- an account or file identifier changes by one character after several turns;
- the assistant repeats an action that already failed;
- a task finishes, but only after many more retries;
- a cache or serving option makes correct output turn into garbage;
- the supposedly efficient model is not faster on the hardware you own.

A benchmark may still mark the job as a success. As a user, you can see that something got worse.

This document explains why that can happen, what public failure reports actually tell us, and how those failures could be used to build a better compressed model. It assumes experience **using** local models, not training them.

> **The central question:** Can real failures from tool-using local models become repeatable tests that tell us which parts of a compressed system need more numerical precision?

The short answer is **possibly, and the first clean experiment is now clear**. The evidence does not justify saying that the problem is solved.

### What the evidence changes

The most important finding is that broken output is far less diagnostic than it appears. When a quantized model produces gibberish, loops, or malformed tool calls, the natural assumption is that compression damaged its reasoning. Yet in our mechanism-confirmed sample, 34 of 37 such cases were associated with the surrounding execution path: kernels, caches, conversion, templates, loading, or runtime interactions. This is not an estimate of how common those faults are in the wild. It establishes a more practical point: the symptom alone cannot tell us what failed. Before increasing precision or retraining the model, we should first isolate whether the problem belongs to the model, its working state, or the software serving it.

The more surprising implication is that failures experienced as random may be repeatable consequences of hidden state. In one report, changing the prefix with a semantically irrelevant nonce rescued 18 of 18 repeated corruptions. In another incident, a cache-related change caused request-isolation violations in 1,151 of 1,152 checks; reverting four lines reduced the count to zero. These cases suggest that output can depend on cache history, chunking, concurrency, and request ownership, not only on the model and prompt. That distinction matters because an apparently invented identifier or answer may, in rare cases, reflect stale or foreign state. What initially looks like a quality problem can therefore become a reliability, privacy, or integrity problem.

This leads to a different research objective. Rather than asking which layers are universally important, we should ask which components require additional precision for a particular failure, inference phase, and deployment environment. Evaluation should follow the complete task: where the first error appears, whether the agent recovers, how many retries and tokens it consumes, and whether it completes correctly at the intended speed and power budget. The strongest compressed agent is not necessarily the one that most closely reproduces the original model’s internal numbers. It is the smallest deployable system that preserves the behaviors required to complete the job reliably.

### If you remember only five things

1. **A tool-using agent is more than a model.** It also includes a prompt template, parser, tools, working memory, runtime, and serving settings.
2. **Compression can damage the journey without changing the final score.** More malformed calls, retries, loops, or state errors can hide inside an eventual success.
3. **A complaint is a clue, not a diagnosis.** The same visible failure can come from the model, parser, cache, kernel, or their interaction.
4. **The useful artifact is a repeatable test.** Generate controlled versions of a failure and check each step with code.
5. **Use those tests to spend precision where behavior needs it.** Keep the same memory and speed budget, but protect the components that prevent important failures.

### The idea in one picture

```text
run the original model
        ↓
make a smaller, lower-precision version
        ↓
replay complete tool-using journeys, not just one prompt
        ↓
record the first bad step and identify the responsible layer
        ↓
temporarily remove or restore precision in candidate components
        ↓
keep more precision only where it prevents verified failures
        ↓
build the real deployable artifact and test the whole system again
```

The proposed method is called **Failure-Conditioned Mixed-Precision Quantization**, shortened to **FC-MPQ**. The name matters less than the rule: **optimize for reliable behavior across the whole tool-use journey, not only for similarity to the original model’s numbers.**

## 1. What is actually running when you use a local agent?

A language model predicts the next piece of text. A tool-using agent surrounds that model with software that turns some text into actions and feeds the results back into the next turn.

A simplified agent contains these parts:

| Part | What it does | A failure a user might see |
|---|---|---|
| Model weights | Store the learned numerical patterns that produce the next token | The model chooses the wrong tool or forgets an instruction |
| Chat template | Arranges system rules, user text, tool descriptions, and prior turns | Tool syntax changes or the model answers in the wrong role |
| Parser | Converts generated text into a tool name and arguments | A sensible-looking call is rejected as invalid |
| Tool and environment | Executes the requested action | A valid call returns an error or unexpected result |
| Working state | Carries prior tokens and intermediate information forward | An exact identifier drifts after a long conversation |
| Runtime and kernels | Perform the numerical computation on CPU, GPU, or accelerator | Output becomes corrupted, the server crashes, or speed collapses |
| Serving policy | Controls batching, caching, concurrency, and speculative decoding | A model works alone but fails under production settings |

This distinction is practical. Suppose the model emits valid JSON, but the parser expects a different delimiter. Retraining the model would treat a software contract bug as a learning problem. Suppose identical model files work in one runtime and emit garbage in another. Changing the model weights would hide a kernel defect rather than fix it.

Throughout this report, **the deployed agent** means the complete path from model weights to executed action and updated state.

## 2. What does compression change?

### 2.1 Weights are stored numbers

A modern model contains billions of learned numbers called **weights**. Think of them as an enormous spreadsheet of settings. During generation, the runtime repeatedly combines those numbers with the current input.

A computer must choose how many binary digits, or **bits**, to use for each number. More bits can represent finer distinctions. Fewer bits use less storage and move less data through memory, but they round many nearby values to the same representation.

A simple analogy is recording prices:

- `$17.43` preserves cents;
- `$17` uses less detail;
- `$20` is coarser still.

Rounding one price may not matter. Rounding millions of interacting values can matter in specific, hard-to-predict places. The model may remain fluent while a thin decision margin between two tool names disappears.

### 2.2 Quantization means using fewer bits

**Quantization** stores or computes model values with lower precision. A model commonly served with 16-bit values may be converted so that many weights use 8 or 4 bits. At the level of raw weight payload, 4 bits is one quarter of 16 bits. Real artifacts also contain scales, metadata, uncompressed tensors, and runtime buffers, so the complete memory saving is not exactly fourfold.

People quantize models because it can:

- make a model fit in available RAM or VRAM;
- reduce memory bandwidth;
- sometimes improve speed;
- lower serving cost or power use.

It can also reduce quality, introduce hardware-specific problems, or make an existing rare failure happen more often.

### 2.3 Uniform and mixed precision

A **uniform** policy gives most eligible components the same precision, such as 4 bits.

A **mixed-precision** policy spends the byte budget unevenly. Most components may stay at 4 bits while a smaller set receives 6, 8, or 16 bits. This is like packing for a strict weight limit: you do not make the bag heavier; you decide which items cannot safely be replaced by lighter versions.

The hard part is choosing what to protect. Existing methods often look for components whose rounded numbers differ most from the original, whose activations are unusual, or whose removal changes a task score. The proposal here adds a different signal: **which components prevent a known family of tool-use failures?**

### 2.4 Weight precision is not the only compression choice

| Choice | Plain-English meaning | Examples you may encounter |
|---|---|---|
| Numerical format | How a value is represented | FP8, NVFP4, MXFP4, INT4, ternary |
| Quantization method | How lower-precision values or rounding choices are selected | GPTQ, AWQ, AutoRound |
| Packaging or deployment recipe | How tensors and metadata are stored for a runtime | GGUF K-quants, IQ-quants, imatrix-guided recipes |
| State compression | How the agent’s working memory is stored | KV-cache quantization, KIVI, TriAxialKV |
| Pruning | Removing selected weights or structures rather than rounding them | Unstructured or structured sparsity |
| Quantization-aware training | Training while simulating or applying low precision | QAT, low-bit distillation |

These labels answer different questions. “GGUF,” “Q4,” “AWQ,” and “KV cache” are not four competing versions of the same thing. One describes packaging, one a precision regime, one a method, and one working state.

### 2.5 Physical efficiency must be measured

A file labeled “4-bit” is not automatically small or fast in use. High-precision exceptions need metadata. Irregular layouts may force dequantization. A hardware kernel may not accelerate the chosen pattern. The relevant measurements are:

- bytes loaded into memory, including scales and metadata;
- peak memory during a real request;
- prompt-processing speed;
- token-generation speed;
- correctly completed tool journeys per second.

The last measure joins speed and reliability. A fast model that wastes time on retries may do less useful work.

## 3. Why normal benchmarks can miss the damage

A single chat answer is one output. An agent task is a **trajectory**: a sequence of model decisions, tool calls, tool results, state updates, recoveries, and a final stop.

```text
user request
  → choose a tool
  → format its arguments
  → parse the call
  → run the tool
  → read the result
  → update working state
  → choose the next action
  → recover if something failed
  → stop at the right time
```

Every arrow creates another place where a small error can change what happens next.

### 3.1 Final success and process quality are different

Consider a benchmark that allows an agent up to ten mistakes or retries before marking a task as failed:

| Run | Internal errors | Final outcome |
|---|---:|---|
| Higher-precision model | 2 | Success |
| 4-bit model | 6 | Success |

The benchmark reports a tie. The second agent was still less stable, slower in useful work, and closer to the failure boundary.

This is not only a hypothetical concern. The July 2026 study [Flat Score, Amplified Failures: How the Error Budget Masks Damage in Quantized LLM Agents](https://arxiv.org/abs/2607.27275) compared higher-precision and quantized tool agents in a controlled setting. In the main experiments, activation and working-state precision stayed at 16 bits, helping isolate the effect of weight precision.

The study found:

- no final-score difference that survived its multiple-comparison correction;
- as much as a **2.5× increase** in an existing failure channel;
- almost no new kinds of failures, with only **0.18%** novel events;
- very similar rankings of failure types across precision, with correlation at least **0.94**;
- a **17-point score gap** when the allowed-error budget was tightened to two errors.

The interpretation is important: quantization often did not invent a new behavior. It made a weakness already present in the original model fire more often. A forgiving benchmark absorbed the extra damage.

The paper does **not** show which individual weights caused that change, how to allocate mixed precision, or how every local runtime behaves. Those are open steps.

### 3.2 What to measure instead of final success alone

A serious agent evaluation should keep final task success, then add:

- the first turn where a required condition fails;
- malformed or rejected calls;
- wrong tools and wrong arguments;
- exact identifier retention;
- repeated failed actions;
- recovery after a tool error;
- state differences between fresh and reused execution;
- completion before a step or token limit;
- retries and extra generated tokens;
- useful completed journeys per second.

These are **process measures**. They show how the result was reached.

## 4. What public failure reports tell us

The repository’s [`complaint_census.json`](complaint_census.json) contains **1,640** normalized public reports and research incidents about sparse or low-precision inference. The collection was built to find failures, so it is not a random survey of all users or all quantizers.

A focused subset contains **101 reports involving tool calling**. Of those:

- 55 have demonstrated evidence or a verified witness;
- only 14 have a confirmed mechanism;
- 43 are unresolved and 12 have unknown resolution;
- 91 come from only one source lineage;
- 51 do not identify an exact model file.

This is enough to find recurring test ideas. It is not enough to rank quantization methods by complaint count.

> **Do not read “62 GGUF-related reports” as a 62% GGUF failure rate.** GGUF appears often because local users deploy and discuss it often. The corpus has no denominator for total successful uses.

### 4.1 Where the visible problem appeared

Each report was assigned one primary operational layer. That layer is the most visible location, not necessarily the proven root cause.

| Primary layer | Rows | Share | Strong evidence | Confirmed mechanism |
|---|---:|---:|---:|---:|
| Protocol, runtime, composition, or loading | 43 | 42.6% | 24 | 9 |
| Model/task-selective behavior | 26 | 25.7% | 10 | 1 |
| Representation, state, conversion, or numerical path | 22 | 21.8% | 14 | 3 |
| Economics, capacity, or serving overhead | 10 | 9.9% | 7 | 1 |

The first row is the largest. Nearly half of the focused reports visibly involve protocol, runtime, composition, or loading. That is why it would be wrong to pour every complaint directly into model training.

### 4.2 Four recurring failure shapes

#### A. The model has the right intention but the call is unusable

The assistant may choose the right tool yet omit a delimiter, use the wrong schema, or produce text the parser does not recognize. To a user, this looks like “tool calling broke.” The fix may belong in precision allocation, the prompt template, or the parser. Raw generated tokens must be saved before deciding.

One report about [Qwen3.5 in MLX](https://github.com/ml-explore/mlx-lm/issues/1011) described plain-text tool calls appearing around round 5 for a 4-bit model and around round 13 for an 8-bit model, while other variants reportedly completed a much longer run. That is a strong reason to sweep conversation length while holding the intended task fixed. It is not yet proof of one causal layer.

#### B. Working memory changes an exact value

A tool agent often has to preserve identifiers exactly. One changed character can target the wrong file, customer, or database row.

In a [llama.cpp cache-rewind report](https://github.com/ggml-org/llama.cpp/issues/21681), identifiers changed after history mutation and prefix rewind. The right test is to replay the same final token sequence with fresh state and with reused or rewound state. If only the reused path drifts, the cache implementation is the primary suspect.

#### C. The agent cannot recover and starts looping

A weak recovery step can become a long sequence of repeated bad actions. A report about [DeepSeek-V4 with compressed state](https://github.com/sgl-project/sglang/issues/31482) retained a strong question-answer score, yet many software tasks reportedly exhausted a 300-step limit amid malformed or repetitive output. The useful measurement is not only whether the task eventually passed. It is the first bad action, repetition rate, recovery behavior, and completion before the cap.

#### D. A runtime option corrupts otherwise valid computation

Some failures are ordinary software defects. A [MiniMax NVFP4 backend report](https://github.com/sgl-project/sglang/issues/26324) described deterministic replacement-character-heavy output on one path and an assertion on another. A [speculative-decoding report](https://github.com/sgl-project/sglang/issues/33800) found rare corruption at one draft depth but not nearby tested depths. These belong first in kernel and runtime qualification, not reinforcement learning.

### 4.3 Gibberish, random tokens, and loops are terminal symptoms

A broad symptom query over the complete corpus matched reports whose title, observation, or complaint contained `gibberish`, `nonsense`, `garbage`, `repetitive`, `repetition`, `looping`, `loops`, long runs of exclamation marks, or the phrase `exclamation marks`. It found **220 rows across 212 source lineages and 15 different primary operational labels**.

The mechanism-confirmed slice is more revealing. Of 37 confirmed lineages, 15 were labeled kernel correctness; four each were composition incompatibility, stateful precision failure, conversion-pipeline damage, and template/tokenizer mismatch; two were support fragmentation; and one was model loading. Those 34 runtime, pipeline, or configuration lineages all reached a similar visible endpoint. The other three were two memory-representation cases and one capability-specific case.

This is not evidence that 34/37 real-world gibberish failures come from software. The corpus was deliberately mined for failures, and its labels are annotations rather than random-sample outcomes. It is evidence for a narrower and operationally important statement:

> **Visible nonsense does not identify the damaged layer.**

The same artifact can emit clean text in one backend and garbage in another. The [bitsandbytes/vLLM report](https://github.com/vllm-project/vllm/issues/5569) reproduced unusable output in vLLM while the same model and nominal 4-bit method worked through Transformers. A separate [2:4 pruning-plus-quantization report](https://github.com/vllm-project/vllm/issues/10819) eventually localized abnormal output to the Marlin 2:4 kernel after the sparse model itself produced normal output through another path.

The first response to gibberish should therefore be a controlled backend, build, template, cache, and concurrency swap with identical model bytes and input tokens. Increasing bitwidth or training the model comes later.

### 4.4 The limits of the corpus

The corpus supports four practical conclusions:

1. agent-relevant failures recur around format, state, looping, runtime, and performance;
2. many symptoms can be checked automatically;
3. exact artifacts, controls, and mechanisms are often missing;
4. failures need routing to different fixes.

It does not establish comparative failure rates, universal mechanisms, open-world reliability, or a strong pruning claim. Only three focused rows directly combine pruning and quantization.

The complete count tables and all thirteen incident-to-test cards are preserved in Appendices A and B.

## 5. How a complaint becomes a useful test

A forum post says, “the model loses tool calling after several turns.” That is a report, not yet an experiment.

A reusable test needs two parts:

- a **generator** that creates controlled versions of the risky situation;
- a **verifier** that decides, with code, whether each required condition held.

### 5.1 Worked example of exact identifier retention

Suppose an assistant must carry the identifier `N7B4Q19X` through a twenty-turn support workflow.

A generator can vary:

- the identifier while keeping its length and character classes;
- tool names and schemas;
- the number of turns;
- the length of tool results;
- fresh, reused, and rewound working state;
- the quantized format and runtime.

The verifier checks every round:

1. Was a tool requested when one was required?
2. Was the declared tool name used?
3. Did the output satisfy the schema?
4. Did the parser accept it?
5. Was `N7B4Q19X` copied exactly?
6. Did the tool run successfully?
7. Did the assistant recover correctly from a controlled error?
8. Did it stop after the task was complete?

Now “identifier drift” is a measurable failure family. The test can locate the first divergence instead of judging only the final prose.

### 5.2 A verifier is more than a model judge

Some checks are exact: JSON parses, a tool exists, an argument has the declared type, an identifier matches, or execution terminates before a cap. These should be ordinary program logic where possible.

A language-model judge may still help with open-ended semantics, but it should not replace exact checks. Otherwise the evaluator can become as fuzzy as the system under test.

### 5.3 Test the failure family

A model can memorize one schema or one identifier. A useful family holds out meaningful variations:

- unseen tool inventories;
- different but structurally related schemas;
- new identifiers;
- longer trajectories;
- new history edits;
- other languages or domains;
- different error and recovery paths.

Renaming one tool in an otherwise identical prompt is not strong evidence of transfer. The held-out set should differ in the mechanism that triggers the failure, not only in wording.

## 6. Locate the faulty layer before changing the model

A visible failure travels through a chain:

```text
model’s intended action
        ↓
raw generated tokens
        ↓
chat template and serialization
        ↓
parser acceptance
        ↓
tool execution
        ↓
working-state or cache update
        ↓
next model action
        ↓
final task contract
```

Save a receipt at each boundary. Then compare controlled lanes:

| Lane | What changes from the previous lane | What it can reveal |
|---|---|---|
| A | Original higher-precision model, fresh state, canonical parser | Whether the parent already has the weakness |
| B | Replace only the model representation with the quantized one | Weight or activation precision damage |
| C | Move the same quantized artifact to the target runtime and parser | Conversion, kernel, template, or parser damage |
| D | Reuse or rewind state instead of starting fresh | Cache or recurrent-state damage |
| E | Enable target concurrency or speculative decoding | Production-only interaction damage |

The decision rules are deliberately boring:

- Correct raw tokens plus parser rejection means fix the contract or parser.
- One backend works and another corrupts the same artifact means fix or reject the runtime path.
- Fresh state works and reused state drifts means fix state handling.
- The quantized model diverges across runtimes while every surrounding condition stays fixed means weight or activation precision becomes a plausible cause.
- The original model already fails in the same way means measure amplification; do not call the failure newly created by quantization.

This attribution step prevents an expensive mistake: training the model to compensate for broken software.

## 7. Let verified failures choose precision

Once a failure family is reproducible and attributable to model representation, it can guide mixed precision.

Start with two reference points:

- the original higher-precision model;
- a conventional deployable 4-bit model.

Then test candidate groups that the runtime can actually store and execute, such as one attention matrix, one feed-forward block, a router group, or a structured set of channels.

### 7.1 Require both damage and rescue

For each candidate group:

1. **Downgrade it** inside a stronger model. Does the verified failure get worse?
2. **Upgrade it** inside a lower-precision model. Does the failure improve?
3. Repeat under several surrounding precision layouts. Does the effect survive interaction with other groups?
4. Build the real artifact. Do memory and speed still meet the budget?

A component is a credible protection target when removing precision causes damage and restoring precision rescues behavior. High activation, gradient, or reconstruction error alone is only a correlation signal.

### 7.2 Protect the weakest failure family

If one policy preserves tool selection but destroys identifier retention, its average may still look good. The allocation should therefore avoid averaging all failure families into one forgiving score.

The plain-English objective is:

> Under a fixed memory and speed budget, choose the precision layout whose worst verified failure family degrades the least relative to the original model.

This is the heart of **Failure-Conditioned Mixed-Precision Quantization (FC-MPQ)**.

### 7.3 The verifier has three jobs

1. **Choose calibration cases.** Include examples that activate known weak boundaries.
2. **Choose precision.** Measure which deployable groups damage or rescue each family.
3. **Qualify the release.** Re-run the same contracts on the final file, runtime, parser, cache, and serving settings.

The text examples alone are not the main asset. The executable checks are.

## 8. What earlier research already solved

The proposal combines several existing research lines. It must not be sold as the first “agent-aware” or “task-aware” quantizer.

### 8.1 Agent compression has already been evaluated

[Can Compressed LLMs Truly Act? / ACBench](https://proceedings.mlr.press/v267/dong25k.html) evaluates several compression methods across agent-related tasks and shows that damage depends on capability and workload. [Flat Score, Amplified Failures](https://arxiv.org/abs/2607.27275) goes further inside the loop and shows that final reward can hide extra process failures.

These works establish the evaluation problem. They do not use failure families to choose weight or state precision.

### 8.2 Task-aware precision allocation already exists

Three especially relevant examples are:

- [TASA: Beyond Activation Alignment](https://arxiv.org/abs/2607.00908), which jointly studies calibration mixture and mixed-precision allocation for target tasks;
- [You Have One Job: Per-Task Quantization Using LLMs’ Hidden Representations](https://arxiv.org/abs/2511.06516), whose TAQ/TAQO methods score transformer layers for a particular task and assign precision per layer;
- [Task-Circuit Quantization](https://arxiv.org/abs/2504.07389), which preserves task-associated weight circuits at higher precision.

**Why “You Have One Job” matters here:** it is a clean scaffold for the first experiment. It already asks which layers matter for a task. The proposed extension replaces one-shot task sensitivity with damage to complete, verifier-checked failure trajectories. It may also need finer groups than whole layers.

### 8.3 Behavior-guided and state-aware precision also exist

[ActQuant](https://arxiv.org/abs/2605.24011) allocates precision using action sensitivity for embodied vision-language-action models. It prevents a claim to the first behavior-guided mixed-precision method.

[KIVI](https://arxiv.org/abs/2402.02750), [IntactKV](https://arxiv.org/abs/2403.01241), and [TriAxialKV](https://arxiv.org/abs/2605.17170) show that working-state precision can depend on channel, token, recency, modality, and semantic role. [Q-Mamba](https://aclanthology.org/2025.findings-acl.551/) shows structured precision choices for recurrent state.

### 8.4 Low-bit training and tool rewards exist

[Reasoning-QAT](https://arxiv.org/abs/2601.14888) studies quantization-aware training for reasoning models. [QaRL](https://arxiv.org/abs/2604.07853) aligns training with quantized rollouts. [RLFactory](https://arxiv.org/abs/2509.06980) supplies executable rewards for multi-turn tool use.

The pieces exist separately. The checked literature did not reveal a text-tool method that performs this complete sequence:

1. mine real compressed-agent failures;
2. turn each family into a generator and executable verifier;
3. causally locate weight and state precision through damage and rescue;
4. allocate deployable mixed precision against the worst family at a fixed physical budget;
5. train only residual model failures with the same verifiers;
6. qualify the complete artifact and runtime against the same contract.

That is a bounded literature finding, not proof that no unpublished or differently named method exists.

## 9. The first experiment that can answer the question

The first experiment should be small enough to interpret. It should test allocation before adding reinforcement learning.

### 9.1 The model and fixed environment

Use the repository’s **Qwen3-8B parent model** first. Keep the already extracted BFCL artifact for a later stress test. Starting with both extraction and quantization would make the cause of a failure ambiguous.

Hold these conditions fixed:

- weight-only quantization;
- 16-bit activations and working-state cache;
- fresh cache;
- speculative decoding off;
- one runtime, chat template, and parser;
- deterministic generation where the runtime allows it;
- a physical byte budget equal to a normal deployable 4-bit artifact;
- a minimum speed declared before quality is measured.

### 9.2 The failure families

Use at least six:

1. malformed structured output or delimiter collapse;
2. exact identifier corruption;
3. wrong choice among similar tools;
4. repeated invalid action and failure to recover;
5. collapse after long tool results or many turns;
6. calling a tool when none is needed, failing to call one when required, or stopping too early.

Calibration and held-out sets must use different tool names, schemas, identifiers, and scenario templates. Include clean tasks and collateral capabilities that the failure set does not target.

### 9.3 The five policies to compare

| Policy | What chooses precision | Question it answers |
|---|---|---|
| Original 16-bit model | No compression | What can the parent do, and how often does it already fail? |
| Uniform deployable 4-bit | Standard recipe | What does ordinary compression change? |
| Generic mixed precision | Reconstruction, activation, or similar numerical signal | Is a conventional allocator enough? |
| Ordinary task-aware mixed precision | Single-turn tool examples | Is normal task calibration enough? |
| Failure-conditioned mixed precision | Closed-loop verifier damage | Does the failure atlas produce a better allocation? |

A combined task-plus-failure policy can test whether failure data replaces or complements ordinary task examples.

### 9.4 What counts as a positive result

All of these must happen:

1. failure-conditioned rankings choose materially different components from generic and ordinary task-aware rankings;
2. the final physical artifact improves the weakest held-out failure family, with uncertainty excluding no improvement;
3. loaded bytes and measured speed stay inside the declared budget;
4. no protected collateral capability falls below its floor;
5. the gain survives unseen schemas, identifiers, and longer trajectories.

The decisive negative result is also useful: rankings do not differ, or they differ but produce no held-out physical improvement at matched cost. That result would show that the complaint atlas is a better evaluation suite, not a new allocation signal.

### 9.5 Why reinforcement learning is not experiment one

Reinforcement learning changes the model while also introducing reward design, policy drift, and train-versus-serving mismatch. That makes causal interpretation harder.

First answer one question:

> Does the failure atlas identify a different and better precision layout?

Only then should training test whether it repairs failures that allocation cannot.

## 10. State precision needs its own experiment

The working-state cache is not the same object as model weights. In transformer models, the **key-value cache**, usually called the **KV cache**, stores attention information from prior tokens so the model does not recompute the entire history at every step. Compressing it saves memory, especially for long contexts, but errors can accumulate across turns.

The focused corpus contains 26 reports involving cache quantization, compression, or related state handling. Keep this out of the first weight experiment. Then run a 2×2 comparison:

| Model weights | Working state | What the cell reveals |
|---|---|---|
| Full precision | Full precision | Parent behavior |
| Quantized | Full precision | Weight or activation effect |
| Full precision | Quantized | State-only effect |
| Quantized | Quantized | Interaction in the production-like setup |

Add fresh versus reused or rewound state. Ask which semantic roles need protection: system rules, tool schemas, exact identifiers, current plans, tool observations, or recovery messages.

For transformer state, KIVI, IntactKV, and TriAxialKV provide useful baselines. Hybrid recurrent models need a separate state design; Q-Mamba is relevant but does not prove cache-rewind correctness for other architectures.

State can fail in more ways than numerical rounding. A cached representation also has a layout, dtype, coordinate transform, position range, owner, lifetime, and creation history. The [DeepSeek-V4 quantized-cache repair](https://github.com/ggml-org/llama.cpp/pull/25202) restored a required Hadamard rotation; the values were not merely “too low precision,” they were interpreted in the wrong geometry. A state experiment must therefore record a cache receipt: model and runtime revision, token-span digest, format and dtype, slot or request owner, creation configuration, and every merge, rewind, reuse, or offload operation.

## 11. Train only after attribution

Two terms often appear in this part of the literature:

- **Quantization-aware training (QAT):** update the model while simulating or using low-precision computation, so it learns to tolerate the representation it will run with.
- **Reinforcement learning (RL):** update the model from rewards assigned to its generated behavior.

The same verifiers could eventually provide rewards for valid actions, exact state, recovery, and correct termination. A softer signal, such as probability assigned to the correct tool or identifier token, can provide more frequent training feedback.

But training should only receive failures already attributed to the model representation. A parser defect needs a parser fix. A cache-rewind defect needs a state or runtime fix. A broken kernel needs a release blocker.

Training also needs a hard comparison: it must beat the best allocation-only physical artifact. A November 2025 study, [The Impact of Quantization on Large Reasoning Model Reinforcement Learning](https://arxiv.org/abs/2511.15694), reports that naive quantization-aware RL underperformed simpler alternatives in its tested math settings. Low-bit training is not automatically beneficial.

## 12. What this work can and cannot claim

### Supported by the local evidence

- Agent-relevant failures recur around format, state, loops, runtimes, and composition.
- Many symptoms can be converted into exact programmatic checks.
- Public incident reports often lack artifacts, controls, and confirmed mechanisms.
- Different failures belong to model, state, runtime, parser, or release interventions.

### Supported by controlled external research

- Compression damage depends on agent capability and workload.
- Final scores can hide increased failures inside a tool trajectory.
- Task-aware, behavior-aware, and state-aware precision allocation already exist in adjacent forms.

### What remains a proposal

- Failure-family verifiers will rank components differently from ordinary numerical or task signals.
- That ranking will produce a better physical mixed-precision artifact at the same cost.
- The gain will transfer to unseen members of the same failure families.
- Verifier-guided training will add value after allocation is exhausted.

### Explicit non-claims

- This is not a failure-rate survey of quantization methods.
- It does not prove that GGUF is worse because GGUF appears often in complaints.
- It does not show that pruning commonly damages agents.
- It does not promise reliability outside a named model, runtime, parser, cache, context, and serving envelope.
- It is not the first agent-aware, task-aware, or behavior-guided quantization proposal.

## 13. What can make the research itself fail

The research program has its own traps:

1. **Selection bias:** failure-oriented searches overrepresent popular tools and difficult cases.
2. **Verifier overfitting:** a finite suite can become another benchmark that the model learns without gaining broad reliability.
3. **Cosmetic holdouts:** changing wording while preserving the exact causal structure does not test transfer.
4. **Runtime contamination:** a parser or kernel repair can look like a better model unless receipts are separated.
5. **Component interactions:** protecting two individually useful groups may fail when combined with the rest of the low-bit model.
6. **Fake physical wins:** irregular high-precision islands may erase memory or kernel advantages.
7. **Aggregate masking:** final success can still hide retries, loops, and token inflation.
8. **Collateral loss:** calibration for tool use can damage languages or unrelated capabilities.
9. **Reward gaming:** a trained model may satisfy the format checker while choosing the wrong action or stopping early.
10. **Open-world overclaiming:** passing known families only establishes measured reliability inside the declared envelope.

A trustworthy release names the exact artifact, runtime, parser, cache policy, context range, concurrency, and speculative settings that passed.

## 14. What a useful result would change for a user

Today, users often choose between files labeled by model name and nominal quantization level. That label leaves out much of the behavior that matters.

A failure-conditioned release would instead provide:

- a physical memory and speed receipt;
- the exact runtime and parser configuration;
- pass rates for short and long tool trajectories;
- malformed-call, wrong-action, state-drift, loop, and recovery rates;
- the error budget under which success was measured;
- the failure families and variations held out from calibration;
- explicit limits on context, cache reuse, concurrency, and speculation;
- collateral capability floors.

The user-facing question becomes more useful than “Is Q4 good?”

> **Inside this exact setup, which behaviors survived compression, where do failures begin, and how much useful work does the agent complete per second?**

## 15. Six deeper deductions worth testing

The following deductions are ranked by how much they change diagnosis, transfer across systems, and create decisive experiments. They are not all established facts. Each item separates the observed evidence from the stronger inference and names a test that could prove the inference wrong.

### 15.1 What nonsense output actually tells you

**[CORPUS]** The 220-row symptom slice in Section 4.3 reaches the same visible endpoint through kernel, state, composition, conversion, template, loading, and model-behavior paths. Thirty-four of its 37 mechanism-confirmed lineages carry a runtime, pipeline, or configuration primary label.

**[INFERENCE]** “The quant is bad” is usually too early a conclusion. Random tokens, repeated punctuation, malformed calls, and loops are analogous to a machine displaying a red warning light: they show that the final computation failed, not which component failed. A single NaN in routing, a wrong cache transform, a dropped packed tensor, a stale slot, or real representation damage can all collapse the next-token distribution.

**[PROPOSAL]** Hold the artifact hash, prompt token IDs, and greedy sampler fixed. Replay the failure through the five lanes in Section 6. If corruption follows one backend, build, template, cache history, or concurrency mode, the model representation is not the first repair target. This deduction weakens if the same first divergent logits appear across independently implemented clean runtimes.

### 15.2 A state bug can cross the boundary from bad quality to data isolation

**[PRIMARY]** A [llama.cpp incident on an integrated HIP GPU](https://github.com/ggml-org/llama.cpp/issues/25992) reported that `-np 4 --kv-unified` sometimes returned another request’s earlier response verbatim and once fused tokens from two requests. The report bisected the behavior to one commit. An independent reproduction later recorded **0/1,152** nonce violations on the control build, **1,151/1,152** on the affected build, and **0/1,152** after reverting the four-line change. That reproducer saw garbage rather than readable foreign text, so the exact symptom was not identical. The affected deployment used a Q4-class model, but the controlled evidence points to the integrated-memory runtime path, not to quantization as the cause.

**[INFERENCE]** A “hallucinated” tool name, identifier, or answer in a concurrent local agent can be a request-isolation failure. That changes the severity from quality regression to a potential privacy and integrity incident. A content-only model grader can miss the distinction.

**[PROPOSAL]** Give every request and tool result a unique nonce. Mix short and long prompts, streamed and non-streamed replies, fresh and unified caches, one and several slots, and supported backends. Reject a release if any output contains another request’s nonce, even when the answer otherwise looks plausible.

### 15.3 Some apparently random failures may be deterministic on hidden cache history

**[PRIMARY, SUGGESTIVE]** One [field report](https://x.com/t0nil0/status/2085707841357140439) claimed that a cache-busting nonce rescued **18/18** repeated low-bit tool-call corruptions and that two long reasoning loops also disappeared after a cache-busted retry. The author tied the failures to near-total prefix-cache hits and repeated microbatch chunking, but did not publish the promised reproduction harness. This is a strong clue, not a confirmed mechanism.

**[PRIMARY]** Adjacent confirmed incidents make the broader state class credible. A [DeepSeek-V4 quantized-KV incident](https://github.com/ggml-org/llama.cpp/pull/25202) omitted a required rotation; other corpus incidents lose cache metadata during merge or rewind, reuse stale state after a request leaves, or fail only on a second interaction.

**[INFERENCE]** A cache buster does not necessarily “give the model another roll.” It changes a hidden execution path: cache identity, token positions, chunk boundaries, allocation history, or kernel choice. A deterministic numerical defect can therefore masquerade as sampling randomness.

**[PROPOSAL]** Run an A/B/A state-lineage test. A is a byte-identical shared prefix; B is a semantics-preserving nonce prefix; the final A returns to the original bytes. Cross fresh, reused, rewound, and cross-configuration caches while holding weights and decoding fixed. Record cache digests, chunk boundaries, first-divergent logits, and request ownership. A failure that follows cache lineage rather than prompt meaning supports the state hypothesis; semantic-prompt failure after state randomization weakens it.

### 15.4 Low-bit damage may cross a mechanism boundary rather than worsen smoothly

**[PRIMARY]** [From Signal Degradation to Computation Collapse](https://arxiv.org/abs/2604.19884) studied factual recall under GPTQ across four model families. In the tested 4-bit failures, the correct signal still emerged in later layers, activation subspaces remained strongly aligned with FP16, and targeted protection plus signal amplification recovered performance. In the tested 2-bit models, the correct-answer signal stayed near zero, attention and feed-forward behavior collapsed, and the same targeted or low-rank compensation strategies failed. The authors stress that these modes are not universally tied to the labels “4-bit” and “2-bit.”

**[INFERENCE]** Precision allocation is useful only while the required computation remains present but weak. Below a model-and-method-specific boundary, restoring a few sensitive groups may be like repairing gauges after the engine has stopped. Training or structural reconstruction becomes the plausible intervention.

**[PROPOSAL]** Build a bitwidth phase diagram rather than testing only Q4 against BF16. Sweep at least 8, 6, 5, 4, 3, and 2 bits while recording failure-family verifiers, correct-token rank, layer-wise logit-lens signal, attention entropy, activation-subspace similarity, and bidirectional rescue. The hypothesis predicts a discontinuity in internal structure and repairability. A smooth curve with stable causal rankings would falsify the sharp-boundary account for that setup.

### 15.5 A faster token can produce a slower completed job

**[PRIMARY]** [Quantization Inflates Reasoning](https://arxiv.org/abs/2606.25519) reports that INT4 and INT3 reasoning models can preserve final accuracy while generating more reasoning tokens, intermediate steps, and semantic repetition across math, code, science, and agentic tool-use evaluations. [Flat Score, Amplified Failures](https://arxiv.org/abs/2607.27275) separately shows extra tool errors hidden inside eventual success. At the kernel layer, a [PyTorch H100 report](https://github.com/pytorch/pytorch/issues/153825) measured a representative 2:4 sparse linear operation at 0.657 ms versus 0.242 ms dense. Sparse setup and shape constraints dominated the nominal compute saving.

The end-to-end quantity is approximately

\[
T_{\text{job}}
\approx
\sum_{a=1}^{A}
\left(
T_{\text{prefill},a}
+
\frac{N_{\text{decode},a}}{R_{\text{decode},a}}
+
T_{\text{tools},a}
\right),
\]

where \(A\) includes retries, \(N_{\text{decode}}\) is generated-token count, and \(R_{\text{decode}}\) is physical decode throughput. A higher \(R_{\text{decode}}\) can lose to more tokens, retries, or failed tool rounds.

**[INFERENCE]** Tokens per second is a component metric, not the user’s efficiency objective. The release metric should be **verified completed trajectories per second and per joule**, with p50 and p95 job time, loaded bytes, token count, retries, and failure channels reported beside it.

**[PROPOSAL]** Replay identical complete jobs under BF16, a deployable Q4 policy, and one aggressive policy. Do not truncate away the extra work except at a declared safety cap. If per-token speed improves but verified completion throughput or energy worsens, the supposed optimization is a physical regression.

### 15.6 Why no component is always critical

**[PRIMARY]** [MixQuant](https://arxiv.org/abs/2607.23047) finds that a layer’s measured sensitivity changes with the bitwidths of upstream layers. [Task-Circuit Quantization](https://arxiv.org/abs/2504.07389) and [TAQ/TAQO](https://arxiv.org/abs/2511.06516) make criticality task-dependent. [TriAxialKV](https://arxiv.org/abs/2605.17170) makes state precision depend on recency, modality, and semantic role. [Mix-Quant for agentic inference](https://arxiv.org/abs/2605.20315) reports that aggressive precision was useful for prefill while precise decoding protected behavior.

**[INFERENCE]** The useful object is not a universal ranking called “important layers.” It is a conditional function:

\[
\text{criticality}
=
f(\text{failure family},\ \text{surrounding precision},\ \text{phase},\ \text{state role},\ \text{runtime}).
\]

A component can be harmless for ordinary chat, decisive for exact identifier copying, safe during prefill, and unsafe during decode. One-at-a-time saliency measured in a single background model can therefore choose the wrong final layout.

**[PROPOSAL]** Re-measure the top candidate groups under at least three surrounding precision layouts and across prefill and decode. Fit explicit pairwise interaction terms or use a budgeted marginal-contribution approximation. Report rank correlation across contexts. The conditional-criticality claim fails if rankings remain stable and interaction-aware allocation produces no held-out advantage over the simple ranking.

### 15.7 The first four experiments to steal

| Priority | Experiment | Minimum controlled grid | Decisive readout |
|---:|---|---|---|
| 1 | **State-isolation nonce harness** | Build × backend × fresh/reused state × one/many slots, fixed artifact | Zero foreign nonces; cache-history invariance; first divergent boundary |
| 2 | **Completed-work economics** | BF16 × deployable Q4 × aggressive low bit on the same complete jobs | Verified completions/s and joules/completion, including tokens, retries, and tools |
| 3 | **Precision phase diagram** | 8/6/5/4/3/2-bit sweep with internal probes and family verifiers | Smooth degradation versus a structural and repairability breakpoint |
| 4 | **Conditional allocation grid** | Top deployable groups × three background layouts × held-out failure families | Ranking stability, interaction size, and matched-cost held-out survival |

The first two require no training and can invalidate entire research branches quickly. The third decides whether mixed precision is still the right intervention. Only the fourth spends serious allocation-search budget.

## 16. A beginner’s reading path

Read these in order. Each answers one new question.

1. **Can compressed agents lose useful capability?**  
   [Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression](https://proceedings.mlr.press/v267/dong25k.html) provides the broad evaluation baseline.

2. **Can the final score hide damage inside a run?**  
   [Flat Score, Amplified Failures](https://arxiv.org/abs/2607.27275) is the closest direct evidence for the problem described here.

3. **Can precision be chosen for one task rather than the average model?**  
   [You Have One Job: Per-Task Quantization Using LLMs’ Hidden Representations](https://arxiv.org/abs/2511.06516) is the clearest first introduction to task-specific layer allocation.

4. **Can calibration examples and precision allocation be designed together?**  
   [TASA: Beyond Activation Alignment](https://arxiv.org/abs/2607.00908) joins calibration composition with mixed-precision allocation.

5. **Can working memory receive precision according to its role?**  
   [TriAxialKV](https://arxiv.org/abs/2605.17170) allocates KV precision by recency, modality, and semantic role in an agentic workload.

6. **Can lower precision be faster per token but slower for the whole answer?**  
   [Quantization Inflates Reasoning](https://arxiv.org/abs/2606.25519) measures reasoning-token inflation and its end-to-end serving cost.

7. **Can lower precision change the kind of failure rather than only its size?**  
   [From Signal Degradation to Computation Collapse](https://arxiv.org/abs/2604.19884) contrasts impaired-but-repairable computation with structural collapse.

The full method map and advanced reading list are in Appendices C and H.

## 17. Glossary

| Term | Meaning in this report |
|---|---|
| 4-bit or Q4 | A broad low-precision label, not one universal format or quantization recipe |
| Activation | A temporary numerical value produced while the model processes an input |
| Agent | A model plus software that turns outputs into actions, observes results, and continues |
| Artifact | The actual model file and metadata that a runtime loads and serves |
| BF16 or FP16 | Common 16-bit number formats used as higher-precision reference points |
| BFCL | Function-calling benchmark data and evaluation machinery used in this repository |
| Calibration data | Examples or signals used to configure a quantizer or precision policy |
| Closed loop | A run where model actions change the environment and the results affect later actions |
| Decode | Token-by-token generation after the runtime has processed the prompt |
| Dense or parent model | The original reference model before the compression step being tested |
| Failure family | Different cases that share the same testable failure mechanism or boundary |
| Generator | Code that creates controlled variations of a risky situation |
| GGUF | A model packaging format and ecosystem used by llama.cpp-style runtimes; not one quantization algorithm |
| Harness | The surrounding code that prompts the model, parses actions, runs tools, and records results |
| Kernel | Low-level code that performs numerical operations on particular hardware |
| KL divergence | A measure of how different two probability distributions are |
| KV cache | Working attention state saved from earlier tokens in a transformer |
| LLM | Large language model |
| Logit | A raw model score that is converted into a probability for a possible next token |
| Mixed precision | Using different numerical precision for different model or state components |
| MoE | Mixture of experts, a model design that activates selected expert blocks for each token |
| Perplexity | A numerical measure of next-token prediction uncertainty |
| Post-training quantization or PTQ | Quantization applied after the original model has been trained |
| Precision | How much numerical detail a representation can retain |
| Prefill | The initial processing of the prompt before token-by-token generation |
| Pruning | Removing selected weights or structures instead of only rounding values |
| Quantization | Representing weights, activations, or state with fewer bits |
| Quantization-aware training or QAT | Training that exposes the model to its lower-precision execution regime |
| Quantizer | The method or software that chooses and creates lower-precision representations |
| Reinforcement learning or RL | Updating a model from rewards assigned to generated behavior |
| Runtime | Software that loads the artifact and performs inference |
| Speculative decoding | A serving method that drafts tokens with one path and verifies them with another |
| State | Information carried from earlier tokens or turns into later computation |
| State provenance | The artifact, configuration, owner, token span, and operations that created a reused state |
| Terminal phenotype | A final visible symptom, such as gibberish or looping, that can be produced by several different causes |
| Trajectory | The complete sequence of states, actions, tool results, recoveries, and termination |
| Verifier | A programmatic check that decides whether a required condition held |

## 18. Bottom line

A smaller model is not trustworthy merely because it answers an isolated tool prompt correctly. A useful local agent must preserve valid actions, exact state, recovery, and correct stopping across the whole job.

Random tokens, wrong tools, and loops are evidence that the complete system violated its contract somewhere. They are not, by themselves, evidence that the low-bit weights are the damaged component.

The research direction is therefore:

1. collect real compressed-agent failures;
2. group them by testable mechanism or boundary;
3. turn each family into controlled generators and executable verifiers;
4. identify whether model precision, working state, runtime, or protocol causes the damage;
5. restore precision only to deployable components that causally prevent important failures;
6. keep the same real memory and speed budget;
7. train only the model failures that allocation cannot solve;
8. qualify the complete artifact and serving setup against the same tests.

This is more precise than “use better calibration data.” It is **spending a fixed compression budget to preserve an executable behavior contract**.

---

# Evidence and implementation appendices

The main report is written for a reader new to model compression. The appendices preserve the audit trail, complete tables, mathematical definition, and experiment gates.

## Appendix A. Evidence rules and complete corpus tables

Four source labels are used in these appendices:

- **[CORPUS]** directly computed from the frozen normalized complaint corpus;
- **[PRIMARY]** directly supported by a paper, official document, or linked incident;
- **[INFERENCE]** a synthesis beyond a source’s measured claim;
- **[PROPOSAL]** a research design recommended here.

The corpus is a failure-discovery collection, not a prevalence sample. Rows were selected by failure-oriented searches across heterogeneous sources. Counts show evidence surface, recurring mechanisms, and missing controls. They do not estimate the probability that a random quantizer fails.

### A.1 Evidence strength in the 101-row cohort

| Evidence class | Rows | Share |
|---|---:|---:|
| Demonstrated | 39 | 38.6% |
| Verified witness | 16 | 15.8% |
| Suggestive | 39 | 38.6% |
| Claim only | 7 | 6.9% |
| **Demonstrated or verified witness** | **55** | **54.5%** |

Only 14/101 rows have a confirmed mechanism. Forty-four carry a source hypothesis, 43 have unknown mechanism, 43 are unresolved, and 12 have unknown resolution. Ninety-one have a single source lineage. Exact model files are missing in 51 rows; a baseline or control is missing in 17; a mechanism description is missing in 39.

### A.2 Annotated operational layer

| Primary layer | Rows | Share | Strong evidence | Confirmed mechanism |
|---|---:|---:|---:|---:|
| Protocol, runtime, composition, or loading | 43 | 42.6% | 24 | 9 |
| Model/task-selective behavior | 26 | 25.7% | 10 | 1 |
| Representation, state, conversion, or numerical path | 22 | 21.8% | 14 | 3 |
| Economics, capacity, or serving overhead | 10 | 9.9% | 7 | 1 |

### A.3 Quantization surfaces

These regex-derived categories overlap. They are counts of mentions, not comparative rates.

| Family surface | Rows | Strong evidence | Confirmed mechanism | Unresolved/unknown resolution |
|---|---:|---:|---:|---:|
| GGUF K/IQ/UD, dynamic, or imatrix | 62 | 37 | 9 | 28 |
| KV-cache quantization/compression | 26 | 14 | 3 | 16 |
| NVFP4, MXFP4, W4A4, or related FP4 | 16 | 8 | 5 | 8 |
| FP8 or block FP8 | 12 | 7 | 3 | 8 |
| Ternary, binary, or 1–2 bit | 9 | 5 | 0 | 6 |
| AutoRound or RTN | 5 | 2 | 1 | 3 |
| GPTQ | 3 | 1 | 0 | 3 |
| Pruning combined with quantization | 3 | 2 | 0 | 3 |
| AWQ | 1 | 0 | 0 | 0 |
| bitsandbytes | 0 | 0 | 0 | 0 |

### A.4 Machine-checkable signatures

These categories also overlap.

| Signature | Rows | Strong evidence | Candidate invariant |
|---|---:|---:|---|
| Format, schema, parser, template, or protocol | 58 | 31 | Raw output parses and satisfies the declared schema |
| State, history, cache, context, or turn depth | 37 | 17 | Fresh and reused-state executions remain semantically invariant |
| Looping or nontermination | 26 | 15 | Completion occurs before step/token cap without repeated bad actions |
| Crash, assertion, load, or server failure | 17 | 12 | Request completes without process or scheduler failure |
| Performance or throughput | 15 | 9 | Correct trajectories per unit wall time improve under the target load |
| Explicit wrong semantic action/tool | 5 | 3 | The selected action and arguments satisfy the task contract |

### A.5 Visible-output symptom audit

**[CORPUS]** The main report’s 220-row symptom slice searched `source_title`, `observed_failure`, and `complaint_or_finding` with the case-insensitive expression `\b(?:gibberish|nonsense|garbage|repetitive|repetition|looping|loops)\b|(?:!{4,}|exclamation marks)`. The matches span 212 deduplicated source lineages and 15 primary labels.

| Primary label among mechanism-confirmed lineages | Lineages |
|---|---:|
| Kernel correctness | 15 |
| Composition incompatibility | 4 |
| Stateful precision failure | 4 |
| Conversion-pipeline damage | 4 |
| Template/tokenizer mismatch | 4 |
| Support fragmentation | 2 |
| Model loading | 1 |
| Memory representation | 2 |
| Capability-specific quality | 1 |
| **Total** | **37** |

The first seven rows form the 34-lineage runtime, pipeline, or configuration subtotal used in Section 4.3. The taxonomy was assigned during corpus construction, so this audit shows causal diversity inside the collected evidence; it does not estimate population prevalence.

## Appendix B. Thirteen incident-to-test cards

The route is conservative because an observed boundary does not always identify the cause.

| Incident | Observed boundary | Generator and verifier | Route |
|---|---|---|---|
| [Qwen3.5 MLX multi-turn degradation](https://github.com/ml-explore/mlx-lm/issues/1011) | MLX 4-bit emitted plain-text tool calls around round 5; 8-bit around round 13; GGUF Q4_K_XL and cloud full precision reportedly completed 70 rounds | Sweep turn count, format, and runtime with the same transcript semantics; require every round to emit a recognized schema-valid call; record first divergent raw token | Representation/calibration candidate, but parser and runtime must be separated first |
| [llama.cpp Qwen3.5 cache rewind drift](https://github.com/ggml-org/llama.cpp/issues/21681) | After history mutation and prefix rewind, exact identifiers silently changed, e.g. `isced1997_r` to shorter variants | Replay identical final token IDs with fresh state and reused/rewound state; require exact identifiers and equivalent next-token distributions | Runtime/cache correctness gate; do not train around corrupted state |
| [DeepSeek-V4 PD HiSparse](https://github.com/sgl-project/sglang/issues/31482) | GPQA remained high, but 144/152 SWE tasks reached 300 steps and four timed out amid malformed or repetitive output | Sweep trajectory horizon and history compression; verify task completion, nonempty valid calls, first bad action, and post-error amplification | State-compression policy; robustness training only after runtime attribution |
| [REAP pruning versus extreme quantization](https://x.com/superalesha/status/2085067196703600740) | Reported agent scores were 118/150 versus 117/150 at the same 90 GB footprint, while the pruned model reportedly lost non-English capability | Match physical budget; stratify held-out trajectories by language and domain; require both target-agent retention and collateral floors | Calibration coverage and pruning/precision policy; evidence remains suggestive |
| [Unsloth Qwen3-Coder refresh](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/discussions/21) | A refreshed calibration/imatrix release produced formatting, instruction, and degenerate-output reports | Immutable artifact revisions; factor calibration mixture, imatrix, layer policy, and template one at a time; test dense-relative KL plus executable code/tool contracts | Quantizer recipe qualification; no causal calibration claim without factorial rebuild |
| [GLM-5.1 FP8 tool-use corruption](https://github.com/NVIDIA/TensorRT-LLM/issues/15295) | At temperature zero, output ran to length with repetitive/random numeric, URL, and mixed-language text and no tool call | Sweep tool-description/system-prompt length across FP8/BF16 and runtimes; require finite nonflat logits, coherent greedy output, and expected call | Activation/weight precision or kernel path; cross-runtime gate |
| [MiniMax NVFP4 backend corruption](https://github.com/sgl-project/sglang/issues/26324) | One backend returned deterministic replacement-character-heavy text; another model asserted on first forward pass | Replay identical checkpoint/input across MoE backends; compare pre-decode tensors/logits and reject replacement degeneration or assertions | Confirmed kernel/runtime class; not an RL target |
| [DSpark draft-depth cliff](https://github.com/sgl-project/sglang/issues/33800) | Draft depth 5 produced rare corruption; tested depths 3, 4, 6, and 7 produced zero events | Sweep adjacent depths over repeated greedy trials; verify zero corruption within a declared trial budget and compare token-acceptance traces | Speculative runtime policy and release gate |
| [Combined calibrated quant regression](https://www.reddit.com/r/LocalLLaMA/comments/1vbbp2n/closed_the_biggest_gap_from_my_last_quant_project/) | Isolated quantization tests passed while the complete combined artifact failed; reverting small attention/state tensors helped | Compare isolated tensor groups with the combined artifact; perform bidirectional upgrade/downgrade interventions; require full-artifact task and KL contracts | Direct mixed-precision group-allocation candidate |
| [Nonce broke repeated low-bit corruption](https://x.com/t0nil0/status/2085707841357140439) | A cache-busting nonce reportedly rescued 18/18 repeated corrupt calls after a shared prefix | Hold semantics fixed; compare byte-stable and nonce-busted prefixes across chunking/microbatch choices; require call invariance and measure rescue rate | Cache/chunking investigation; suggestive, not yet causal |
| [Ternary Bonsai looping/tool failure](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/discussions/41) | Looping, hallucination, and unusable tool-call forms reportedly improved after KV precision changes | Hold weights and prompts fixed; sweep F16/Q8/Q4 KV; require valid calls, bounded repeated n-grams, and successful completion | KV precision candidate; preserve exact reproduction details |
| [Speculation and grammar composition failure](https://github.com/sgl-project/sglang/issues/16541) | One speculative version crashed; another failed structured output without crashing; grammar modes differed | Factor speculative version, grammar type, and serving topology; require every schema request to return accepted valid structure without scheduler failure | Runtime/grammar composition gate |
| [llama.cpp integrated-HIP cross-request replay](https://github.com/ggml-org/llama.cpp/issues/25992) | Under mixed concurrent load, one build returned earlier requests’ nonce responses or chimeric output; an independent 1,152-check reproduction failed on the affected build and passed before and after the targeted revert | Assign every request and tool result a unique nonce; cross build, backend, slot count, unified-cache mode, prompt length, and streaming; require each response to contain only its own nonce | Release-blocking state-isolation and security gate; runtime repair, not model training |

## Appendix C. Where failure verifiers can enter existing methods

Keep five object types separate:

- **methods or objectives:** GPTQ, AWQ, AutoRound, TASA, TAQ/TAQO, Task-Circuit Quantization, SliM-LLM, GuidedQuant;
- **numerical formats:** FP8, NVFP4, MXFP4, W4A4, ternary, binary;
- **deployment encodings and recipes:** GGUF K-quants, IQ-quants, imatrix-guided recipes;
- **state policies:** KIVI, IntactKV, TriAxialKV, Q-Mamba;
- **training and rollout regimes:** ParetoQ, UPQ, Reasoning-QAT, QaRL, QuRL.

The operational question is not which acronym sounds best. It is: **what decision can the verifier change?**

| Method family | Decision exposed today | Current signal | Where the failure verifier enters | Assessment |
|---|---|---|---|---|
| [TASA](https://arxiv.org/abs/2607.00908) | Calibration mixture plus inter-/intra-layer bits | Gradient-trace alignment, perplexity, reasoning sensitivity | Replace or augment reasoning sensitivity with family-specific trajectory process loss; search mixtures that cover failure mechanisms | Closest task-aware allocator; no agent trajectories in the paper |
| [You Have One Job: Per-Task Quantization Using LLMs’ Hidden Representations (TAQ/TAQO)](https://arxiv.org/abs/2511.06516) | Per-layer bit width | Hidden-representation stability or direct layer/output sensitivity | Score each layer intervention by first-failure hazard and process errors | Best simple first scaffold; layer granularity may miss small circuits |
| [Task-Circuit Quantization](https://arxiv.org/abs/2504.07389) | Individual task-associated weights retained at 16 bit | Expected quantization change times task gradient | Differentiate a decision-token surrogate, then validate candidate circuits with executable trajectories | Fine-grained and task-specific; irregular storage may erase systems gains |
| [SliM-LLM](https://arxiv.org/abs/2405.14917) | Structured group bit width | Generic salience and salience-weighted calibration | Replace salience with conditional marginal verifier damage | Strong hardware-conscious scaffold |
| [GuidedQuant](https://arxiv.org/abs/2505.07004) | Output-channel quantization objective | Final LM-loss gradients and cross-weight interactions | Use differentiable proxies for action, schema, identifier, recovery, and termination margins; retain the executable verifier as the outer gate | Best bridge from local reconstruction to end behavior; proxies can miss delayed discrete failure |
| [OWQ](https://arxiv.org/abs/2306.02272), [SpQR](https://arxiv.org/abs/2306.03078), [CherryQ](https://arxiv.org/abs/2404.02837), [Super Weight](https://arxiv.org/abs/2411.07191) | Protected columns or sparse individual parameters | Generic quantization sensitivity, outliers, or generation impact | Redefine “critical” as causal damage to a declared verifier family | Useful localization baselines; unstructured high-precision side data may not accelerate |
| [MixQuant](https://arxiv.org/abs/2607.23047) | Layer allocation under variable budgets and upstream bit interactions | Distortion marginalized over upstream quantization plans | Marginalize verifier damage over random prior allocations; re-evaluate coalitions | Essential correction to naive one-layer-at-a-time scoring |
| [Unsloth Dynamic 2.0](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs) and [llama.cpp imatrix](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md) | Tensor/layer quant type in deployable GGUF recipes | KL, activation importance, heuristics, calibration corpus | Use trajectory sensitivity to protect tensor types; qualify the same GGUF in the target runtime | Fastest path to a real local artifact; exact recipe disclosure can be incomplete |
| [AutoRound](https://arxiv.org/abs/2309.05516), GPTQ, AWQ | Block/group rounding and sometimes mixed bits or protected channels | Reconstruction, approximate curvature, or activation salience | Supply failure-conditioned calibration cases and use an outer allocation loop | Mature baselines; their native objectives remain local |
| [MicroMix](https://arxiv.org/abs/2508.02343) | MXFP4/6/8 channels | Activation-error thresholds | Rank channel upgrades by held-out trajectory benefit before fitting thresholds | Hardware-real Blackwell path; platform-specific |
| [Mix-Quant for agentic prefill](https://arxiv.org/abs/2605.20315) | NVFP4 prefill versus BF16 decode | Phase-level agent benchmark tolerance | Use as an orthogonal baseline: test whether protecting decode alone covers the failures | Existing agent-aware policy; does not localize circuits and leaves decode high precision |
| [KIVI](https://arxiv.org/abs/2402.02750), [IntactKV](https://arxiv.org/abs/2403.01241) | KV channels/tokens or pivot tokens | KV distributions and generic token importance | Protect system/tool schema, exact identifiers, plan/recovery tokens, and failure-sensitive channels | State-precision foundation; not recurrent-state rewind correctness |
| [TriAxialKV](https://arxiv.org/abs/2605.17170) | KV bits by recency × modality × semantic role | Per-tag agent-task sensitivity | Replace aggregate tag sensitivity with family-specific hazard and state-transition contracts | Closest agent-aware KV allocator; shown on a VLM computer-use workload |
| [Q-Mamba](https://aclanthology.org/2025.findings-acl.551/) | Recurrent-state channel/state-dimension precision | State outliers and selectivity reconstruction | Score recurrent state under history edits, prefix rewind, and long loops | Important for hybrid recurrent agents; architecture transfer is unproven |
| [ParetoQ](https://arxiv.org/abs/2502.02631), [UPQ](https://arxiv.org/abs/2506.09104), [Reasoning-QAT](https://arxiv.org/abs/2601.14888) | Whole low-bit representation learning | Distillation/QAT loss and reasoning tasks | Add family rewards and a failure-boundary curriculum after a viable PTQ cold start | Most relevant in the 2-bit regime or after allocation saturates |
| [QaRL](https://arxiv.org/abs/2604.07853), [QuRL](https://arxiv.org/abs/2602.13953), [RLFactory](https://arxiv.org/abs/2509.06980) | Quantized rollout/training alignment plus multi-turn tool rewards | RLVR, policy-ratio controls, rule/tool-verification rewards | Train in the actual quantized execution regime with executable family rewards | The necessary pieces exist separately; naive quantized RL has negative evidence |
| [ActQuant](https://arxiv.org/abs/2605.24011) | Inter-tensor bits and intra-tensor scales for VLA models | Action sensitivity and curvature | Replace embodied action loss with symbolic action and trajectory contracts | Strong conceptual precedent; prevents a broad “first behavior-guided quantizer” claim |

### C.1 Novelty boundary

The closest weight-allocation sequence is TASA, You Have One Job (TAQ/TAQO), Task-Circuit Quantization, GuidedQuant, and ActQuant. The closest agent-execution sequence is ACBench, Flat Score, Mix-Quant, TriAxialKV, and RLFactory.

The proposal is the bridge between those sequences. Targeted searches for `verifier-guided`, `failure-conditioned`, `trajectory-aware`, `tool-aware`, and `agent-aware` quantization found adjacent work but no text-LLM method implementing the full six-step sequence stated in Section 8. This is a checked-literature boundary, not proof of absence.

## Appendix D. Optional mathematical definition

A tool-agent trajectory can be written as

\[
\tau=(s_0,a_0,o_1,s_1,a_1,o_2,\ldots,s_T),
\]

where `s` is agent/runtime state, `a` is a raw or parsed action, and `o` is an observation or tool result.

For failure family \(f\), define a step verifier \(v_f(\tau,t)\in\{0,1\}\). Examples include:

- declared tool name;
- schema-valid arguments;
- exact identifier retention;
- no repeated invalid action;
- correct response to tool error;
- state equality under fresh versus replayed execution;
- successful termination before the step cap.

Define first-failure survival:

\[
S_f(t;\mathbf b)=P\!\left(\bigwedge_{j=1}^{t}v_f(\tau,j)=1\;\middle|\;\mathbf b\right),
\]

where \(\mathbf b\) is the precision policy over weights, activations, KV state, recurrent state, and possibly phases.

A simple family process loss is one minus restricted mean survival:

\[
L_f(\mathbf b)=1-\frac{1}{T}\sum_{t=1}^{T}S_f(t;\mathbf b).
\]

Do not average families into one forgiving score. Use a min-max objective:

\[
\min_{\mathbf b}\;\max_{f\in\mathcal F}
\frac{\max\{0,L_f(\mathbf b)-L_f(\mathrm{FP})\}}{\epsilon_f}
\]

subject to

\[
\mathrm{loaded\_bytes}(\mathbf b)\le B,\qquad
\mathrm{prefill\_throughput}(\mathbf b)\ge P_{\min},\qquad
\mathrm{decode\_throughput}(\mathbf b)\ge D_{\min}.
\]

Here \(\epsilon_f\) is the allowed dense-relative degradation for family \(f\). The physical constraints prevent a nominal 4-bit policy from winning by adding irregular high-precision side data or falling back to slow kernels.

### D.1 Measure causal component importance

For component group \(g\), measure both directions:

\[
I^{\downarrow}_{g,f}=L_f(\mathbf b_{g\downarrow})-L_f(\mathbf b),
\]

\[
I^{\uparrow}_{g,f}=L_f(\mathbf b)-L_f(\mathbf b_{g\uparrow}).
\]

A credible critical group should hurt when downgraded and rescue when upgraded. Because the corpus contains a demonstrated combined-artifact failure and MixQuant shows upstream bit interactions, one-at-a-time importance is not enough. Estimate conditional marginal damage over random or allocator-produced surrounding precision plans, then test the strongest pairs or coalitions.

### D.2 Hard verifiers and differentiable proxies have different jobs

- **Hard executable verifiers** decide calibration membership, outer-loop allocation, acceptance, and release.
- **Differentiable proxies** make optimization affordable: decision-token margins, correct-tool probability, schema-token likelihood, identifier-copy margin, stop-versus-loop margin, and dense-relative KL at verified boundary tokens.
- The proxy never replaces the hard verifier. A model can assign good likelihood to a call and still fail in parsing, execution, state update, or recovery.

## Appendix E. Full research program and stage gates

### Stage 0 declares the executable capability envelope

Freeze:

- model and exact revision;
- quantizer and configuration;
- runtime and commit;
- hardware;
- chat template and parser;
- context range;
- cache policy;
- batch/concurrency policy;
- speculative-decoding policy;
- hard model-byte and throughput budgets;
- target and collateral capability floors.

**Gate:** no comparison advances unless it can run inside the same envelope.

### Stage 1 converts incidents into generators and verifiers

For each family, store:

- source incident and evidence grade;
- minimal trigger dimensions;
- controlled generator;
- raw token receipt;
- parser receipt;
- executed action/result;
- state snapshot or digest;
- final task contract;
- dense and runtime controls;
- split lineage.

Split by **failure-family lineage**, not random prompt rows. Held-out schemas, tool inventories, languages, history mutations, and longer horizons must not be near-duplicates of calibration examples.

**Gate:** the family is reproducible enough to estimate a rate and to distinguish model, parser, runtime, and state failure.

### Stage 2 establishes compression-caused process damage

Report:

- per-channel error rates;
- first-failure survival \(S_f(t)\);
- conditional failure hazard;
- success as the allowed-error budget is tightened;
- repeated-invalid-action and recovery rates;
- reasoning/recovery token inflation;
- fresh-state versus reused/rewound-state divergence;
- final task success, but never alone.

**Gate:** only families with a reproducible dense-to-quantized gap advance to model-component localization. Runtime-only failures go to a runtime issue and release gate.

### Stage 3 localizes causal precision and state components

Start with deployable groups:

- embeddings and output head;
- per-layer attention Q/K/V/O matrices;
- per-layer MLP gate/up/down matrices;
- small attention/state tensors;
- MoE router and expert groups;
- KV semantic/recency groups;
- recurrent-state channel and state dimensions.

Run bidirectional upgrade/downgrade interventions. Hold tokenized input, runtime, template, parser, cache policy, and sampling fixed. Repeat where kernels or batching are nondeterministic.

**Gate:** a “critical component” must show both damage and rescue, not merely high activation or gradient magnitude.

### Stage 4 allocates precision under physical constraints

Compare four signals:

1. generic reconstruction/KL/perplexity;
2. ordinary task-aware single-turn calibration;
3. failure-aware process sensitivity;
4. combined task plus failure signal.

Build actual artifacts and plot quality against:

- on-disk and loaded bytes including scales/metadata;
- prefill tokens/s;
- decode tokens/s;
- peak memory;
- correctly completed trajectories/s;
- worst-family survival regret.

**Gate:** the failure-conditioned policy must improve held-out worst-family survival at matched loaded bytes and measured throughput.

### Stage 5 trains residual representation failures

If allocation alone cannot meet the envelope:

1. initialize from the best PTQ policy;
2. distill from the dense parent;
3. align the training forward pass with the quantized rollout path;
4. use differentiable boundary-token proxies for dense credit;
5. use executable family verifiers for terminal and process rewards;
6. retain clean-task and collateral-capability rewards;
7. evaluate reward gaming under unseen schemas and longer horizons.

This sequence follows the positive evidence from [Reasoning-QAT](https://arxiv.org/abs/2601.14888) and mismatch-aware [QaRL](https://arxiv.org/abs/2604.07853). It also respects a negative result: naive quantization-aware RL underperformed PTQ and QLoRA in tested math settings in [The Impact of Quantization on Large Reasoning Model RL](https://arxiv.org/abs/2511.15694).

**Gate:** training must beat the best allocation-only physical artifact. Otherwise keep the cheaper PTQ result.

### Stage 6 qualifies the complete artifact

Run the same contracts on:

- the converted physical model;
- the target runtime and parser;
- the actual KV/recurrent-state precision;
- fresh and reused/rewound cache;
- target concurrency;
- target speculative path;
- context and tool-schema bounds declared for release.

**Gate:** release the artifact only inside the envelope it actually passed. A logical mask, fake-quant model, or isolated layer test is not a receipt for the deployed object.

## Appendix F. Detailed first-experiment protocol

### F.1 Use the parent before the extracted artifact

[PROPOSAL] Use the repository’s **Qwen3-8B parent** for the first precision-allocation experiment. Keep the existing physically extracted BFCL artifact as a later stress test. Starting from the extracted artifact would confound channel extraction, physical conversion, and quantization in the first causal study.

### F.2 Fixed envelope

- Weight-only quantization first.
- BF16/FP16 activations and KV cache.
- Fresh cache.
- Speculative decoding off.
- One runtime, template, and parser.
- Temperature zero plus repeated trials where the serving path is nondeterministic.
- Physical budget equal to a conventional deployable 4-bit artifact, including metadata and scales.
- Throughput floor declared before evaluating quality.

This deliberately excludes state compression and production composition. Those become the next factorial study after the weight-allocation result is known.

### F.3 Failure families

Use at least these six families:

1. structured-output or delimiter collapse;
2. exact identifier corruption;
3. wrong tool among semantically overlapping tools;
4. repeated invalid action and failure to recover;
5. long tool output and turn-depth collapse;
6. no-call/tool-call boundary and premature termination.

For each family, generate calibration and held-out sets with disjoint tool names, schemas, identifiers, and scenario templates. Evaluate short, medium, and longer horizons. Include clean controls where no tool should be called and collateral tasks not represented in the calibration mix.

### F.4 Compared policies

| Policy | Signal | Purpose |
|---|---|---|
| BF16/FP16 | None | Parent capability and failure propensity |
| Uniform deployable 4-bit | Standard recipe | Compression baseline |
| Generic mixed precision | KL, activation, or reconstruction sensitivity | Conventional allocator baseline |
| Ordinary task-aware mixed precision | Single-turn BFCL/tool prompts | Tests whether ordinary task data is enough |
| Failure-conditioned mixed precision | Closed-loop verifier damage | Proposed policy |

A sixth combined policy, ordinary task plus failure signal, can determine whether failure data replaces or complements standard task calibration.

### F.5 Search procedure

1. Build a low-bit baseline artifact.
2. Upgrade one deployable group at a time; measure family-specific rescue.
3. Starting from high precision, downgrade the same groups; measure damage.
4. Re-score leading groups under random surrounding allocations to account for interaction.
5. Test the strongest pairs and small coalitions.
6. Solve the fixed-byte allocation with a min-max family objective.
7. Convert every finalist to the target physical format.
8. Re-run the held-out suite and physical throughput measurement.

### F.6 Primary metrics

- restricted mean first-failure survival by family;
- success versus allowed-error budget;
- valid/correct tool and argument rates;
- exact identifier retention;
- repeated-invalid-action rate;
- recovery-after-error rate;
- completion before step cap;
- extra reasoning/recovery tokens;
- target and collateral task success;
- loaded bytes, peak memory, prefill/decode tokens/s;
- **correctly completed trajectories per second**.

### F.7 Positive and negative results

A positive result requires all of the following:

1. failure-conditioned component ranking differs materially from generic and ordinary task-aware rankings;
2. the physical policy improves held-out worst-family survival with confidence intervals excluding no improvement;
3. loaded bytes and measured throughput satisfy the predeclared budget;
4. no collateral capability falls below its floor;
5. the gain survives unseen schemas, identifiers, and longer horizons.

The decisive negative result is equally useful:

- rankings do not differ, or
- they differ but produce no held-out physical gain at matched cost.

Either result should be published. It directly tests whether the complaint atlas becomes a new optimization signal rather than merely a better benchmark.

### F.8 Why RL is not in experiment one

RL introduces policy drift, reward design, quantized-rollout mismatch, and additional attribution ambiguity. The first experiment asks a cleaner question:

> Does the failure atlas identify a different, better precision allocation?

Only after that question is answered should the same verifiers become QAT/RL rewards.

### F.9 Separate state experiment

The corpus contains 26 tool-calling incidents involving KV-cache quantization, compression, or related state handling. State should not be folded into the first weight experiment.

Run a 2×2 attribution grid:

| Weights | State | Interpretation |
|---|---|---|
| Full precision | Full precision | Parent control |
| Quantized | Full precision | Weight/activation effect |
| Full precision | Quantized | KV or recurrent-state effect |
| Quantized | Quantized | Interaction and production configuration |

Then add fresh versus reused/rewound state.

For transformer KV, compare homogeneous precision, KIVI/IntactKV-style structural policies, and TriAxialKV-style semantic roles. For hybrid recurrent architectures, include recurrent state explicitly; Q-Mamba demonstrates that state-channel and state-dimension scaling can matter, but it does not validate cache rewind or DeltaNet semantics.

A failure-conditioned state policy should ask:

- Which tokens must preserve exact identity?
- Which semantic roles (system rules, tool schemas, plans, observations, recovery messages) need more precision?
- Which state channels accumulate errors across turns?
- Which rewind or reuse operations violate state equivalence?

## Appendix G. Research risk register and falsifiable hypotheses

### G.1 Complaint-selection bias

The corpus overrepresents popular local formats, difficult failures, and users willing to report them. Use it to generate hypotheses, not estimate rates.

### G.2 Benchmark and verifier overfitting

A finite suite can become another narrow benchmark. Split by generator lineage and evaluate unseen schemas, languages, tool inventories, horizons, and state mutations.

### G.3 Conflating failure family with prompt surface

The same schema text with renamed tools is not a held-out mechanism. Hold out causal dimensions, not cosmetic wording.

### G.4 Runtime contamination

A parser, cache, or kernel repair can look like a better model. Preserve raw tokens, parser decisions, execution results, and state receipts separately.

### G.5 Non-additive precision effects

Isolated layer or tensor success does not prove the combined artifact will work. The corpus contains a direct counterexample; MixQuant independently shows upstream allocation changes downstream sensitivity.

### G.6 Hardware-fake mixed precision

A mathematically elegant sparse set of FP16 weights may add metadata, trigger dequantization, or lose kernel efficiency. Count physical bytes and measure the actual runtime.

### G.7 Aggregate-score masking

Final success can hide extra retries, errors, tokens, and recovery. Report process channels, survival, error-budget curves, and correctly completed trajectories/s.

### G.8 Calibration-distribution specification

A high target score can coexist with erased languages or collateral behaviors. Treat calibration composition as a capability specification and retain explicit collateral floors.

### G.9 RL reward gaming and mismatch

A model can satisfy format checks while choosing the wrong action, terminate early to avoid errors, or exploit a parser. Use layered verifiers and quantized-runtime-aligned rollouts; keep held-out executable contracts.

### G.10 Open-world reliability overclaim

Failure-family saturation inside a declared envelope is not “almost all failures.” The defensible claim is improved measured reliability within that envelope.

### G.11 Falsifiable hypotheses

1. **Process-metric replication.** Uniform low-bit quantization increases at least one pre-existing process-failure channel without a commensurate final-score change.
2. **Ranking divergence.** Failure-conditioned precision rankings have low enough agreement with perplexity/activation and ordinary task-aware rankings to change the chosen artifact.
3. **Fixed-budget benefit.** FC-MPQ improves worst-family held-out trajectory survival at matched loaded bytes and measured throughput.
4. **Family transfer.** The gain transfers to unseen prompts and schemas from the same failure family.
5. **State separability.** Weight, KV, recurrent-state, runtime, and parser lanes explain distinct subsets of failures.
6. **Training increment.** Quantized-runtime-aligned QAT/RL adds robustness beyond the best allocation-only artifact.

The program fails its central claim if hypothesis 2 or 3 fails. Hypothesis 1 is now largely replication and extension because the July 2026 paper already demonstrated the phenomenon in a controlled setting.

The central claim fails if the component rankings do not differ in a useful way or if they produce no held-out physical gain at matched cost. The first hypothesis is now mainly replication and extension because Flat Score, Amplified Failures already demonstrated process amplification in a controlled setting.

## Appendix H. Full advanced reading list

1. [Flat Score, Amplified Failures](https://arxiv.org/abs/2607.27275): controlled evidence for hidden closed-loop quantization damage.
2. [Can Compressed LLMs Truly Act? / ACBench](https://proceedings.mlr.press/v267/dong25k.html): broad agent-compression evaluation.
3. [TASA: Beyond Activation Alignment](https://arxiv.org/abs/2607.00908): calibration mixture and mixed-precision allocation.
4. [Task-Circuit Quantization](https://arxiv.org/abs/2504.07389): task-conditioned critical-weight preservation.
5. [You Have One Job: Per-Task Quantization Using LLMs’ Hidden Representations](https://arxiv.org/abs/2511.06516): per-task layer allocation through TAQ/TAQO.
6. [GuidedQuant](https://arxiv.org/abs/2505.07004): final-model loss rather than only local reconstruction guidance.
7. [Preserving LLM Capabilities through Calibration Data Curation](https://arxiv.org/abs/2510.10618): calibration composition as capability retention.
8. [MixQuant](https://arxiv.org/abs/2607.23047): non-additive upstream allocation effects.
9. [SliM-LLM](https://arxiv.org/abs/2405.14917): structured, salience-driven mixed precision.
10. [OWQ](https://arxiv.org/abs/2306.02272), [SpQR](https://arxiv.org/abs/2306.03078), [CherryQ](https://arxiv.org/abs/2404.02837), and [The Super Weight in Large Language Models](https://arxiv.org/abs/2411.07191): outlier and critical-parameter preservation.
11. [KIVI](https://arxiv.org/abs/2402.02750), [IntactKV](https://arxiv.org/abs/2403.01241), and [TriAxialKV](https://arxiv.org/abs/2605.17170): structured and role-aware KV-cache precision.
12. [Q-Mamba](https://aclanthology.org/2025.findings-acl.551/): recurrent-state quantization.
13. [ActQuant](https://arxiv.org/abs/2605.24011): action-guided precision for embodied agents.
14. [Reasoning-QAT](https://arxiv.org/abs/2601.14888), [QaRL](https://arxiv.org/abs/2604.07853), and [QuRL](https://arxiv.org/abs/2602.13953): low-bit training and rollout alignment.
15. [RLFactory](https://arxiv.org/abs/2509.06980): executable multi-turn tool rewards.
16. [Tracing Agentic Failure from the Flow of Success](https://arxiv.org/abs/2607.12747): trajectory-step failure attribution without manual step labels.
17. [From Signal Degradation to Computation Collapse](https://arxiv.org/abs/2604.19884): distinct post-training quantization failure modes.
18. [MicroMix](https://arxiv.org/abs/2508.02343): hardware-oriented microscaling mixed precision.
19. [Mix-Quant: Quantized Prefilling, Precise Decoding](https://arxiv.org/abs/2605.20315): phase-aware precision for agentic inference.
20. [The Impact of Quantization on Large Reasoning Model Reinforcement Learning](https://arxiv.org/abs/2511.15694): negative evidence for naive quantization-aware RL in tested math settings.
21. [Quantization Inflates Reasoning](https://arxiv.org/abs/2606.25519): reasoning-token inflation as an end-to-end efficiency cost hidden by accuracy and per-token latency.
