<h1 align="center">
  <img src="assets/logo.png" alt="LapMoE - Big models. Small laptops. No excuses." width="520">
</h1>

<p align="center"><b>Yes, a mid-size model can run on your laptop.</b></p>

<p align="center">
  <img src="assets/hero.png" alt="The LapMoE mascot, a hooded girl, carrying a huge MoE model on her back while typing on a laptop" width="640">
</p>

One command sets up a local server for a 35B-parameter Mixture-of-Experts model (Qwen3.6-35B-A3B) on a
modest PC or laptop — an 8 GB NVIDIA GPU with 16 GB of RAM is enough.

It installs and wires together three things:

| Piece | What it is |
|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | the inference engine, built from a pinned commit for your GPU (CUDA) |
| [llama-swap](https://github.com/mostlygeek/llama-swap) | an OpenAI-compatible server that starts/stops models on demand |
| a GGUF model | **Qwen3.6-35B-A3B** (Unsloth quant) picked for your RAM, downloaded and checksum-verified |

Why a 35B model fits: it is a Mixture-of-Experts model with only ~3B parameters active per token. The experts
live in system RAM and the always-used parts live on the GPU, so you need RAM more than VRAM.

## Quick start

```bash
git clone https://github.com/kyudongkim/LapMoE.git && cd LapMoE
./install.sh --dry-run      # see what it detected and what it would do, change nothing
./install.sh                # do it (asks before the big download)
~/lapmoe/start.sh        # start the server on http://127.0.0.1:8080/v1
~/lapmoe/test.sh         # smoke test (first call loads the model, 10-60 s)
```

<p align="center">
  <img src="assets/pose-idle.png" alt="mascot, idle" height="230">&nbsp;
  <img src="assets/pose-loading.png" alt="mascot carrying the model, loading" height="230">&nbsp;
  <img src="assets/pose-success.png" alt="mascot celebrating, it runs" height="230">
  <br><sub><code>./install.sh</code> &rarr; building, downloading, loading&hellip; &rarr; running</sub>
</p>

Point any OpenAI-compatible client at `http://127.0.0.1:8080/v1` and use the model name `default`.

Requirements: Linux (x86_64 or aarch64), an **NVIDIA GPU** with the CUDA toolkit (`nvcc`), `git cmake make g++ curl python3`,
~25 GB free disk (more for the larger quants). There is no CPU-only mode. On Debian/Ubuntu `./install.sh --install-deps`
installs the compiler/cmake prerequisites with `sudo apt-get`. The script never uses sudo otherwise.

## What it picks for your machine

The model file must fit in RAM (the experts are memory-mapped), keeping ~3 GiB for the OS and desktop:

| RAM (total) | Chosen model | Status |
|---|---|---|
| ≥ 23.6 GiB | `UD-Q4_K_M` (≈ 20.6 GiB) | heuristic, not measured |
| ≥ 18.5 GiB | `UD-Q3_K_M` (≈ 15.5 GiB) | the file itself was measured to be **too big for 16 GB RAM** (heavy paging) |
| ≥ 15.3 GiB | **`UD-IQ3_XXS` (≈ 12.3 GiB)** | **measured, see below** |
| ≥ 14.5 GiB | `UD-Q2_K_XL` (≈ 11.5 GiB) | heuristic, not measured |
| ≥ 13.1 GiB | `UD-IQ2_XXS` (≈ 10.1 GiB) | heuristic, not measured |
| less | Qwen3-8B (dense, 3.4-4.3 GB) | small fallback, heuristic |

The rule is simply "model file + 3 GiB ≤ total RAM". A "16 GB" machine usually reports a bit less than 16 GiB
(integrated graphics, firmware); the author's reports 15.5 GiB, which is just enough for `UD-IQ3_XXS`.

Override with `--quant`, `--profile small|mid`, or `--model-path` (use a GGUF you already have).
Run `./install.sh --dry-run` first to see the choice.

GPU offload is **automatic** (`llama-server --fit`: llama.cpp decides how much goes to VRAM) except for one
preset that was tuned by hand: **`mid-8gb-tuned`** (RTX 2070 Super Max-Q 8 GB + 16 GB RAM, `UD-IQ3_XXS`:
`--n-cpu-moe 28 -ngl 99 -ub 2048 --ctx-size 131072`). It is generated only on 8 GB-class NVIDIA cards with
≥ 15 GB RAM and becomes the `default` model there. The automatic `mid` entry is always generated too.

## Measured performance (one laptop — your numbers will differ)

<img src="assets/expr-happy.png" alt="" align="right" height="90">

**Test machine:** MSI GS66 Stealth 10SFS — Intel i7-10875H (8 cores / 16 threads), 16 GB DDR4-3200 (2×8 GB, dual channel),
NVIDIA RTX 2070 Super Max-Q **8 GB**, Ubuntu 24.04. llama.cpp build 9612 (`a01eb2646`), llama-swap v250,
model `Qwen3.6-35B-A3B` `UD-IQ3_XXS` (13.2 GB), KV cache q4_0.

**Even with the CPU turbo limited to 70 %** (the laptop's setting during the generation-speed runs; its fans / "cooler
boost" ran automatically when hot), this 35B model generates at **35+ tokens/s**. Every table states its own conditions.

### Generation speed (tokens/s while generating)

Continuous generation for 3 minutes, short prompts, 64k context, default `-ub`; `--n-cpu-moe N` = how many layers keep
their experts in system RAM (the rest are on the GPU):

| `--n-cpu-moe` | decode tok/s | CPU package power | CPU temp (sustained) |
|---|---|---|---|
| 26 | **36.5** | 50 W | 95 °C |
| 30 | **35.0** (34.5 over a 5-minute run) | 50 W | 95 °C |
| 34 | 33.2 | 50 W | 95 °C |

* The shipped `mid-8gb-tuned` preset uses 28 (so it should land between the 26 and 30 rows; not measured at this cap).
  Measured at a 50 % CPU cap with the preset's 131k context it gave 27.5 tok/s.
* Reference on the same machine: the previous-generation `Qwen3-30B-A3B Q4_K_M` ran at ≈ 19-22 tok/s.
* Sustained generation is CPU-heavy: expect ~95 °C and loud fans at this setting. If your laptop cannot hold that,
  cap the CPU (`/sys/devices/system/cpu/intel_pstate/max_perf_pct`): at 50 % the author got 24-31 tok/s at ~84 °C with
  the fans not even needing boost, and 35-45 % less energy per token.

### Prompt processing (tokens/s while reading the prompt)

Measured with the CPU cap at 50 %, 12-23k-token prompts, no cache hits:

| `-ub` | prompt tok/s | extra VRAM |
|---|---|---|
| 512 (llama.cpp default) | 378 | — |
| 1024 | 541 | +0.1 GiB |
| **2048** (used by the presets) | **734** | +0.4 GiB |
| 4096 | 881 | +0.9 GiB (left only 0.5 GiB free) |

### Memory (the 8 GB-class preset)

`--ctx-size 131072`, `-ub 2048`, `--n-cpu-moe 28`: about **7,330 MiB of 8,192 MiB** VRAM in use (including ~350 MiB used
by the desktop), ~860 MiB free. For comparison, 98k context with 26 layers uses ~7,500 MiB; 131k with 26 layers does not fit.

### The two presets, side by side

Both measured back to back with the freshly built llama.cpp from this installer (same machine, laptop already warm from the
build, CPU cap managed by the author's governor at roughly 48-58 %, so absolute speeds sit below the 70 % table above):

| | `mid-8gb-tuned` (default on 8 GB cards) | `mid` (automatic) |
|---|---|---|
| Layer placement | hand-set: `--n-cpu-moe 28 -ngl 99` | `-fit on -fitt 1024`: llama.cpp decides |
| Context window | 131,072 | 65,536 |
| `-ub` | 2048 | 1024 |
| VRAM in use / free after a 12k-token prompt | 7,226 / 966 MiB | 6,528 / 1,664 MiB |
| Generation | ≈ 27.5 tok/s | **≈ 30.8 tok/s** |
| Prompt processing (12k tokens) | **695 tok/s** | 560 tok/s |
| Model load (weights already in the page cache) | 18 s | 23 s |

Pick `mid-8gb-tuned` for agents and long documents (twice the context, faster prompt reading); ask for the model name `mid`
instead of `default` for chat-style use where generation speed matters most. The first request after a model loads is
slower (weights are still being paged in); the numbers above are from warmed-up runs (6-9 runs each).

### Quality of the quantization (vs the 22 GB `Q4_K_M` of the same model)

| Quant | file | wikitext perplexity | picks the same next token | Korean text | code |
|---|---|---|---|---|---|
| `Q4_K_M` (reference) | 22.1 GB | 5.82 | — | — | — |
| `UD-Q3_K_M` | 16.6 GB | 5.95 (+2.2 %) | 91.1 % | +1.3 % | +1.6 % |
| **`UD-IQ3_XXS`** (default for 16 GB RAM) | 13.2 GB | 6.19 (+6.3 %) | 87.8 % | +5.4 % | +3.4 % |

`UD-Q3_K_M` is closer to the reference but did not fit in 16 GB of RAM (heavy paging: 17-22 tok/s here).

### Tool calling

Agents live or die by well-formed tool calls, so this was checked two ways.

**A scripted test** (33 scenarios × 3 runs = 99 requests, `UD-IQ3_XXS`, the model's built-in chat template, thinking off,
temperature 0.3). A generic agent toolset of 7 tools (`read_file`, `write_file`, `list_directory`, `grep_search`,
`run_shell`, `web_search`, `send_message`); 18 scenarios in English and 15 in Korean. A run passes only if the right tool
is called with valid JSON whose argument *values and types* match (booleans, integers, enums), or if no tool is called
when none is needed:

| Category | Scenarios | Runs passed |
|---|---|---|
| Single call | 7 | 21 / 21 |
| Argument types (bool / int / enum / file content) | 5 | 15 / 15 |
| Choosing between similar tools | 2 | 6 / 6 |
| Parallel calls | 1 | 3 / 3 |
| Questions that need **no** tool (must not call one) | 4 | 12 / 12 |
| Multi-turn (use a tool result to decide the next call or the answer) | 7 | 21 / 21 |
| Hard cases (paths with spaces, quotes + Korean inside file content, "do not run anything", 3 parallel calls) | 7 | 21 / 21 |
| **Total** | **33** | **99 / 99** |

English 54 / 54, Korean 45 / 45; 0 invalid JSON, 0 tool calls leaked into the reply text, identical results on all three
repetitions, ~1.8 s per request. The scorer itself was checked against 12 deliberately wrong answers first.
*Limits:* one configuration (this quant + built-in template), synthetic scenarios, thinking mode not tested, and no
comparison against other quants or models was run — a clean sweep also means the suite may be easy.

**Real use with an agent (Hermes):** the tool calls behind the author's own requests — web searches, fetching and reading
pages, investigating the system with shell commands — worked without problems, and no malformed or unparsable tool call
was logged in the session.

The author's impression after daily use: tool calls worked reliably and felt clearly better than the previous Qwen
generation used before (subjective, not measured).

### In real agent use (Hermes)

The numbers above are the machine's best case. This is what the author's laptop did when driven by an agent framework
(Hermes), with the preset `mid-8gb-tuned` (131k window). The CPU cap was managed between 45 % and 60 % by a small
governor script (not part of this repo) that keeps the hot laptop from throttling, so these are lower than the 70 % table.

What the agent's per-turn readout showed (its "speed" is *output tokens ÷ the whole turn's wall-clock time*, prompt
reading included; "decode" is the server's pure generation speed for the same request):

| Turn | Output tokens | Input tokens | Prompt-cache hit | Speed in the readout | Pure decode | Situation |
|---|---|---|---|---|---|---|
| A | 2,278 | 114.7K (3 model calls, 36-40k each) | 104.5K (91.1 %) | 16.6 tok/s | 19.7 tok/s | normal multi-step turn |
| B | 2,533 | 33.8K | 0 (0 %) | 13.2 tok/s | 19.0 tok/s | first turn after the agent compacted its context, laptop heat-soaked |
| C | 3,091 | 37.2K | 35.8K (96.3 %) | 21.1 tok/s | 21.7 tok/s | laptop rested first, fans on auto |
| D | 1,281 | 36.1K | 33.5K (93 %) | ≈ 21 tok/s * | 23.4 tok/s | laptop rested first, manual cooler boost |

\* computed from the agent's log (1,281 tokens / 61 s); the readout itself was not captured for this turn.

Over 41 requests with 25k-98k tokens of context (this includes the agent's own background "review" passes):

| | Typical | Range |
|---|---|---|
| Generation speed, 30-50k context | **21 tok/s** | 19-23 |
| Generation speed, 50k+ context | 18.6 tok/s | 16.5-21 |
| Reading the prompt for a normal follow-up turn (≈ 2k new tokens, rest cached) | **6 s** | up to 11 s (p90) |
| Reading a whole 30k+ prompt with no cache | 48 s (≈ 620 tok/s) | 31-58 s |
| Share of time spent reading prompts vs generating | 19 % / 81 % | |

Generation slows as the context grows (≈ 21 tok/s at 40k, ≈ 18 at 60k+; the one 25k-context sample ran at 27), and a
cache miss costs about a minute for a 30k-token prompt. Prompt-cache hits were 91-98 % outside of the turn right after a compaction.

**Untested:** AMD/Intel GPUs, ARM, other NVIDIA cards, and any RAM size other than 16 GB. The automatic
choices there are heuristics. If something is slow, try a smaller `--quant`, a smaller `--ctx`, and watch temperatures.

## What has been verified

<img src="assets/expr-can-do.png" alt="" align="right" height="90">

On the author's laptop, running this installer itself (not hand-built binaries): clean build of the pinned upstream llama.cpp
commit with CUDA 12.4 (compute capability 7.5), llama-swap v250 installed with its SHA-256 checked, config generation, model
download with resume after an interruption and SHA-256 checked against Hugging Face, and then a real model load, generation,
JSON-mode reply, the 33-scenario tool-calling suite (99 / 99 again on the freshly built binary), and swapping between the two
presets through llama-swap. Hardware selection for other RAM/GPU sizes was checked with `--dry-run` simulations only.

**Not verified:** the `--systemd` service, `--install-deps`, `--backend vulkan`, other GPUs/CPUs/distributions, ARM.

## Options

```
--prefix DIR        install location (default ~/lapmoe)      --port N        llama-swap port (8080)
--backend NAME      cuda (default) | vulkan (experimental)      --ctx N         context size override
--profile NAME      auto | mid | small                          --jobs N        parallel compile jobs
--quant NAME        e.g. UD-IQ3_XXS                             --model-path F  use an existing GGUF
--llama-ref REF     llama.cpp commit/tag to build               --swap-version  llama-swap tag or "latest"
--no-build --no-swap --no-model   skip a step                   --systemd       install + start a systemd *user* service
--demo              preview the ASCII art, then exit            --plain         no art / colours
--install-deps      apt-get prerequisites (sudo)                --dry-run -y --force
```

Environment: `HF_TOKEN` (only for gated repos), and `LSM_TEST_RAM_MB`, `LSM_TEST_GPU`, `LSM_TEST_VRAM_MB`
to *simulate* other hardware together with `--dry-run` (checks the selection logic, nothing more).

## What it touches

Everything is inside `--prefix` (default `~/lapmoe`): `src/` (llama.cpp), `bin/llama-swap`, `models/`,
`downloads/`, `config.yaml`, `start.sh`, `test.sh`. With `--systemd` it also writes
`~/.config/systemd/user/lapmoe.service`. Downloads are verified: llama-swap against the SHA-256 GitHub
publishes for the release asset, the model against the SHA-256 Hugging Face publishes for the file.
The server listens on `127.0.0.1` only. Re-running is safe: finished steps are skipped (`--force` redoes them)
and an existing `config.yaml` is backed up first.

Uninstall: stop the service (`systemctl --user disable --now lapmoe`), then `rm -rf` the prefix.

## Troubleshooting

<img src="assets/expr-surprised.png" alt="" align="right" height="90">

* **`nvcc` not found** — install the CUDA toolkit and re-run. (`--backend vulkan` exists but is experimental and untested; there is no CPU-only mode.)
* **Out-of-memory when loading** — lower `--ctx` in `config.yaml`, raise `--n-cpu-moe` (moves more experts to RAM), or pick a smaller quant.
* **Very slow, disk busy** — the model does not fit in RAM; choose a smaller `--quant` (see the table).
* **Port in use** — `--port 8081`.
* A model switch takes a few seconds to minutes: llama-swap keeps one model resident and loads others on request.

## Credits and licenses

**LapMoE is released under the [MIT License](LICENSE)** — © 2026 Kyudong Kim. That covers the installer, this README and
the mascot artwork in `assets/`.

The software it installs keeps its own licenses:

llama.cpp (MIT), llama-swap (MIT, Benson Wong), Qwen3.6 weights (Apache-2.0, Qwen team) quantized by Unsloth.
This installer only automates their official release/download channels.
