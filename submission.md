# Project: m.AI.leage

### Challenge
**Best Open-Source AI Project**

### GitHub Repository
[https://github.com/DhanviND360/m.AI.leage](https://github.com/DhanviND360/m.AI.leage)

---

## What It Does

**m.AI.leage** is a 100% local, voice-driven, autonomous AI software engineer and telemetry companion. It allows developers to dictate software goals hands-free, watches open-weight models plan and build code autonomously in their local repository, verifies acceptance criteria through native automated testing, and escalates gracefully to cloud AI (GitHub Copilot) only when local models hit overcapacity.

### Core Capabilities

1. **One-Command Autonomous Workspace (`mileage start`)**:
   - Launches a minimal, Claude-Code-style persistent workspace with a speedometer mascot companion.
   - Provides hands-free voice interaction: speak your objective, and the agent plans and builds without manual typing or subcommand wrangling.

2. **Structured Intent Planning (`Gemma`)**:
   - Converts conversational speech or text into strict, testable `ActionPlan` schemas with prioritized requirements (`R1`, `R2`, ...) and measurable acceptance criteria (`AC1`, `AC2`, ...).
   - Features token truncation defense and multi-stage JSON auto-repair.

3. **Intelligent Local Model Routing**:
   - Automatically inspects task complexity and routes execution to the best locally installed Ollama model (e.g., `qwen2.5-coder` for implementation, `gemma` for planning and evaluation).

4. **Autonomous Coding & Self-Repair Loop**:
   - Executes a closed-loop development cycle: **Inspect → Plan → Edit → Test → Repair**.
   - Guarded tools: Sandboxed read, write, search, and native execution with strict directory permissions.
   - Automatically detects build/test failures and repairs code autonomously. Halts on repeated loop stagnation.

5. **Local Verification Evaluator**:
   - Evaluates implementation against the generated `ActionPlan` acceptance criteria with test execution evidence.
   - Never claims success without proof; returns `PASS`, `PARTIAL`, or `FAIL` verdicts.

6. **Overcapacity Escalation to GitHub Copilot**:
   - When local models fail repeatedly or exceed local capacity thresholds, m.AI.leage automatically generates a rich context handoff prompt (`.mileage_copilot_prompt.md`) and opens the project in VS Code with the Copilot workflow ready.

7. **Real-time Mission Control Dashboard (`mileage serve`)**:
   - A dark-mode, responsive React + Vite + TypeScript dashboard streaming live agent states via Server-Sent Events (SSE).
   - Shows live pipeline transitions (`Planning` → `Building` → `Testing` → `Evaluating` → `Complete/Escalating`), tokens saved, estimated cloud cost avoided, and build history.
   - Accessible remotely on mobile devices over the local Wi-Fi network (`http://<local-ip>:3000`).

---

## How It Uses Open-Source / Open-Weight AI

m.AI.leage operates **100% offline and on-device** with zero reliance on closed, paid cloud APIs:

- **Ollama Engine**: Core local inference runtime serving open-weight models locally via standard HTTP APIs.
- **Google Gemma (Open-Weight)**: Orchestrates natural language understanding, multimodal image analysis, and strict JSON ActionPlan planning.
- **Qwen 2.5 Coder (Open-Weight)**: Drives the autonomous coding loop, handling complex multi-file codebase edits, syntax parsing, and automated test repair.
- **OpenAI Whisper (Open-Source Weights)**: Powers on-device speech-to-text transcription with offline models (`tiny.en` / `base.en`).
- **Silero VAD / webrtcvad (Open-Source)**: Enables zero-latency voice activity detection with real-time speech boundary segmentation.
- **Local Speech Synthesis (pyttsx3 / piper)**: Provides situational spoken status updates with strict acoustic isolation to prevent mic feedback loops.

---

## How to Run

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- [Ollama](https://ollama.ai) installed and running locally
- Pull the recommended models:
  ```bash
  ollama pull gemma:2b       # or gemma4:e2b
  ollama pull qwen2.5-coder  # or qwen2.5-coder:1.5b / 7b
  ```
- Node.js 18+ (for building the optional dashboard)

### 1. Installation
Clone the repository and install in editable mode:
```bash
git clone https://github.com/DhanviND360/m.AI.leage.git
cd m.AI.leage
pip install -e .
```

### 2. Verify Your Environment
Run the diagnostic check:
```bash
mileage doctor
```

### 3. Launch the Interactive One-Command Experience
```bash
mileage start
```
*Options:*
- `mileage start --no-voice` : Run in text-only keyboard mode.
- `mileage start --model qwen2.5-coder` : Specify your preferred local coding model.

### 4. Launch the Web Command Center (Optional / Dual-Screen Demo)
In a separate terminal or background process:
```bash
mileage serve
```
Open `http://localhost:3000` on your desktop, or open `http://<your-local-ip>:3000` from your phone on the same Wi-Fi network to monitor builds in real-time.

### 5. Running Tests
Run the comprehensive test suite (181 tests):
```bash
pytest -q
```

---

## Demo Walkthrough

1. **Launch**: Run `mileage start`. The terminal displays the friendly speedometer mascot logo and system status pills (Model, Voice, Workspace, Local Status).
2. **Listen**: m.AI.leage announces *"m.AI.leage is online. Ready for your goal."* and activates hands-free listening.
3. **Prompt**: Say *"Add a fibonacci utility function with unit tests"*.
4. **Planning**: Gemma analyzes the prompt and outlines the `ActionPlan` with requirements and acceptance criteria.
5. **Routing & Building**: Model router assigns the task to `qwen2.5-coder`. The coding agent inspects existing files, creates the implementation, and writes tests.
6. **Testing**: Native project tester runs tests (`pytest`). If a failure is found, the agent triggers an autonomous repair cycle.
7. **Evaluating**: Evaluator compares code and test evidence against the criteria.
8. **Completion**: Acceptance criteria are verified, telemetry records saved tokens and estimated cloud cost avoided, and the agent confirms completion.
9. **Escalation Demo**: If an impossible or oversized task is requested, overcapacity is detected; m.AI.leage automatically generates a Copilot prompt and opens VS Code for cloud pair programming.
10. **Live Dashboard**: The React command center shows the pipeline steps updating in real time, with instant mobile responsiveness.
