# SOAR 2026 Collaboration Instructions (MiniCPM-SALA)

This repository is used for SOAR 2026 optimization work on MiniCPM-SALA.

## Mandatory workflow for every optimization

1. **Proposal first, no direct code changes**
   - Before any source change, provide a detailed optimization proposal including:
     - objective and expected gain
     - rule-compliance check (SOAR constraints)
     - risk to accuracy/stability
     - exact files/functions to change
     - test and benchmark commands
   - Wait for explicit user approval before editing code.

2. **One improving feature at a time**
   - Each iteration should deliver one complete optimization feature (can include related updates across multiple files).
   - Keep scope cohesive: all edits in the iteration must serve the same optimization objective.
   - Avoid mixing unrelated goals in one iteration.

3. **Bilingual documentation per feature (required)**
   - For each approved change, create two documents:
     - English: `docs/soar_2026_changes/CHANGE_XXXX_<short_title>.en.md`
     - Chinese: `docs/soar_2026_changes/CHANGE_XXXX_<short_title>.zh.md`
   - One document pair corresponds to exactly one optimization feature iteration.

4. **Documentation must include**
   - Background and motivation
   - Rule-compliance statement (what is allowed and why)
   - Detailed implementation plan (before change)
   - Actual code changes (after change)
   - Validation commands (correctness + speed)
   - Result summary table (baseline vs new)
   - Rollback instructions
   - Next-step suggestions

5. **Execution model with user’s fcloud instance**
   - Agent proposes and documents changes in this workspace.
   - User applies/runs commands in fcloud instance and reports metrics/errors.
   - Agent iterates based on returned results.

## Competition guardrails (must enforce)

- Keep model correctness above SOAR threshold (accuracy coefficient must not be zero).
- Do not rely on forbidden tricks (e.g., privately re-enabling prefix cache during official eval).
- Respect fixed concurrency evaluation settings.
- Keep submission package constraints in mind (including size/time limits and reproducibility).

## Rule freshness requirement (must enforce)

- Whenever optimization/compliance decisions depend on competition rules, re-check the latest official pages first:
   - https://soar.openbmb.cn/competition
   - https://soar.openbmb.cn/toolkit
- Before starting optimization/customization stages, explicitly review and refer to the `技术路径指引` section on the toolkit page to align with officially suggested technical directions.
- If any conflict appears between prior assumptions and latest official text, follow the official pages and explicitly call out the update.

## Submission preparation requirement (must enforce)

- When the task involves preparing competition submission artifacts (e.g., `prepare_env.sh`, `prepare_model.sh`, `preprocess_model.py`, packaging layout), explicitly refer to and follow the latest `提交说明` section on:
   - https://soar.openbmb.cn/toolkit
- For submission-related customization, align scripts with official execution model and interfaces (including `prepare_env.sh` and `prepare_model.sh --input/--output` contract), and state any assumptions if local/fcloud environment differs from official runtime.

## Prioritization strategy

1. Low-risk, high-impact runtime optimizations first.
2. Then quantization/preprocessing path that preserves correctness.
3. Finally higher-risk algorithmic changes (e.g., speculative decoding variants).

## Communication style for this project

- Always provide:
  - what to change,
  - why it should help,
  - how to verify,
  - and what success/failure looks like.
- Ask for approval before each code modification.
- Keep English + Chinese docs synchronized.

## Instruction priority order

When multiple instructions exist, follow this priority (high -> low):

1. System/developer policy constraints from the runtime.
2. This file (`.github/copilot-instructions.md`) for repository-specific rules.
3. User's current request in the active conversation.
4. Other repository docs (`README`, `AGENTS.md`, scripts, comments) as supporting context.

If conflicts happen, follow the higher-priority source and explain the conflict briefly.
