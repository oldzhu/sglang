# CHANGE_0008_instruction_priority

## 1) Background & Motivation
- We need stable behavior across long optimization conversations with multiple possible instruction sources.
- Goal: define explicit instruction precedence to avoid ambiguity.

## 2) SOAR Rule-Compliance Check
- Documentation/process update only.
- No impact on model runtime, correctness, or benchmark logic.

## 3) Plan Before Code Change
- Update `.github/copilot-instructions.md` with an instruction-priority section.
- Add bilingual change documentation.

## 4) Actual Code Change
- Updated `.github/copilot-instructions.md` with:
  - ordered instruction precedence (system/developer -> project instructions -> current user request -> other repo docs)
  - conflict handling note (follow higher priority and explain briefly)

## 5) Validation
- Manual verification by reading the updated instruction file.

## 6) Results Summary
| Item | Status |
|---|---|
| Priority policy added | Done |
| Future conflict handling guidance | Done |

## 7) Risks
- None (process clarification only).

## 8) Rollback
1. Remove the `Instruction priority order` section from `.github/copilot-instructions.md`.

## 9) Next-Step Suggestions
- Optionally add one short "Examples of conflicts" subsection in a future change.
