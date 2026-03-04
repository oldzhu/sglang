# CHANGE_0012_toolkit_tech_path_reference

## 1) Background & Motivation
- To prevent optimization drift, we want every optimization/customization stage to explicitly reference official SOAR guidance.
- User requested mandatory review of toolkit page section `技术路径指引` before optimization decisions.

## 2) SOAR Rule-Compliance Check
- Process/instruction update only.
- No runtime/model/kernel changes.

## 3) Plan Before Code Change
- Update `.github/copilot-instructions.md` rule-freshness section with explicit `技术路径指引` review requirement.

## 4) Actual Code Change
- Added one line requiring explicit review of the toolkit `技术路径指引` section before optimization/customization stages.

## 5) Validation
- Manual verification by reading `.github/copilot-instructions.md`.

## 6) Risks
- None (policy clarification only).

## 7) Rollback
1. Remove the added `技术路径指引` requirement line from `.github/copilot-instructions.md`.

## 8) Next-Step Suggestions
- Keep citing which technical-path items are selected/rejected in each future change proposal.
