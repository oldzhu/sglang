# Calibration Candidate Results Template

Use this template to record each candidate on fcloud with the same metrics and decision rules.

## Candidate

- Name:
- Output file:
- Indices:
- Intent:

## Run 1

- ori_accuracy:
- total_duration:
- mcq|len_0_4k:
- qa|len_4k_32k:
- qa|len_32k_128k:
- niah overall:
- Notes:

## Run 2

- ori_accuracy:
- total_duration:
- mcq|len_0_4k:
- qa|len_4k_32k:
- qa|len_32k_128k:
- niah overall:
- Notes:

## Decision

- Keep testing / Reject / Submission candidate:
- Reason:
- Follow-up:

## Ranking Rule

Prioritize candidates that satisfy all of the following:

1. ori_accuracy >= 78.5
2. mcq|len_0_4k >= 63.33
3. qa|len_4k_32k >= 60.00
4. No severe collapse in niah
5. No clear local duration regression