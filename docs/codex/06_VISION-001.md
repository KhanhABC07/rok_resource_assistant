# VISION-001 — Template registry and screen analyzer

Use together with `00_SHARED_CONTEXT.md`. Implement only this task.

## Codex Prompt

```text
Implement a semantic TemplateRegistry and ScreenAnalyzer. Templates must be referenced by keys such as `city.collect.food.ready`, not file paths. Store template pack version, language, resolution profile, ROI, threshold, scale range, optional mask, and scene constraints. Add image normalization, ROI matching, multi-scale matching, scene classification, and typed DetectionResult with confidence and coordinates. Capture evidence and matching metadata. Create a validator CLI/GUI that checks missing/invalid templates. Add positive and negative screenshot replay tests and calibrate thresholds from data rather than hard-coding one global value.
```
