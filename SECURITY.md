# Security Policy

## Crisis Detection

Clanker includes a crisis detection system that identifies text indicating potential self-harm or suicidal ideation. This system is designed as a screening tool, not a clinical instrument.

**Current performance**: 70.6% recall on crisis sentences, 0.0% false positive rate on safe text (including dark humor and metaphor).

**This system is not a substitute for professional mental health assessment.** It is a signal layer that can flag text for human review. Do not use crisis detection scores as the sole basis for clinical decisions.

## Reporting Vulnerabilities

If you discover a security vulnerability, especially one that could cause:
- False negatives on crisis-relevant text (failing to flag genuine risk)
- False positives that could trigger unnecessary interventions
- Data leakage from the engine's internal state

Please report it privately to jerrymares@gmail.com rather than opening a public issue.

## Scope

The engine processes text locally. It does not:
- Send data to external services
- Store conversation history (unless the caller implements it)
- Require network access for scoring

## Supported Versions

| Version | Supported |
|---------|-----------|
| V8.x    | Yes       |
| V9.x    | Experimental |
| V2.x    | No        |
