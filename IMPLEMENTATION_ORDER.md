# Implementation Order

## Rationale
Build the visual pipeline from input to output first, tackling the hardest CV problems early to surface risk, then layer in the AIS data integration and tie everything together with structured output.

## Order
1. F1 — Video ingestion: Foundation everything else reads from; need a stable frame source before any CV work makes sense
2. F2 — Waterline detection: The core CV problem; validates that the approach is feasible before building dependent features
3. F3 — Plimsoll line detection: Depends on a working visual pipeline; harder than waterline detection, worth tackling early to surface risk
4. F4 — Draft depth estimation: Pure geometry/math layer that sits on top of F2 + F3; low implementation risk, high demo value
5. F5 — AIS data integration: Independent data engineering track; can be developed in parallel with F2–F4 but needs to be ready before F6
6. F6 — Structured output: Ties everything together; last because it requires all upstream signals to be working
