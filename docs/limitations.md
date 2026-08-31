# Limitations

This is an independent synthetic study. Read these before interpreting results.

1. **Entirely synthetic data** — no production bank or client data.
2. **Stylized fraud-like decision problem** — fraud is a simulation domain, not a product claim.
3. **Not a production banking system.**
4. **No realized financial impact** — costs are simulated under a stylized objective.
5. **Simulated economic loss exposure** — `v` is severity conditional on fraud in the simulator, not a specific organization's true loss process.
6. **Simplified review** — binary selection under a capacity ceiling with a constant review cost.
7. **Deterministic fractional review effectiveness** — `e` is not a Bernoulli perfect-prevention probability. Real review systems are stochastic and operationally more complicated. The simplified form is useful here because it isolates the policy mechanics, not because it is intended as a faithful model of investigation outcomes.
8. **No adversarial adaptation.**
9. **No delayed-label dynamics.**
10. **No queueing dynamics.**
11. **No fairness analysis.**
12. **No causal inference.**
13. **Stylized cost function** — cost-aware policies are evaluated against an objective closely related to the quantities used in their ranking functions. That alignment matters when reading oracle results.
14. **Oracle policies are reference bounds**, not deployable policies.
15. **Results are not universal policy recommendations.** They are conditional on the pre-specified grids and assumptions.

The 1% near-equivalence band is project-defined, not an industry standard.
