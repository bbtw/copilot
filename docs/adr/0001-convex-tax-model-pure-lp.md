# Convex tax model so the optimization stays a pure LP

The optimizer's value comes from tax-aware contribution and withdrawal decisions, which requires modeling progressive federal brackets — a flat rate collapses the optimum to a corner solution. US bracket schedules are convex piecewise-linear (marginal rates only increase), and since tax always hurts the objective, brackets can be modeled in a pure LP with no binary variables. We commit to keeping the tax model convex: filing status + standard deduction + progressive federal brackets on ordinary income, flat capital-gains rate on the taxable account, no state tax.

## Consequences

- Tax features that break convexity are out of scope until this constraint is consciously dropped: Social Security benefit taxation phase-in, capital-gains bracket stacking, IRMAA cliffs, AMT.
- Adding any of those later means migrating from LP to MILP (piecewise with binaries) — a formulation change, not just a feature addition. Revisit this ADR at that point.
- The model works in real (today's) dollars: bracket thresholds and contribution limits are held constant across years (the IRS indexes them to inflation, so this is approximately correct), and return inputs are real returns. No inflation parameter exists anywhere.
