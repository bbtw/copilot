# Low-confidence plan revision workflow

In v2, a low `ConfidenceBand` means the plan is unlikely to fund `RetirementAnnualExpenses` for `RetirementYearsToPlan` years after retirement age, not merely unlikely to hit the LP's projected wealth. The graph will respond by running one customer-approved `PlanRevisionIntake`, assembling a complete `RevisedPlanScenario`, rerunning optimization and Monte Carlo once, and presenting the revised result as the primary plan with the baseline kept for comparison.

This keeps the LP as an after-tax accumulation optimizer while the Monte Carlo processor owns retirement-readiness scoring. We are deliberately not creating an unbounded advisory loop, not rerunning after each individual intake answer, and not letting the graph unilaterally change customer assumptions without approval.
