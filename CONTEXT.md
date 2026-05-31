# Financial Planning Co-Pilot

A LangGraph-based agent that assists licensed US financial advisors by synthesizing **Decision-Support Briefs** from a **Client Profile**. Internal-facing only at v1: outputs are consumed by the advisor, not delivered to the client.

## Language

**Advisor**:
The licensed US financial advisor who operates the agent and is the regulated entity-of-record for any downstream client communication.
_Avoid_: Planner, user, broker (these have specific industry meanings that don't fit here)

**Client**:
The household being analyzed. Always the subject of a brief, never an operator of the system.
_Avoid_: Customer, user, account holder

**Client Profile**:
The structured Pydantic-typed input describing the **Client**'s financial situation. The single input boundary at v1 — no documents, no aggregator feeds, no CRM read-through.
_Avoid_: Intake, household data, financial picture

**Decision-Support Brief** (or **Brief**):
The v1 output artifact. Surfaces 2–3 strategic options for the **Advisor** to evaluate, plus situation triage and flagged issues. Internal work product — never a client deliverable.
_Avoid_: Plan, report, recommendation memo (these imply client-facing or directive output)

**Plan Synthesis**:
The act of producing a **Brief** from a **Client Profile**. The v1 task scope. Distinct from **Plan Generation** (full client-facing plan, deferred to later versions).
_Avoid_: Plan creation, planning, advice generation

**Co-Pilot**:
The agent's positioning — the **Advisor** retains decision authority and liability. The agent surfaces options; it does not recommend.
_Avoid_: Autonomous agent, robo-advisor, assistant

**Wellness Model**:
The firm's pre-existing weighted multi-domain financial wellness scoring system. Produces per-domain **Wellness Scores** for a **Client Profile**. The agent consumes its output; it does not compute scores or override its rankings.
_Avoid_: Scoring engine, financial health model, planning model

**Wellness Score**:
A per-domain numeric output of the **Wellness Model** for a given **Client Profile**. Drives deterministic ranking in the **Brief**.
_Avoid_: Health score, planning score, rating

**Wellness Domain**:
One of the firm-defined areas covered by the **Wellness Model** (e.g., retirement, tax, insurance, estate, cash flow). The exact list is owned by the firm, not the agent.
_Avoid_: Category, area, pillar (these are used inconsistently across the industry)

**Calculator**:
A pre-existing deterministic computation owned by the firm (e.g., retirement Monte Carlo, tax projection). Produces structured outputs that the agent reads but does not modify.
_Avoid_: Engine, model (overloaded), tool (overloaded with LangChain tool-calling)

## Relationships

- An **Advisor** runs **Plan Synthesis** on one **Client Profile** at a time
- **Plan Synthesis** produces exactly one **Decision-Support Brief** per run
- A **Decision-Support Brief** belongs to one **Client** and is reviewed by one **Advisor**
- A **Client Profile** describes one **Client** (single household; joint clients modeled as one profile)
- The **Wellness Model** reads a **Client Profile** and emits one **Wellness Score** per **Wellness Domain**
- Each **Calculator** reads (parts of) a **Client Profile** and emits structured numerical output
- **Plan Synthesis** consumes: a **Client Profile**, a set of **Wellness Scores**, and a set of **Calculator** outputs

## Example dialogue

> **Dev:** "When the **Advisor** triggers **Plan Synthesis**, can the **Brief** include a specific Roth conversion amount?"
> **Domain expert:** "It can model the tradeoff and show 2–3 conversion sizes as options, but it doesn't pick one — that's the **Advisor**'s call. If it picks, we're no longer a **Co-Pilot**, we're a robo-advisor and the regulatory profile changes."

## Flagged ambiguities

- "Plan" was used to refer to both the v1 output and a future client-facing artifact — resolved: v1 output is a **Decision-Support Brief**, the term "Plan" is reserved for the client-facing artifact built in later versions.
- "User" was ambiguous between **Advisor** and **Client** — resolved: only the **Advisor** uses the system; the **Client** is a subject, not a user.

## Open questions (next session pickup)

These shape what the **Wellness Model** emits and therefore what the agent has to work with. They should be the first items resolved when work resumes:

- **Wellness Domains** — what is the firm's actual taxonomy? (Retirement, tax, insurance, estate, cash flow are the working straw man.)
- **Wellness Score scale and semantics** — what range, and what does "good" vs "concerning" mean numerically?
- **Ranking output** — does the **Wellness Model** emit a pre-ranked list, or only per-domain scores (leaving ranking as a thin deterministic node downstream)?
- **Wellness Score drivers** — does each score come with structured "why this score" data (e.g., "retirement = 6/10 because savings rate is 8% vs target 15%"), or only the bare number? Load-bearing for the agent's narrative grounding (decision-spectrum row #5).

Live status of these is mirrored in `docs/agent-decision-spectrum.html`.
