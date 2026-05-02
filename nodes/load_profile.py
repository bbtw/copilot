from models import CustomerProfile
from state import RetirementPlanState


def load_customer_profile(state: RetirementPlanState) -> dict:
    profile = CustomerProfile(
        age=42,
        retirement_age=65,
        annual_income=120_000,
        annual_expenses=85_000,
        retirement_annual_expenses=75_000,
        retirement_years_to_plan=30,
        expected_retirement_income=30_000,
        balances={
            "401k": 180_000,
            "roth_401k": 0,
            "traditional_ira": 25_000,
            "roth_ira": 40_000,
            "hsa": 8_000,
            "taxable_brokerage": 15_000,
        },
        employer_match_rate=0.50,
        employer_match_cap=0.06,
        hdhp_enrolled=True,
        current_marginal_tax_rate=0.22,
        assumed_retirement_marginal_tax_rate=0.12,
        filing_status="single",
    )
    return {"customer_profile": profile}
