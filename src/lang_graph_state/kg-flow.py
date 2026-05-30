from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ==========================================
# 1. Domain Models & Schema Definitions
# ==========================================

@dataclass
class TaskNode:
    id: str
    required_data: List[str]
    tool_endpoint: str


@dataclass
class ClientSession:
    client_id: str
    context: Dict[str, Any] = field(default_factory=dict)


# Mock Enterprise Databases
MOCK_CRM_DATABASE = {
    "client_123": {"current_ages": [62, 60], "retirement_age": 65}
}
MOCK_PORTFOLIO_DATABASE = {
    "client_123": {"tax_lots": [{"ticker": "VOO", "basis": 150, "value": 450}]}
}


# ==========================================
# 2. Knowledge Graph (Deterministic Rules)
# ==========================================

class FinancialPlanningKG:
    def __init__(self):
        # Explicitly define tasks, their prerequisites, and the correct tool
        self.tasks: Dict[str, TaskNode] = {
            "optimize_withdrawal_sequence": TaskNode(
                id="optimize_withdrawal_sequence",
                required_data=["current_ages", "tax_lots"],
                tool_endpoint="gurobi_optimization_solver"
            )
        }

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        return self.tasks.get(task_id)


# ==========================================
# 3. Execution Tools (Deterministic Engines)
# ==========================================

class ExecutionEngines:
    @staticmethod
    def crm_api_gateway(client_id: str) -> Dict[str, Any]:
        print(f"[Tool] Fetching demographic details from CRM for {client_id}...")
        return MOCK_CRM_DATABASE.get(client_id, {})

    @staticmethod
    def portfolio_api_gateway(client_id: str) -> Dict[str, Any]:
        print(f"[Tool] Fetching active account tax lots for {client_id}...")
        return MOCK_PORTFOLIO_DATABASE.get(client_id, {})

    @staticmethod
    def gurobi_optimization_solver(payload: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[Deterministic Engine] Executing Linear Programming Solver...")
        print(f"[Engine Input Payload]: {payload}")

        # Simulate deterministic math optimization output
        return {
            "status": "Optimal Solution Found",
            "recommended_sequence": ["Pre-tax IRA", "Taxable Brokerage", "Roth IRA"],
            "tax_savings_estimate_pct": 14.2
        }


# ==========================================
# 4. Agentic Orchestrator (Reasoning Router)
# ==========================================

class Orchestrator:
    def __init__(self, kg: FinancialPlanningKG):
        self.kg = kg

    def execute_planning_step(self, session: ClientSession, intent_task_id: str) -> str:
        # Step 1: Locate the node in the Knowledge Graph
        task = self.kg.get_task(intent_task_id)
        if not task:
            return f"Error: Task '{intent_task_id}' is not an approved workflow."

        print(f"\n[Orchestrator] Initiating: {task.id}")

        # Step 2: Validate data prerequisites defined by the graph edges
        for data_key in task.required_data:
            if data_key not in session.context:
                print(f"[Orchestrator] Missing required data: '{data_key}'. Resolving...")

                # Step 3: Use the designated tool to fill missing state
                if data_key == "current_ages":
                    crm_data = ExecutionEngines.crm_api_gateway(session.client_id)
                    session.context.update(crm_data)
                elif data_key == "tax_lots":
                    portfolio_data = ExecutionEngines.portfolio_api_gateway(session.client_id)
                    session.context.update(portfolio_data)

        # Step 4: Verify all prerequisites are now satisfied
        missing = [d for d in task.required_data if d not in session.context]
        if missing:
            return f"Error: Unable to satisfy data dependencies: {missing}"

        # Step 5: Format payload and isolate the execution to the math engine
        optimization_payload = {
            "constraints": {
                "ages": session.context["current_ages"],
                "target_retirement": session.context.get("retirement_age", 65)
            },
            "variables": {
                "assets": session.context["tax_lots"]
            }
        }

        # Route execution to the function specified by the graph node
        tool_to_call = getattr(ExecutionEngines, task.tool_endpoint)
        result = tool_to_call(optimization_payload)

        # Step 6: Translate deterministic output back to the UI context
        return (
            f"\n[Assistant Response to Advisor]:\n"
            f"Based on the mathematical optimization, the recommended withdrawal order is "
            f"{' -> '.join(result['recommended_sequence'])}. This strategy minimizes the multi-year "
            f"tax hit, improving net lifetime cash flow by an estimated {result['tax_savings_estimate_pct']}%."
        )


# ==========================================
# 5. Runtime Demonstration
# ==========================================

if __name__ == "__main__":
    # Initialize the core infrastructure
    knowledge_graph = FinancialPlanningKG()
    agent_orchestrator = Orchestrator(knowledge_graph)

    # Simulate a fresh session with no cached data
    active_session = ClientSession(client_id="client_123")

    # Trigger the workflow execution
    final_output = agent_orchestrator.execute_planning_step(
        session=active_session,
        intent_task_id="optimize_withdrawal_sequence"
    )
    print(final_output)