import argparse
import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="tau2_customer_service",
        name="Tau2 Customer Service Reasoning",
        description=(
            "Solves tau2 customer-service and troubleshooting tasks with "
            "policy-aware tool selection, clarification, and shared-control dialogue."
        ),
        tags=["benchmark", "tau2", "customer-service", "tool-use", "policy"],
        examples=[
            "Ask a short clarification question before changing a plan or billing setting.",
            "Use the provided tool schema to choose one safe next action per turn.",
            "Explain what the user must do when the shared environment requires user-side action.",
        ],
    )

    agent_card = AgentCard(
        name="Tau2 Purple Agent",
        description=(
            "A tau2-bench purple agent for AgentBeats that follows domain policy, "
            "asks clarifying questions when needed, and emits one structured action per turn."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
