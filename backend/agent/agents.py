from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from pydantic import BaseModel

from rag.llm import extract_text, get_llm
from logger import GLOBAL_LOGGER as log

from agent.tools import (
    web_search,
    check_sanctions_list,
    get_vendor_credit_score,
    calculate_cross_border_tax,
    validate_fx_hedge,
    categorize_expense,
)
from .prompts import (
    RISK_AGENT_PROMPT,
    TAX_AGENT_PROMPT,
    CONTROL_AGENT_PROMPT,
    SYNTHESIS_PROMPT_TEMPLATE,
)
from .pydantic_models import RiskAssessment, TaxAssessment, ControlAssessment
from .middleware import build_agent_middleware


def create_specialized_agent(tools, system_prompt: str, response_schema: type[BaseModel], *, model=None):
    """Build a specialist with tools, middleware, and provider-native structured output."""
    return create_agent(
        model=model if model is not None else get_llm(),
        tools=tools,
        system_prompt=system_prompt,
        middleware=build_agent_middleware(),
        response_format=ProviderStrategy(schema=response_schema),
    )


class ProcurementSupervisor:
    """Own the shared LLM and the three procurement specialist agents."""

    def __init__(self):
        self.llm = get_llm()
        self.risk_agent = create_specialized_agent(
            tools=[check_sanctions_list, get_vendor_credit_score],
            system_prompt=RISK_AGENT_PROMPT,
            response_schema=RiskAssessment,
            model=self.llm,
        )
        self.tax_agent = create_specialized_agent(
            tools=[calculate_cross_border_tax, validate_fx_hedge],
            system_prompt=TAX_AGENT_PROMPT,
            response_schema=TaxAssessment,
            model=self.llm,
        )
        self.control_agent = create_specialized_agent(
            tools=[categorize_expense],
            system_prompt=CONTROL_AGENT_PROMPT,
            response_schema=ControlAssessment,
            model=self.llm,
        )

    @staticmethod
    def _invoke_agent(agent, request: str) -> str:
        """Invoke a specialist and serialize its validated response for CFO synthesis."""
        if not request.strip():
            raise ValueError("A procurement request is required.")
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        response = result.get("structured_response")
        if not isinstance(response, BaseModel):
            raise ValueError("Agent did not return a validated structured response.")
        return response.model_dump_json()

    def run_audit(self, request: str) -> dict[str, str]:
        """Run the three specialist reviews, then synthesise the CFO memorandum."""
        log.info("Starting audit phase", phase="1 - Risk & Compliance")
        risk_result = self._invoke_agent(self.risk_agent, request)
        log.info("Audit phase complete", phase="1 - Risk & Compliance")

        log.info("Starting audit phase", phase="2 - Tax & Treasury")
        tax_result = self._invoke_agent(self.tax_agent, request)
        log.info("Audit phase complete", phase="2 - Tax & Treasury")

        log.info("Starting audit phase", phase="3 - Financial Control")
        control_result = self._invoke_agent(self.control_agent, request)
        log.info("Audit phase complete", phase="3 - Financial Control")

        synthesis_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            risk_result=risk_result,
            tax_result=tax_result,
            control_result=control_result,
        )
        log.info("Starting audit phase", phase="4 - CFO Synthesis")
        cfo_memo = extract_text(self.llm.invoke(synthesis_prompt).content)
        if not isinstance(cfo_memo, str) or not cfo_memo.strip():
            raise ValueError("CFO synthesis returned an empty memorandum.")
        log.info("Audit phase complete", phase="4 - CFO Synthesis")

        return {
            "risk_result": risk_result,
            "tax_result": tax_result,
            "control_result": control_result,
            "cfo_memo": cfo_memo,
        }

    def stream_audit(self, request: str):
        """Yield progress, validated specialist reports, memo deltas and final output.

        Raw tool evidence and intermediate model messages stay on the server.
        """
        if not request.strip():
            raise ValueError("A procurement request is required.")
        results = {}
        for key, title, agent in (
            ("risk_result", "Risk & Compliance", self.risk_agent),
            ("tax_result", "Tax & Treasury", self.tax_agent),
            ("control_result", "Financial Control", self.control_agent),
        ):
            yield {"type": "progress", "message": f"Reviewing {title}…"}
            response = None
            for update in agent.stream(
                {"messages": [{"role": "user", "content": request}]},
                stream_mode="updates",
            ):
                for node, data in update.items():
                    if not isinstance(data, dict):
                        continue
                    if isinstance(data.get("structured_response"), BaseModel):
                        response = data["structured_response"]
                    if node == "tools":
                        yield {"type": "progress", "message": f"{title}: tool checks completed."}
            if response is None:
                raise ValueError("Agent did not return a validated structured response.")
            results[key] = response.model_dump_json()
            yield {"type": "specialist", "key": key, "report": results[key]}

        yield {"type": "progress", "message": "Writing CFO memorandum…"}
        memo = ""
        for chunk in self.llm.stream(SYNTHESIS_PROMPT_TEMPLATE.format(**results)):
            text = extract_text(chunk.content)
            if text:
                memo += text
                yield {"type": "memo_delta", "text": text}
        if not memo.strip():
            raise ValueError("CFO synthesis returned an empty memorandum.")
        yield {"type": "complete", "audit": {**results, "cfo_memo": memo}}


if __name__ == "__main__":
    sample_request = """
    Purchase Request — Aldermoor Industries Global Procurement:
    - Vendor: Takumi Controls Europe B.V. (Netherlands)
    - Item: Industrial PLC Controllers + HMI Panels (Qty: 200) for the Aldermoor filling line
    - Total Cost: 480,000 EUR
    - Origin: JP (Japan)
    - Destination: DE (Aldermoor Industries, Hamburg, Germany)
    - FX Rate quoted: 1 EUR = 163.5 JPY
    """
    memo = ProcurementSupervisor().run_audit(sample_request)
    print(memo["cfo_memo"])
