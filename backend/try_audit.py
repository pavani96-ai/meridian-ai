"""Run a live audit from backend/: uv run python try_audit.py approved|hold|rejected.

Scenario names are illustrative, not guaranteed outcomes; live evidence determines verdicts.
"""
import argparse
import json
import time

from agent.agents import ProcurementSupervisor


REQUESTS = {
    "approved": """
    Purchase Request - Aldermoor Industries Global Procurement:
    - Vendor: Mitsubishi Electric (Japan)
    - Item: 20 spare servo motors for the Aldermoor filling line
    - Total Cost: 38,000 EUR
    - Origin: JP (Japan)
    - Destination: DE (Aldermoor Industries, Hamburg, Germany)
    - FX Rate quoted: 1 EUR = 176.5 JPY
    """,
    "hold": """
    Purchase Request - Aldermoor Industries Global Procurement:
    - Vendor: Takumi Controls Europe B.V. (Netherlands)
    - Item: Industrial PLC Controllers + HMI Panels (Qty: 200) for the Aldermoor filling line
    - Total Cost: 480,000 EUR
    - Origin: JP (Japan)
    - Destination: DE (Aldermoor Industries, Hamburg, Germany)
    - FX Rate quoted: 1 EUR = 163.5 JPY
    """,
    "rejected": """
    Purchase Request - Aldermoor Industries Global Procurement:
    - Vendor: JSC Rosoboronexport (Russia)
    - Item: Precision steel components (Qty: 40 tonnes)
    - Total Cost: 150,000 EUR
    - Origin: RU (Russia)
    - Destination: DE (Aldermoor Industries, Hamburg, Germany)
    - FX Rate quoted: not applicable, invoiced in EUR
    """,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario', choices=REQUESTS, nargs='?', default='hold')
    scenario = parser.parse_args().scenario
    started = time.perf_counter()
    audit = ProcurementSupervisor().run_audit(REQUESTS[scenario])

    print(f"\n=== {scenario}: audit finished in {time.perf_counter() - started:.0f} s")
    for name, field in (
        ('risk_result', 'overall_risk_verdict'),
        ('tax_result', 'treasury_verdict'),
        ('control_result', 'control_verdict'),
    ):
        report = json.loads(audit[name])
        print(f"{name:15}", report.get(field, '(no verdict)'))
    print('\n' + audit['cfo_memo'])


if __name__ == '__main__':
    main()
