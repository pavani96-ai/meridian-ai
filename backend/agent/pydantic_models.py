"""Structured specialist-agent outputs for CFO synthesis."""
from typing import Literal
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class RiskAssessment(BaseModel):
    """Risk agent response; absent credit data must not be invented."""

    model_config = ConfigDict(extra='forbid')

    sanctions_check: Literal['CLEARED', 'RED ALERT', 'INCONCLUSIVE']
    sanctions_reason: str = Field(min_length=1, description='Reason, specific lists, source URLs, and coverage limitations.')
    credit_score: int | None = Field(ge=0, le=100, description='Null when unavailable or screening stopped the assessment.')
    credit_risk_level: Literal['LOW', 'MODERATE', 'HIGH'] | None
    recommended_payment_terms: str = Field(min_length=1)
    supply_chain_risk_flag: Literal['NONE', 'SINGLE-SOURCE RISK', 'CRITICAL-PATH SUPPLIER', 'UNKNOWN']
    supply_chain_risk_details: str = Field(min_length=1)
    overall_risk_verdict: Literal['APPROVED', 'CONDITIONAL APPROVAL', 'REJECTED']
    risk_notes: str = Field(min_length=1)


class TaxAssessment(BaseModel):
    """Tax and Treasury output; monetary values serialize as decimal strings."""

    model_config = ConfigDict(extra='forbid')

    tax_route: str = Field(min_length=1, description='Origin to destination.')
    transaction_type: Literal['DOMESTIC', 'INTRA-EU B2B', 'EXTRA-EU IMPORT', 'OTHER', 'UNDETERMINED']
    effective_tax_rate: Decimal | None = Field(ge=0, le=1, description='Combined decimal rate; null if not established.')
    tax_liability: Decimal | None = Field(ge=0, decimal_places=2, description='EUR, two decimal places; null if unavailable.')
    total_landed_cost: Decimal | None = Field(ge=0, decimal_places=2, description='EUR, transaction value plus calculated taxes; disclose excluded costs.')
    fx_validation: str = Field(min_length=1, description='Exact FX SUCCESS, FX ALERT, or FX WARNING result, or NOT REQUIRED with reason.')
    treasury_verdict: Literal['APPROVED', 'HOLD FOR TREASURY AUDIT', 'MANUAL APPROVAL REQUIRED']
    tax_notes: str = Field(min_length=1, description='Sources, applicable dates, assumptions, missing details, and next actions.')


class ControlAssessment(BaseModel):
    """Financial Control response for CFO synthesis."""

    model_config = ConfigDict(extra='forbid')

    expense_amount: Decimal = Field(ge=0, decimal_places=2, description='EUR amount.')
    item_description: str = Field(min_length=1)
    classification: Literal['CapEx', 'OpEx'] | None
    ifrs_reference: str = Field(min_length=1)
    gl_account_range: str = Field(min_length=1)
    depreciation_schedule: str = Field(min_length=1)
    approval_threshold: str = Field(min_length=1)
    anomaly_flags: list[str]
    control_verdict: Literal['APPROVED', 'FLAGGED FOR CONTROLLING REVIEW', 'REJECTED']
