"""Validated API request and response schemas."""

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class InputType(str, Enum):
    text = "text"
    url = "url"


class Verdict(str, Enum):
    real = "REAL"
    fake = "FAKE"


class RiskLevel(str, Enum):
    low = "LOW"
    medium = "MEDIUM"
    high = "HIGH"


class CredibilityLabel(str, Enum):
    reliable = "RELIABLE"
    mixed = "MIXED"
    unreliable = "UNRELIABLE"
    unknown = "UNKNOWN"


class FeatureImpact(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class AnalyzeRequest(APIModel):
    text: str | None = Field(default=None, min_length=1, max_length=100_000)
    url: HttpUrl | None = None

    @model_validator(mode="after")
    def require_exactly_one_input(self) -> "AnalyzeRequest":
        if (self.text is None) == (self.url is None):
            raise ValueError("Provide exactly one of 'text' or 'url'.")
        return self


class FeatureContribution(APIModel):
    feature: str = Field(min_length=1, max_length=200)
    value: float
    impact: FeatureImpact


class ModelContribution(APIModel):
    prediction: Verdict
    confidence: float = Field(ge=0, le=1)


class SourceCredibility(APIModel):
    domain: str | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    label: CredibilityLabel = CredibilityLabel.unknown
    known_flags: list[str] = Field(default_factory=list)


class AnalysisExplanation(APIModel):
    top_features: list[FeatureContribution] = Field(default_factory=list)
    linguistic_flags: list[str] = Field(default_factory=list)
    model_contributions: dict[str, ModelContribution] = Field(default_factory=dict)


class AnalyzeResponse(APIModel):
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input_type: InputType
    original_input: str
    extracted_text: str | None = None
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    source_credibility: SourceCredibility = Field(default_factory=SourceCredibility)
    explanation: AnalysisExplanation = Field(default_factory=AnalysisExplanation)


class BatchAnalysisItem(APIModel):
    row: int = Field(ge=1)
    result: AnalyzeResponse | None = None
    error: str | None = None

    @model_validator(mode="after")
    def require_result_or_error(self) -> "BatchAnalysisItem":
        if (self.result is None) == (self.error is None):
            raise ValueError("Provide exactly one of 'result' or 'error'.")
        return self


class BatchAnalyzeResponse(APIModel):
    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    results: list[BatchAnalysisItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def counts_match_results(self) -> "BatchAnalyzeResponse":
        if self.total != self.succeeded + self.failed:
            raise ValueError("total must equal succeeded + failed.")
        if len(self.results) != self.total:
            raise ValueError("total must equal the number of result items.")
        return self


class AnalysisListResponse(APIModel):
    items: list[AnalyzeResponse]
    page: int = Field(ge=1)
    limit: int = Field(ge=1, le=100)
    total: int = Field(ge=0)


class DashboardStats(APIModel):
    total_analyzed: int = Field(ge=0)
    fake_count: int = Field(ge=0)
    real_count: int = Field(ge=0)
    average_confidence: float | None = Field(default=None, ge=0, le=1)


class ModelStatus(APIModel):
    name: str
    loaded: bool
    detail: str | None = None


class HealthResponse(APIModel):
    status: str
    models: list[ModelStatus] = Field(default_factory=list)
    last_loaded_at: datetime | None = None


class ErrorResponse(APIModel):
    detail: str
    code: str | None = None
