from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolArgsModel(BaseModel):
    """Shared validation rules for deterministic tool arguments."""

    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class InventoryToolArgs(ToolArgsModel):
    brand: str | None = None
    machine_model: str
    part_name: str
    quality_level: str | None = None


class QuoteToolArgs(ToolArgsModel):
    brand: str | None = None
    machine_model: str
    part_name: str
    quality_level: str
    quantity: int = Field(default=1, ge=1)


class LogisticsToolArgs(ToolArgsModel):
    city: str
    part_name: str
    urgent: bool | None = None


class TicketToolArgs(ToolArgsModel):
    order_id: str
    raw_question: str


class KnowledgeToolArgs(ToolArgsModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=10)


class AgentSlots(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brand: str | None = None
    machine_model: str | None = None
    part_name: str | None = None
    quality_level: str | None = None
    quantity: int | None = Field(default=None, ge=1)
    city: str | None = None
    urgent: bool | None = None
    order_id: str | None = None
    part_number: str | None = None


class AgentParsePlan(BaseModel):
    """Structured semantic parse produced by a LangChain chat model."""

    model_config = ConfigDict(extra="forbid")

    intents: list[
        Literal[
            "inventory",
            "quote",
            "logistics",
            "after_sales",
            "compatibility",
            "diagnosis",
            "general_consulting",
            "image_inspection",
        ]
    ]
    slots: AgentSlots
    confidence: float = Field(ge=0, le=1)
    reason: str


class HumanReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000)
    agent_name: str = Field(default="人工客服", min_length=1, max_length=80)

    @field_validator("message", "agent_name")
    @classmethod
    def normalize_human_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("人工回复和客服名称不能为空")
        return normalized


class HandoffDecision(BaseModel):
    required: bool
    reason_code: str = ""
    reason_text: str = ""
    priority: Literal["普通", "中", "高"] = "普通"


class ImageInspectionResult(BaseModel):
    """Structured, non-authoritative evidence extracted from one image."""

    model_config = ConfigDict(extra="forbid")

    image_type: Literal[
        "nameplate",
        "part_label",
        "part",
        "damage",
        "document",
        "irrelevant",
        "unknown",
    ]
    extracted_text: list[str] = Field(default_factory=list, max_length=50)
    brand: str | None = Field(default=None, max_length=80)
    machine_model: str | None = Field(default=None, max_length=120)
    part_name_candidate: str | None = Field(default=None, max_length=120)
    part_number: str | None = Field(default=None, max_length=160)
    visible_damage: list[str] = Field(default_factory=list, max_length=30)
    observed_features: list[str] = Field(default_factory=list, max_length=30)
    image_quality: Literal["good", "fair", "poor", "unusable"]
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list, max_length=30)
    required_followups: list[str] = Field(default_factory=list, max_length=30)
    safe_for_auto_merge: bool = False

    @field_validator(
        "brand",
        "machine_model",
        "part_name_candidate",
        "part_number",
        mode="before",
    )
    @classmethod
    def normalize_optional_visual_text(cls, value: Any) -> Any:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator(
        "extracted_text",
        "visible_damage",
        "observed_features",
        "warnings",
        "required_followups",
        mode="before",
    )
    @classmethod
    def normalize_visual_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("视觉结果列表字段必须是数组")
        return list(
            dict.fromkeys(
                normalized
                for item in value
                if (normalized := str(item).strip())
            )
        )


class ImageConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["confirm", "edit", "reject", "human"]
    confirmed_fields: dict[str, Any] = Field(default_factory=dict)
    comment: str = Field(default="", max_length=500)

    @field_validator("comment")
    @classmethod
    def normalize_confirmation_comment(cls, value: str) -> str:
        return value.strip()


def dump_args(model: ToolArgsModel) -> dict[str, Any]:
    return model.model_dump()
