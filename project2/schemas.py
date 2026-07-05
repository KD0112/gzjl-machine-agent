from __future__ import annotations

from typing import Any

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


def dump_args(model: ToolArgsModel) -> dict[str, Any]:
    return model.model_dump()
