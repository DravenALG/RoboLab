"""Explicit policy failures that consume an evaluation episode."""


class PolicyExecutionError(ValueError):
    """Invalid policy output/protocol; infrastructure exceptions stay separate."""

    def __init__(self, message: str, *, error_type: str = "PolicyExecutionError") -> None:
        super().__init__(message)
        self.error_type = error_type

    def as_record(self) -> dict:
        return {"error_kind": "policy_execution", "error_type": self.error_type, "error": str(self)}
