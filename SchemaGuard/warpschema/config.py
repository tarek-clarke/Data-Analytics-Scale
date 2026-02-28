"""
WarpSchema Configuration
Copyright (c) 2026 Tarek Clarke. All Rights Reserved.

Defines target schemas, model settings, and similarity thresholds.
"""

from dataclasses import dataclass, field


@dataclass
class SchemaGuardConfig:
    """Central configuration for the SchemaGuard engine."""

    # BERT model used for semantic key matching.
    # distilbert-base-uncased is ~2x faster than bert-base with comparable accuracy.
    model_name: str = "distilbert-base-uncased"

    # Minimum cosine similarity score to accept a key mapping.
    # Keys scoring below this threshold are flagged as unmapped.
    similarity_threshold: float = 0.85

    # Maximum sequence length passed to the tokenizer (keeps latency low).
    max_token_length: int = 32

    # When True, unmapped keys are dropped from the output payload.
    # When False, they are preserved under an "_unmapped" namespace.
    drop_unmapped: bool = False

    # Pre-warm the model on startup to avoid cold-start latency on first request.
    pre_warm: bool = True


# ---------------------------------------------------------------------------
# Target Schemas
# ---------------------------------------------------------------------------
# Each schema is a dict mapping canonical field names to their expected types.
# The engine matches incoming keys against these canonical names.

TARGET_SCHEMAS: dict[str, dict[str, str]] = {
    "telemetry_v1": {
        "engine_rpm": "float",
        "oil_pressure": "float",
        "coolant_temp": "float",
        "vehicle_speed": "float",
        "throttle_position": "float",
        "fuel_level": "float",
        "battery_voltage": "float",
        "timestamp": "str",
        "device_id": "str",
    },
    "sensor_v1": {
        "temperature": "float",
        "humidity": "float",
        "pressure": "float",
        "altitude": "float",
        "sensor_id": "str",
        "timestamp": "str",
    },
}

# Default schema used when no schema name is specified in the request.
DEFAULT_SCHEMA = "telemetry_v1"
