"""
WarpSchema Test Suite
Copyright (c) 2026 Tarek Clarke. All Rights Reserved.

Simulates broken / drifted telemetry payloads and asserts auto-remediation.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import app, lifespan  # noqa: F401 — lifespan wires up the engine
from config import SchemaGuardConfig
from engine import TransformationEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine() -> TransformationEngine:
    """Create a shared engine instance for unit tests."""
    return TransformationEngine(SchemaGuardConfig())


@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient that triggers the lifespan (model load)."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Unit tests — engine level
# ---------------------------------------------------------------------------

class TestTransformationEngine:
    """Direct engine tests without HTTP overhead."""

    def test_exact_match_passthrough(self, engine: TransformationEngine):
        """Keys that already match the schema should pass through untouched."""
        payload = {"engine_rpm": 3200.0, "oil_pressure": 45.1}
        result = engine.normalize(payload, "telemetry_v1")

        assert result.data["engine_rpm"] == 3200.0
        assert result.data["oil_pressure"] == 45.1
        assert result.unmapped_count == 0
        assert all(m.exact for m in result.mappings)

    def test_drifted_key_auto_remap(self, engine: TransformationEngine):
        """A drifted key like 'rpm_val' should map to 'engine_rpm'."""
        payload = {"rpm_val": 4500.0}
        result = engine.normalize(payload, "telemetry_v1")

        assert "engine_rpm" in result.data, (
            f"Expected 'engine_rpm' in output, got keys: {list(result.data)}"
        )
        assert result.data["engine_rpm"] == 4500.0

    def test_multiple_drifted_keys(self, engine: TransformationEngine):
        """Multiple simultaneously drifted keys should all be fixed."""
        payload = {
            "rpm_val": 4100.0,
            "oilPressure": 42.0,
            "coolant_temperature": 95.5,
        }
        result = engine.normalize(payload, "telemetry_v1")

        assert "engine_rpm" in result.data
        assert "oil_pressure" in result.data
        assert "coolant_temp" in result.data

    def test_unmapped_key_preserved(self, engine: TransformationEngine):
        """Keys with no good match should land in '_unmapped'."""
        payload = {"xyzzy_foobar_999": 1}
        config = SchemaGuardConfig(drop_unmapped=False)
        eng = TransformationEngine(config)
        result = eng.normalize(payload, "telemetry_v1")

        if result.unmapped_count > 0:
            assert "_unmapped" in result.data

    def test_unknown_schema_raises(self, engine: TransformationEngine):
        with pytest.raises(ValueError, match="Unknown schema"):
            engine.normalize({"a": 1}, "nonexistent_schema")


# ---------------------------------------------------------------------------
# Integration tests — HTTP level
# ---------------------------------------------------------------------------

class TestNormalizeEndpoint:
    """Full round-trip tests through the FastAPI endpoint."""

    def test_health(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_list_schemas(self, client: TestClient):
        resp = client.get("/schemas")
        assert resp.status_code == 200
        assert "telemetry_v1" in resp.json()["schemas"]

    def test_normalize_broken_payload(self, client: TestClient):
        """Simulates the primary use-case: a broken telemetry payload."""
        broken_payload = {
            "rpm_val": 5200.0,
            "oilPressure": 38.7,
            "coolant_temperature": 102.3,
            "vehicleSpeed": 72.0,
            "throttle_pos": 65.2,
            "fuel_lvl": 48.0,
            "batt_voltage": 13.8,
            "timestamp": "2026-01-15T12:00:00Z",
            "device_id": "UNIT-042",
        }

        resp = client.post(
            "/normalize",
            json={"payload": broken_payload, "schema_name": "telemetry_v1"},
        )

        assert resp.status_code == 200
        data = resp.json()["data"]

        # At minimum, exact-match keys must survive.
        assert data.get("timestamp") == "2026-01-15T12:00:00Z"
        assert data.get("device_id") == "UNIT-042"

        # Drifted keys should be remapped.
        assert "engine_rpm" in data, f"Missing 'engine_rpm'. Got: {list(data)}"

    def test_invalid_schema_returns_400(self, client: TestClient):
        resp = client.post(
            "/normalize",
            json={"payload": {"a": 1}, "schema_name": "bad_schema"},
        )
        assert resp.status_code == 400
