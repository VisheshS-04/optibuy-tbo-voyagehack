from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_plan_endpoint_respects_budget():
    r = client.post("/plan", json={"budget": 5e7, "n_scenarios": 30})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["spend"] <= 5e7 + 1
    assert len(body["plan"]) == 16 * 12
