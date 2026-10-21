
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ask_endpoint(monkeypatch):
    # Mock the RAG pipeline to keep this test fast and predictable.
    def fake_ask(question):
        return {
            "answer": (
                "Applicants may be asked to provide identity "
                "verification, address verification, income proof, "
                "and employment-related documents."
            ),
            "sources": [
                {
                    "document": "sample_loan_policy.txt",
                    "chunk_id": 1,
                    "content": (
                        "Applicants may be asked to provide identity "
                        "verification, address verification, income proof, "
                        "and employment-related documents."
                    ),
                }
            ],
        }

    monkeypatch.setattr("app.main.ask_banking_question", fake_ask)

    response = client.post(
        "/ask",
        json={
            "question": "What documents are needed for a personal loan?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "sources" in data
    assert data["sources"][0]["document"] == "sample_loan_policy.txt"
    assert data["sources"][0]["chunk_id"] == 1
    assert "income proof" in data["sources"][0]["content"]


def test_ask_rejects_short_question():
    response = client.post(
        "/ask",
        json={"question": "Hi"},
    )

    assert response.status_code == 422
