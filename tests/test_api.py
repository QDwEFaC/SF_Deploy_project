def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "Ok"
    assert "version" in data


def test_predict(client, sample_payload):
    sample_payload["LIMIT_BAL"] = 200000
    sample_payload["AGE"] = 35
    r = client.post("/predict", json=sample_payload)
    assert r.status_code == 200
    data = r.get_json()
    assert "prediction" in data
    assert "probability_default" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability_default"] <= 1


def test_missing_feature(client, sample_payload):
    del sample_payload["AGE"]
    r = client.post("/predict", json=sample_payload)
    assert r.status_code == 400
    assert "Missing" in r.get_json()["error"]


def test_not_json(client):
    r = client.post("/predict", data="not-json", content_type="text/plain")
    assert r.status_code == 400