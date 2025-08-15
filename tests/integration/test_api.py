# tests/integration/test_api.py
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_check_spelling_with_brand_rules():
    # 1. Arrange
    payload = {
        "texts": [
            "Please log in to see your save account"
        ]
    }
    
    # 2. Act
    response = client.post("/v1/check", json=payload)
    response_data = response.json()
    
    # 3. Assert
    assert response.status_code == 200
    result = response_data['results'][0]
    corrected_text = result['corrected_text']
    
    # NEW, more flexible assertions:
    # Check that the key brand rules were applied, regardless of minor rewording.
    assert "Sign in" in corrected_text
    assert "Save Account" in corrected_text
    
    # We can also still check the corrections_log
    assert "Changed 'log in' to 'Sign in'" in result['corrections_log'][0]
    assert "Capitalized 'Save Account'" in result['corrections_log'][1]