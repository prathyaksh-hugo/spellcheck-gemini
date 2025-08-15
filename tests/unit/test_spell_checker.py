# tests/unit/test_spell_checker.py
import pytest
from unittest.mock import MagicMock
from src.core.spell_checker import SpellChecker

def test_find_relevant_rules():
    # 1. Arrange
    # Mock the GeminiClient and ChromaDB collection to avoid real API/DB calls
    mock_gemini_client = MagicMock()
    mock_collection = MagicMock()

    # Define what the mock DB should return when queried
    mock_collection.query.return_value = {
        'documents': [["Correction Rule: Use 'Sign in' not 'Log in'"]]
    }

    # Instantiate SpellChecker with the mock client
    spell_checker = SpellChecker(client=mock_gemini_client)
    # Manually replace the real collection with our mock
    spell_checker.collection = mock_collection

    # 2. Act
    rules = spell_checker._find_relevant_rules("please log in")

    # 3. Assert
    assert "Use 'Sign in' not 'Log in'" in rules