from unittest.mock import MagicMock
import sys
sys.modules["streamlit"] = MagicMock()

from comparison_engine import run_comparison
import json

# Mock Gemini — we test prompt structure, not actual API response
def mock_gemini(prompt):
    print("\n=== PROMPT SENT TO GEMINI ===")
    print(prompt)
    print("=== END PROMPT ===\n")
    return "Mock LLM review: Notebook B shows improved val_accuracy but suspicious overfitting pattern."

notebook_a_text = """
[CODE]
import torch
BATCH_SIZE = 32
lr = 0.001
torch.manual_seed(42)
conv2d = nn.Conv2d(3, 64, 3)
from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(X, random_state=42)

[OUTPUT]
Epoch [10/10], Train Loss: 0.3500, Val Loss: 0.3800, Val Acc: 82.00%
"""

notebook_b_text = """
[CODE]
import torch
BATCH_SIZE = 64
lr = 0.0001
conv2d = nn.Conv2d(3, 64, 3)
from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(X, test_size=0.2)

[OUTPUT]
Epoch [80/80], Train Loss: 0.1200, Val Loss: 0.4800, Val Acc: 98.50%
"""

# Mock uploaded file objects
mock_file_a = MagicMock()
mock_file_a.name = "run_a.ipynb"
mock_file_a.size = 10000

mock_file_b = MagicMock()
mock_file_b.name = "run_b.ipynb"
mock_file_b.size = 12000

# Mock notebook objects (nbformat structure)
mock_nb_a = MagicMock()
mock_nb_a.cells = []

mock_nb_b = MagicMock()
mock_nb_b.cells = []

schema = run_comparison(
    mock_nb_a, mock_file_a, notebook_a_text,
    mock_nb_b, mock_file_b, notebook_b_text,
    comparison_type="run_vs_run",
    gemini_call_fn=mock_gemini
)

print("=== FINAL SCHEMA SUMMARY ===")
print(f"Winner:       {schema['comparison_result']['winner']}")
print(f"Confidence:   {schema['comparison_result']['confidence']}")
print(f"Risk Flags:   {len(schema['comparison_result']['risk_flags'])} found")
for flag in schema["comparison_result"]["risk_flags"]:
    print(f"  - {flag}")
print(f"\nLLM Review:   {schema['comparison_result']['llm_review']}")