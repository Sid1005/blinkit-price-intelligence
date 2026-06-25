"""India Commerce SignalForge — one shared Indian-commerce decision engine.

Three decision surfaces over a single agent spine:
  * Deal predictor      — INR price band + festival context + buy/wait/avoid.
  * Substitution ranker  — alternatives when a SKU is unavailable / bad value.
  * Complaint triage     — classify complaints + draft policy-grounded resolutions.

Runtime LLM inference is Groq-only; open models/datasets/Hub are Hugging Face-only;
web evidence is gathered with Tavily. See RUNBOOK.md for external credentials.
"""

__version__ = "1.0.0"
