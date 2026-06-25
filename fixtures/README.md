# Multimodal fixtures

Drop demo inputs here to exercise the multimodal paths without the UI:

- **Image** (`*.png` / `*.jpg`) — a product screenshot, receipt, or damaged-item photo.
  ```python
  from app.llm import groq_client
  print(groq_client.vision("Product, brand, INR price, any defect?", image_path="fixtures/product.png"))
  ```
- **Audio** (`*.wav` / `*.mp3` / `*.m4a`) — a spoken complaint or shopping voice-note.
  ```python
  from app.media import generate
  print(generate.transcribe_voice("fixtures/complaint.m4a"))
  ```

A generated SVG deal card (offline, no inputs needed) is written to `data/media/` by
`app/media/generate.py::generate_deal_card`. For raster image/TTS generation see
`RUNBOOK.md` §9 (Hugging Face Inference or an external provider).
