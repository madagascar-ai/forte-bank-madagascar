# Working on Forte Guard

- Follow the user's pipeline: frames -> detection/tracking -> object journal -> action journal -> parallel rule engine and local LLM -> merged candidates -> minimal validation -> UI/API.
- Runtime analysis must stay local. Download dependencies/weights only during explicit preparation. Never transmit dataset videos, frames or derived descriptions to cloud AI, including through tool output.
- Never commit datasets, weights, generated job artifacts, API keys or `.env`. Runtime must not read `open_labels.xlsx`.
- Shared module boundaries are Pydantic models in `src/forte_guard/contracts.py`. Keep normalized coordinates and real decoded frame timestamps. Preserve uncertainty and distinguish unknown from false.
- Vision ownership: `vision/`; event/state ownership: `journal/`; rule/LLM ownership: `reasoning/`. Keep API and UI independent of detector implementation.
- Minimal verification is intentional for this MVP. Do not silently add per-candidate VLM calls or claim visual verification has occurred.
- Update meaningful synthetic regression tests when changing temporal inference, omission logic, candidate merging, authentication or API contracts. Run `python -m pytest -q` and `ruff check src scripts examples tests`.
- Do not claim recognition accuracy from passing integration tests. Normative mappings with `needs_review` must stay visibly provisional.
