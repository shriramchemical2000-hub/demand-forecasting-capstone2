"""Optional LLM helper. Disabled by default so the pipeline is fully reproducible offline.

When ``llm.enabled`` is true and ANTHROPIC_API_KEY is set, agents ask Claude to turn
their structured findings into a short rationale. If anything fails, the template
text is used instead, so the numbers never depend on the LLM.
"""
import os


class LLMHelper:
    def __init__(self, cfg: dict):
        llm_cfg = cfg.get("llm", {})
        self.enabled = bool(llm_cfg.get("enabled")) and bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.model = llm_cfg.get("model", "claude-sonnet-5-5")
        self._client = None

    def explain(self, fallback: str, facts: str) -> str:
        if not self.enabled:
            return fallback
        try:
            if self._client is None:
                import anthropic

                self._client = anthropic.Anthropic()
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=300,
                system=(
                    "You are part of a supply-chain planning team. Write a concise, factual "
                    "rationale (max 3 sentences) using only the facts provided."
                ),
                messages=[{"role": "user", "content": facts}],
            )
            return msg.content[0].text.strip()
        except Exception:  # never let the LLM break the pipeline
            return fallback
