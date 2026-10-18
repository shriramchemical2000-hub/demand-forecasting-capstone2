from .state import PipelineState


class BaseAgent:
    name = "Agent"
    role = ""

    def __init__(self, state: PipelineState):
        self.state = state
        self.cfg = state.cfg

    def _log(self, kind, text, **data):
        return self.state.tracer.log(self.name, kind, text, data)

    def think(self, text, **data):
        return self._log("thought", text, **data)

    def act(self, text, **data):
        return self._log("action", text, **data)

    def observe(self, text, **data):
        return self._log("observation", text, **data)

    def decide(self, text, **data):
        return self._log("decision", text, **data)

    def request_revision(self, receiver: str, text: str, **data):
        return self._log("revision_request", f"-> {receiver}: {text}", **data)
