"""Private scratchpad for a single analysis agent's own reasoning within one
task execution -- not shared across agents. Same shape as SeniorDevAgent/
scripts/find_bug_ollama.py's ContextStore: a plain list of dicts, append or
retract, no embeddings/similarity search. Each agent that wants one gets its
own instance (see agents.py); nothing here is global or keyed by run_id."""


class ContextStore:
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self._observations: list[dict] = []
        self._next_id = 0

    def add(self, kind: str, text: str) -> int:
        obs_id = self._next_id
        self._next_id += 1
        self._observations.append({"id": obs_id, "kind": kind, "text": text})
        return obs_id

    def list(self) -> list[dict]:
        return list(self._observations)

    def retract(self, obs_id: int) -> bool:
        for i, obs in enumerate(self._observations):
            if obs["id"] == obs_id:
                self._observations.pop(i)
                return True
        return False
