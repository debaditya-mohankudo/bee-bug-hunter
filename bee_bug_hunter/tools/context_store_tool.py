"""Tools bound to one ContextStore instance, private to whichever agent they
were constructed for -- see agents.py. Not shared with other agents."""
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions
from pydantic import BaseModel, Field

from bee_bug_hunter.context_store import ContextStore


class AddObservationInput(BaseModel):
    kind: str = Field(..., description="Short category for this observation, e.g. 'schema', 'error', 'sql', 'timing'")
    text: str = Field(..., description="The atomic fact itself -- one concrete observation, not a summary of your whole turn")


class AddObservationTool(Tool[AddObservationInput, ToolRunOptions, StringToolOutput]):
    name = "add_observation"
    description = (
        "Record one atomic observation (a single concrete fact -- a column name, an error message, "
        "a query, a timing) to your own private scratchpad, so you can refer back to it later in "
        "your own reasoning instead of re-deriving it. Do not use this for summaries -- one fact "
        "per call. This scratchpad is yours alone; no other agent sees it."
    )
    input_schema = AddObservationInput

    def __init__(self, store: ContextStore, **kwargs) -> None:
        super().__init__(**kwargs)
        self.store = store

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "add_observation"], creator=self)

    async def _run(self, input: AddObservationInput, options, context) -> StringToolOutput:
        obs_id = self.store.add(input.kind, input.text)
        return StringToolOutput(f"Recorded observation #{obs_id}.")


class ListObservationsInput(BaseModel):
    pass


class ListObservationsTool(Tool[ListObservationsInput, ToolRunOptions, StringToolOutput]):
    name = "list_observations"
    description = (
        "List every atomic observation you've recorded so far in your own private scratchpad."
    )
    input_schema = ListObservationsInput

    def __init__(self, store: ContextStore, **kwargs) -> None:
        super().__init__(**kwargs)
        self.store = store

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "list_observations"], creator=self)

    async def _run(self, input: ListObservationsInput, options, context) -> StringToolOutput:
        obs = self.store.list()
        if not obs:
            return StringToolOutput("No observations recorded yet.")
        return StringToolOutput("\n".join(f"#{o['id']} [{o['kind']}] {o['text']}" for o in obs))
