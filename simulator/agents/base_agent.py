from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from uuid import uuid4
from simulator.event_schema import BaseEvent

class BaseAgent(BaseModel, ABC):
    user_id         : str = Field(default_factory=lambda: str(uuid4()))
    session_id      : str = Field(default_factory=lambda: str(uuid4()))
    ip_address      : str = "127.0.0.1" # Default IP
    current_state   : str = "LANDING"

    
    def generate_session(self) -> list[BaseEvent]:
        """The loop: generates events until the user 'leaves' the site."""
        events: list[BaseEvent] = []
        max_events = 1000
        while self.current_state != "EXIT" and len(events) < max_events:
            event = self.generate_event()
            events.append(event)
        return events

    @abstractmethod
    def generate_event(self) -> BaseEvent:
        """Decides what the user does next and returns that event."""
        pass