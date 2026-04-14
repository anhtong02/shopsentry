from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from uuid import uuid4
from simulator.event_schema import BaseEvent
from datetime import datetime, timezone, timedelta
import random

def generate_random_ip() -> str:
    """Generates a realistic-looking public IPv4 address."""
    # Skipping 10.x.x.x and 192.168.x.x local blocks for realism
    return f"{random.randint(11, 250)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

class BaseAgent(BaseModel, ABC):
    user_id         : str = Field(default_factory=lambda: str(uuid4()))
    session_id      : str = Field(default_factory=lambda: str(uuid4()))
    ip_address      : str = "127.0.0.1" # Default IP
    current_state   : str = "LANDING"
    current_time    : datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    min_gap         : float = 5.0     # seconds between events
    max_gap         : float = 30.0    # subclasses override these
    
    def generate_session(self) -> list[BaseEvent]:
        """The loop: generates events until the user 'leaves' the site."""
        events: list[BaseEvent] = []
        max_events = 1000
        while self.current_state != "EXIT" and len(events) < max_events:
            event = self.generate_event()
            event.timestamp = self.current_time
            
            # ← NEW: Global IP Injection
            # Automatically attaches the agent's current IP to the event payload
            if "ip" not in event.properties:
                event.properties["ip"] = self.ip_address

            self.current_time += timedelta(seconds=random.uniform(self.min_gap, self.max_gap))
            events.append(event)
        return events

    @abstractmethod
    def generate_event(self) -> BaseEvent:
        """Decides what the user does next and returns that event."""
        pass