import random
from uuid import uuid4
from simulator.agents.base_agent import BaseAgent
from simulator.event_schema import (
    BaseEvent, PageViewEvent, UserSignupEvent, ProductViewEvent
)

class BotAgent(BaseAgent):
    min_gap: float = 0.5
    max_gap: float = 1.2
    bot_type: str = ""
    event_counter: int = 0
    max_bot_events: int = 0

    def generate_session(self) -> list[BaseEvent]:
        self.bot_type = random.choice(["SCRAPER", "SPAMMER"])
        self.max_bot_events = random.randint(100, 500)
        self.event_counter = 0
        return super().generate_session()

    def generate_event(self) -> BaseEvent:
        self.event_counter += 1
        if self.event_counter >= self.max_bot_events:
            self.current_state = "EXIT"
            return PageViewEvent(user_id=self.user_id, 
                                 session_id=self.session_id, 
                                 url="https://shop.com/exit")

        if self.bot_type == "SCRAPER":
            prod_id = f"prod_{(self.event_counter % 50) + 1:03}"
            return ProductViewEvent(user_id=self.user_id, 
                                    session_id=self.session_id, 
                                    product_id=prod_id)
        else:
            new_identity = str(uuid4())
            return UserSignupEvent(user_id=new_identity, 
                                   session_id=self.session_id, 
                                   email=f"fake_{random.getrandbits(24)}@botnet.com")