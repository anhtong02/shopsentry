import random
from datetime import timedelta 
from simulator.agents.base_agent import BaseAgent
from simulator.event_schema import (
    BaseEvent, PageViewEvent, ProductViewEvent
)

class ChurningUser(BaseAgent):
    session_index: int = 0
    min_gap: float = 8.0
    max_gap: float = 45.0
    agent_type: str = "churning"

    def generate_event(self) -> BaseEvent:
        # The 'Boredom Factor' increases with every session
        # session_index 0 = 0% bored, session_index 5 = 100% bored
        boredom = min(self.session_index * 0.25, 1.0)

        if self.current_state == "LANDING":
            self.current_state = "BROWSING"
            return PageViewEvent(user_id=self.user_id, 
                                 session_id=self.session_id, 
                                 url="https://shop.com/home")
        

        if self.current_state == "BROWSING":
            choice = random.choices(
                ["VIEW_PROD", "EXIT"],
                weights = [1-boredom, boredom + 0.1]
            )[0]
            
            if choice == "VIEW_PROD" and boredom < 0.8:
                return ProductViewEvent(
                    user_id=self.user_id, 
                    session_id=self.session_id, 
                    product_id=f"prod_{random.randint(1,10)}"
                )
            else:
                self.current_state = "EXIT"
                return PageViewEvent(user_id=self.user_id, 
                                     session_id=self.session_id, 
                                     url="https://shop.com/exit")
            
        self.current_state = "EXIT"
        return PageViewEvent(user_id=self.user_id, 
                             session_id=self.session_id, 
                             url="https://shop.com/exit")

    
    def generate_lifecycle(self) -> list[BaseEvent]:
        all_events: list[BaseEvent] = []
        for i in range(5):
            self.session_index = i
            self.current_state = "LANDING"
            self.session_id = str(random.getrandbits(32))
            if i > 0:
                self.current_time += timedelta(days=random.randint(6, 24))
            session_events = self.generate_session()
            all_events.extend(session_events)
        return all_events