import random
from datetime import timedelta
from typing import List
from simulator.agents.base_agent import BaseAgent
from simulator.event_schema import (
    BaseEvent, PageViewEvent, ProductViewEvent
)

class ChurningUser(BaseAgent):
    session_index: int = 0

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

    
    def generate_lifecycle(self) -> List[BaseEvent]:
        """Generates a series of sessions with realistic, random gaps."""
        all_events: List[BaseEvent] = []
        cumulative_days = 0
        
        for i in range(5):
            self.session_index = i
            self.current_state = "LANDING"
            self.session_id = str(random.getrandbits(32))
            
            # 1. Randomize the gap between sessions (e.g., 3 to 14 days)
            # This makes the 'ghosting' feel much more organic.
            if i > 0:
                cumulative_days += random.randint(6, 24)
            
            session_events = self.generate_session()
            time_offset = timedelta(days=cumulative_days)
            
            for e in session_events:
                # 2. Also add a tiny bit of random minutes/seconds 
                # so they don't all start at the exact same second of the day.
                intra_day_offset = timedelta(
                    minutes=random.randint(0, 59), 
                    seconds=random.randint(0, 59)
                )
                e.timestamp = e.timestamp + time_offset + intra_day_offset
                
            all_events.extend(session_events)
            
        return all_events