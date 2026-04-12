import random
from uuid import uuid4
from typing import List
from simulator.agents.base_agent import BaseAgent
from simulator.event_schema import (
    BaseEvent, PageViewEvent, UserSignupEvent, ProductViewEvent
)

class BotAgent(BaseAgent):
    """
    The BotAgent doesn't care about the 'funnel'. 
    It has a specific mission: Scrape data or Spam signups.
    """
    def generate_session(self) -> List[BaseEvent]:
        events: list[BaseEvent] = []
        bot_type = random.choice(["SCRAPER", "SPAMMER"])
        num_events = random.randint(100, 500)


        for i in range(num_events):
            #SCRAPER: Goal: It wants to steal your data, not your products. 
            #It wants to know the price of every item in the shop to 
            # put it on a price-comparison site.
            # Pattern: It hits product 1, then 2, then 3... across the whole catalog.

            if bot_type == "SCRAPER":
                prod_id = f"prod_{ (i % 50) + 1 :03}"
                events.append(ProductViewEvent(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    product_id=prod_id
                ))
            
            elif bot_type == "SPAMMER":
                new_identity = str(uuid4())
                events.append(UserSignupEvent(
                    user_id=new_identity,
                    session_id = self.session_id,
                    email=f"fake_user_{random.getrandbits(24)}@botnet.com"
                ))

        return events
    
    def generate_event(self) -> BaseEvent:
        return PageViewEvent(user_id=self.user_id, 
                             session_id=self.session_id, 
                             url="https://shop.com/bot-activity")