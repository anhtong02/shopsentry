import random
from typing import List
from simulator.agents.base_agent import BaseAgent
from simulator.event_schema import (
    BaseEvent, ProductViewEvent, AddToCartEvent, 
    CheckoutEvent, PaymentEvent, PageViewEvent
)

class FraudAgent(BaseAgent):
    """
    The Fraud Ring consists of thieves.
    Goal: They want to steal high-value inventory using stolen credit cards.
    Behavior: They don't care about the price of socks or USB cables. 
    They want that $2,000 Luxury Watch.
    Pattern: 20 "different" users all land on the site and 
    go directly to the exact same product ID at the exact same time 
    to buy it before the credit card gets flagged.
    """
    min_gap: float = 0.5
    max_gap: float = 2.0
    target_product_id: str
    
    def generate_event(self) -> BaseEvent:
        #1 Landing, then view product

        if self.current_state == "LANDING":
            self.current_state = "PRODUCT_VIEW"
            return PageViewEvent(
                user_id=self.user_id,
                session_id=self.session_id,
                url=f"https://shop.com/p/{self.target_product_id}",
                properties={"ip": self.ip_address}
            )
        # 2. PRODUCT_VIEW -> ADD_TO_CART
        if self.current_state == "PRODUCT_VIEW":
            self.current_state = "ADDING_TO_CART"
            return ProductViewEvent(
                user_id=self.user_id,
                session_id=self.session_id,
                product_id=self.target_product_id,
                properties={"ip": self.ip_address}
            )

        # 3. ADDING_TO_CART -> CHECKOUT
        if self.current_state == "ADDING_TO_CART":
            self.current_state = "CHECKOUT"
            return AddToCartEvent(
                user_id=self.user_id,
                session_id=self.session_id,
                product_id=self.target_product_id,
                properties={"ip": self.ip_address}
            )
        
        # 4. CHECKOUT -> PAYMENT
        if self.current_state == "CHECKOUT":
            self.current_state = "PAYMENT"
            return CheckoutEvent(
                user_id=self.user_id,
                session_id=self.session_id,
                cart_total=999.99, # Fraudsters often go for high-value items
                item_count=1,
                properties={"ip": self.ip_address}
            )

        # 5. PAYMENT -> EXIT
        if self.current_state == "PAYMENT":
            self.current_state = "EXIT"
            return PaymentEvent(
                user_id=self.user_id,
                session_id=self.session_id,
                amount=999.99,
                status="success", # They use 'good' stolen cards
                payment_method="credit_card",
                properties={"ip": self.ip_address}
            )

        self.current_state = "EXIT"
        return PageViewEvent(user_id=self.user_id, 
                             session_id=self.session_id, 
                             url="https://shop.com/exit")
    
class FraudRing:
    """
    Coordinates a group of FraudAgents to attack a specific product
    from a single IP address.
    """
    def __init__(self, size: int = 10):
        self.shared_ip = f"192.168.1.{random.randint(10, 255)}"

        #Generate a random base subnet (e.g., 45.12.88.x)
        base_subnet = f"{random.randint(11, 250)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

        self.target_product = f"luxury_watch_{random.randint(1, 5)}"
        self.agents = [
            FraudAgent(
                ip_address=f"{base_subnet}.{random.randint(1, 254)}", 
                target_product_id=self.target_product
            ) for _ in range(size)
        ]

    def generate_all_events(self) -> List[BaseEvent]:
        all_events = []
        for agent in self.agents:
            all_events.extend(agent.generate_session())
        
        # Shuffle slightly so they are "interleaved" 
        # (This simulates them hitting the server at the same time)
        random.shuffle(all_events)
        return all_events