import random
from simulator.agents.base_agent import BaseAgent
from simulator.constants import SEARCH_QUERIES, PRODUCT_IDS
from simulator.event_schema import (
    BaseEvent, PageViewEvent, SearchEvent, AddToCartEvent, 
    CheckoutEvent, PaymentEvent, UserSignupEvent, ProductViewEvent
)

class NormalUser(BaseAgent):
    def generate_event(self) -> BaseEvent:
        # 1. LANDING -> SIGNUP or BROWSING
        if self.current_state == "LANDING":
            if random.random() < 0.2: # 20% of new visitors sign up
                self.current_state = "SIGNUP"
                return UserSignupEvent(
                    user_id=self.user_id, 
                    session_id=self.session_id, 
                    email=f"{self.user_id}@example.com"
                )
            self.current_state = "BROWSING"
            return PageViewEvent(user_id=self.user_id, session_id=self.session_id, url="https://shop.com/home")

        # 2. SIGNUP -> BROWSING
        if self.current_state == "SIGNUP":
            self.current_state = "BROWSING"
            return PageViewEvent(user_id=self.user_id, session_id=self.session_id, url="https://shop.com/welcome")

        # 3. BROWSING (The main loop)
        if self.current_state == "BROWSING":
            choice = random.choices(
                ["SEARCH", "VIEW_PROD", "EXIT"], 
                weights=[0.3, 0.5, 0.2]
            )[0]
            
            if choice == "SEARCH":
                self.current_state = "SEARCHING"
                return SearchEvent(user_id=self.user_id, session_id=self.session_id, query=random.choice(SEARCH_QUERIES))
            elif choice == "VIEW_PROD":
                product_id = random.choice(PRODUCT_IDS)
                # Decide if they like the product enough to add to cart
                if random.random() < 0.12: # 15% chance to add to cart
                    self.current_state = "ADDING_TO_CART"
                return ProductViewEvent(
                    user_id=self.user_id, 
                    session_id=self.session_id, 
                    product_id=product_id
                )
            else:
                self.current_state = "EXIT"
                return PageViewEvent(user_id=self.user_id, session_id=self.session_id, url="https://shop.com/exit")

        # 4. SEARCHING -> BROWSING
        if self.current_state == "SEARCHING":
            self.current_state = "BROWSING"
            return PageViewEvent(user_id=self.user_id, session_id=self.session_id, url="https://shop.com/search-results")

        # 5. ADDING_TO_CART -> CHECKOUT or BROWSING
        if self.current_state == "ADDING_TO_CART":
            event = AddToCartEvent(user_id=self.user_id, session_id=self.session_id, product_id=random.choice(PRODUCT_IDS))
            self.current_state = "CHECKOUT" if random.random() < 0.7 else "BROWSING"
            return event

        # 6. CHECKOUT -> PAYMENT
        if self.current_state == "CHECKOUT":
            self.current_state = "PAYMENT"
            return CheckoutEvent(user_id=self.user_id, session_id=self.session_id, cart_total=random.uniform(20.0, 500.0), item_count=random.randint(1, 5))

        # 7. PAYMENT -> EXIT
        if self.current_state == "PAYMENT":
            status = "success" if random.random() < 0.95 else "failed"
            self.current_state = "EXIT"
            return PaymentEvent(
                user_id=self.user_id, 
                session_id=self.session_id, 
                amount=random.uniform(20.0, 500.0), 
                status=status, 
                payment_method="credit_card"
            )

        self.current_state = "EXIT"
        return PageViewEvent(user_id=self.user_id, 
                             session_id=self.session_id, 
                             url="https://shop.com/exit")