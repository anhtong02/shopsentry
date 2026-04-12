import json
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Type, Dict, Any
from uuid import uuid4
from pydantic import BaseModel, Field

# 1. Define the Enum
class EventType(str, Enum):
    PAGEVIEW = "pageview"
    SEARCH = "search"
    ADD_TO_CART = "add_to_cart" 
    PAYMENT = "payment"
    USER_SIGNUP = "user_signup" 
    PRODUCT_VIEW = "product_view"
    CHECKOUT = "checkout"
    CLICK = "click"
class BaseEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str
    properties: dict[str, Any] = Field(default_factory=dict)
    user_id: str

    def to_kafka_value(self) -> bytes:
        # serialize events to bytes
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_kafka_value(cls, data: bytes) -> "BaseEvent":
        obj_dict = json.loads(data.decode("utf-8"))
        return cls(**obj_dict)

# --- Subclasses ---

class PageViewEvent(BaseEvent):
    event_type: EventType = EventType.PAGEVIEW
    url: str = "https://example.com"
    referrer: Optional[str] = None


class SearchEvent(BaseEvent):
    event_type: EventType = EventType.SEARCH
    query: str


class AddToCartEvent(BaseEvent):
    event_type: EventType = EventType.ADD_TO_CART
    product_id: str
    quantity: int = 1  # <--- Add this line!


class CheckoutEvent(BaseEvent):
    event_type: EventType = EventType.CHECKOUT
    cart_total: float
    item_count: int

class PaymentEvent(BaseEvent):
    event_type: EventType = EventType.PAYMENT
    amount: float
    status: str 
    payment_method: str

class UserSignupEvent(BaseEvent):
    event_type: EventType = EventType.USER_SIGNUP
    email: str
    method: str ="email"

class ClickEvent(BaseEvent):
    event_type: EventType = EventType.CLICK
    element_id: str
    target_url: Optional[str] = None

class ProductViewEvent(BaseEvent):
    event_type: EventType = EventType.PRODUCT_VIEW
    product_id: str
    category: Optional[str] = None
    price: Optional[float] = None


# --- The Updated Registry ---

EVENT_MAP: Dict[str, Type[BaseEvent]] = {
    EventType.PAGEVIEW.value: PageViewEvent,
    EventType.SEARCH.value: SearchEvent,
    EventType.ADD_TO_CART.value: AddToCartEvent,
    EventType.CHECKOUT.value: CheckoutEvent,
    EventType.PAYMENT.value: PaymentEvent,
    EventType.USER_SIGNUP.value: UserSignupEvent, # Fixed name here
    EventType.CLICK.value: ClickEvent,
    EventType.PRODUCT_VIEW.value: ProductViewEvent,
}
def deserialize_event(data: bytes) -> BaseEvent:
    """Peeks at the JSON, finds the type, returns the right class instance."""
    payload = json.loads(data.decode("utf-8"))
    event_type_str = payload.get("event_type")
    
    event_class = EVENT_MAP.get(event_type_str)
    
    if not event_class:
        return BaseEvent(**payload)
        
    return event_class(**payload)