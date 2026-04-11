from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4
import json
from enum import Enum


# 1. Define the Enum
class EventType(str, Enum):
    PAGEVIEW = "pageview"
    SEARCH = "search"
    ADD_TO_CART = "addtocart"
    CHECKOUT = "checkout"
    PAYMENT = "payment"
    USER_SIGNUP = "usersignup"
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


class PaymentEvent(BaseEvent):
    event_type: EventType = EventType.PAYMENT


class UserSignupEvent(BaseEvent):
    event_type: EventType = EventType.USER_SIGNUP


class ClickEvent(BaseEvent):
    event_type: EventType = EventType.CLICK
