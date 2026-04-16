###
#----FEATURE ENGINEERING----
#This file takes a raw list of events (a session) 
# and return a single floating-point number.
#
#----PURPOSE---
#ML models can't "read" a list of JSON objects directly. 
#they need a fixed-size table of numbers. 
#This is the "Translator" that turns a user's story into a mathematical vector.
#
#
#currently each event has: event_id, event_type, timestamp, session_id, user_id, properties

from typing import List
from simulator.event_schema import (
    BaseEvent, PageViewEvent, ProductViewEvent, AddToCartEvent, PaymentEvent,
    UserSignupEvent
)

def calculate_events_per_minute(events: List[BaseEvent]) -> float:
    """
    What it measures: How fast the user is clicking. Are they a bot or a human?

    Algorithm: Delta-T Velocity calculation.
    Logic: (Total Events / Total Seconds) * 60.
    """

    if len(events) < 2:
        return 0.0
    
    # Ensure chronological order for math
    sorted_events = sorted(events, key=lambda x: x.timestamp)
    duration_seconds = (sorted_events[-1].timestamp - sorted_events[0].timestamp).total_seconds()

    # Handle instantaneous bursts (common in bots)
    if duration_seconds <= 0:
        return float(len(events)) 
        
    return (len(events) / duration_seconds) * 60

def calculate_unique_pages_visited(events: List[BaseEvent]) -> float:
    """
    Technique: Set De-duplication.
    Logic: Count distinct URLs and Product IDs.
    """
    visited = set()
    for e in events:
        if isinstance(e, PageViewEvent):
            visited.add(e.url)
        elif isinstance(e, ProductViewEvent):
            visited.add(e.product_id)
    return float(len(visited))

def calculate_avg_time_between_events(events: List[BaseEvent]) -> float:
    """
    Technique: Mean Inter-arrival Time.
    Logic: Sum of gaps between events / count of gaps.
    """
    if len(events) < 2:
        return 0.0
    
    sorted_events = sorted(events, key=lambda x: x.timestamp)
    gaps = []

    for i in range(len(sorted_events) - 1):
        gap = (sorted_events[i+1].timestamp - sorted_events[i].timestamp).total_seconds()
        gaps.append(gap)

    return sum(gaps) / len(gaps) if gaps else 0.0

def calculate_cart_to_purchase_ratio(events: List[BaseEvent]) -> float:
    """
    What it measures: Conversion rate. Of people who put something in their cart, how many actually paid?
    Technique: Conversion Rate Analysis.
    Logic: Successful Payments / Total AddToCarts.
    """
    adds = sum(1 for e in events if isinstance(e, AddToCartEvent))
    buys = sum(1 for e in events if isinstance(e, PaymentEvent) and e.status =="success")
    if adds == 0:
        return 0.0
        
    return float(buys / adds)

def session_duration_seconds(events: List[BaseEvent]) -> float:
    """
    What it measures: duration of a session in seconds
    Technique: just take time of last event minus time of first event.
    Logic: Longer sessions = more engaged users. Bots often have very short or unusually long sessions
    """
    if len(events) < 2:
        return 0.0
    sorted_events = sorted(events, key=lambda x: x.timestamp)
    total_seconds = (sorted_events[-1].timestamp - sorted_events[0].timestamp).total_seconds()
        
    return total_seconds

def event_type_diversity(events: List[BaseEvent]) -> int:
    """
    What it measures: how many event types during this session
    Technique: 
    Logic: Bots do one thing repeatedly (low diversity), humans do many things (high diversity).

    """
    types = {e.event_type for e in events}
    return len(types)

def has_payment(events: List[BaseEvent]) -> int:
    """
    What it measures: whether that session succesfully pays for something
    Technique:
    Logic: bots never pay, fraud rings always pay, normal sometimes pay
    """
    for e in events:
        if isinstance(e, PaymentEvent) and e.status == "success":
            return 1
    
    return 0

def signup_to_purchase_speed(events: List[BaseEvent]) -> float:

    """
    What it measures: time in seconds of when user sign up and purchases
    Logic: Fraud rings sign up and buy within seconds (stolen cards, rush before detection). 
    Real users take minutes, hours, or days. Returns 0 if either event is missing.
    """
    time_of_signup = None
    time_of_purchase = None
    for e in events:
        if isinstance(e, UserSignupEvent):
            time_of_signup = e.timestamp
        elif isinstance(e, PaymentEvent) and e.status == "success":
            time_of_purchase = e.timestamp
    if time_of_signup is None or time_of_purchase is None:
        return 0.0
    return (time_of_purchase - time_of_signup).total_seconds()

def page_revisit_ratio(events: List[BaseEvent]) -> float:
    """
    What it measures: fraction of page views that were revisits vs new pages.
    Technique: 1 - (unique URLs / total page views).
    Logic: scrapers hit the same page over and over. Humans explore.
    1.0 being very high of revisit ratiom 0.0 being all pages visited were unique and likely not bots.
    """

    urls = [e.url for e in events if isinstance(e, PageViewEvent)]
    if len(urls) == 0:
        return 0.0
    unique = len(set(urls))
    return 1.0 - (unique / len(urls))
