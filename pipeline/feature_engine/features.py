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
###
from typing import List
from simulator.event_schema import (
    BaseEvent, PageViewEvent, ProductViewEvent, AddToCartEvent, PaymentEvent
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