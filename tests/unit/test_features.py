from datetime import datetime, timezone, timedelta
from simulator.event_schema import (
    PageViewEvent, ProductViewEvent, AddToCartEvent, PaymentEvent, UserSignupEvent
)
from pipeline.feature_engine import features as f

# Setup a baseline timestamp for our tests
t0 = datetime.now(timezone.utc)

def test_events_per_minute():
    events = [
        PageViewEvent(event_id="1", timestamp=t0, session_id="s1", user_id="u1", url="/"),
        # Second event happens exactly 30 seconds later
        PageViewEvent(event_id="2", timestamp=t0 + timedelta(seconds=30), session_id="s1", user_id="u1", url="/")
    ]
    # 2 events in 30 seconds = exactly 4 events per minute
    assert f.calculate_events_per_minute(events) == 4.0

def test_unique_pages_and_revisit_ratio():
    events = [
        PageViewEvent(event_id="1", timestamp=t0, session_id="s1", user_id="u1", url="/home"),
        PageViewEvent(event_id="2", timestamp=t0, session_id="s1", user_id="u1", url="/home"), # Revisit
        PageViewEvent(event_id="3", timestamp=t0, session_id="s1", user_id="u1", url="/cart")
    ]
    assert f.calculate_unique_pages_visited(events) == 2.0
    
    # 2 unique URLs / 3 total views = 0.666 unique ratio. Revisit ratio = 1 - 0.666 = 0.333
    assert round(f.page_revisit_ratio(events), 3) == 0.333

def test_cart_to_purchase():
    events = [
        AddToCartEvent(event_id="1", timestamp=t0, session_id="s1", user_id="u1", product_id="p1"),
        PaymentEvent(event_id="2", timestamp=t0, session_id="s1", user_id="u1", amount=10.0, status="success"
                     , payment_method="credit_card")
    ]
    assert f.calculate_cart_to_purchase_ratio(events) == 1.0

def test_session_duration_and_gaps():
    events = [
        PageViewEvent(event_id="1", timestamp=t0, session_id="s1", user_id="u1", url="/"),
        PageViewEvent(event_id="2", timestamp=t0 + timedelta(seconds=10), session_id="s1", user_id="u1", url="/"),
        PageViewEvent(event_id="3", timestamp=t0 + timedelta(seconds=20), session_id="s1", user_id="u1", url="/")
    ]
    assert f.session_duration_seconds(events) == 20.0
    assert f.calculate_avg_time_between_events(events) == 10.0

def test_diversity_and_payment():
    events = [
        PageViewEvent(event_id="1", timestamp=t0, session_id="s1", user_id="u1", url="/"),
        PaymentEvent(event_id="2", timestamp=t0, session_id="s1", user_id="u1", amount=10.0, status="success"
                     , payment_method="credit_card")
    ]
    assert f.event_type_diversity(events) == 2
    assert f.has_payment(events) == 1

def test_signup_to_purchase_speed():
    events = [
        UserSignupEvent(event_id="1", timestamp=t0, session_id="s1", user_id="u1", email="test@test.com"),
        PaymentEvent(event_id="2", timestamp=t0 + timedelta(seconds=5), 
                     session_id="s1", user_id="u1", amount=10.0, status="success"
                     , payment_method="credit_card")
    ]
    assert f.signup_to_purchase_speed(events) == 5.0 