from simulator.agents.normal_user import NormalUser
from simulator.event_schema import (
    PageViewEvent, 
    UserSignupEvent,
    ProductViewEvent,
    PaymentEvent)
from simulator.agents.fraud_ring import FraudRing

def test_normal_user_session_logic() -> None :
    # 1. Setup
    user = NormalUser()
    session = user.generate_session()
    
    # 2. Check: Non-empty list
    assert len(session) > 0, "Session should not be empty"
    
    # 3. Check: Consistent session_id
    session_ids = {e.session_id for e in session}
    assert len(session_ids) == 1, "All events must share the same session_id"
    assert list(session_ids)[0] == user.session_id
    
    # 4. Check: Landing Logic
    # The first event must be either a PageView (Home) or a UserSignup
    first_event = session[0]
    assert isinstance(first_event, (PageViewEvent, UserSignupEvent)), \
        f"First event should be PageView or Signup, got {type(first_event)}"

    # 5. Check: Safety Limit & Exit
    # Ensure it didn't just hit the 1000 limit (Normal users shouldn't browse that much)
    assert len(session) < 1000, "Normal user session is suspiciously long (possible loop)"
    assert user.current_state == "EXIT", "User session should end in EXIT state"

def test_normal_user_purchase_flow() -> None:
    """Verify that if a payment exists, it followed a checkout."""
    # We might need to run this a few times to catch a session that actually buys
    found_purchase = False
    for _ in range(100): # Run 100 times to find at least one buyer
        user = NormalUser()
        session = user.generate_session()
        
        types = [e.event_type.value for e in session]
        if "payment" in types:
            found_purchase = True
            pay_idx = types.index("payment")
            check_idx = types.index("checkout")
            # Payment must happen AFTER checkout
            assert check_idx < pay_idx, "Payment occurred before Checkout!"
            break
    
    assert found_purchase, "In 100 tries, no NormalUser made a purchase. Check your weights!"

def test_fraud_ring_coordination() -> None :
    """Verify that a fraud ring acts as a coordinated unit."""
    # 1. Setup a ring of 5 attackers
    ring_size = 5
    ring = FraudRing(size=ring_size)
    events = ring.generate_all_events()

    # 2. Check: Shared IP (The "Smoking Gun")
    # All events should have the same IP address in their properties
    ips = {e.properties.get("ip") for e in events}
    assert len(ips) == 1, "Fraud ring must share exactly one IP address"
    assert None not in ips, "All fraud events must have an IP address"

    # 3. Check: Target Product (The "Heist")
    # All product-related events must target the exact same ID
    target_ids = {
        e.product_id for e in events 
        if isinstance(e, ProductViewEvent) or hasattr(e, 'product_id')
    }
    assert len(target_ids) == 1, "Fraud ring must target exactly one product"
    
    # 4. Check: Identity Count
    # There should be exactly 'ring_size' unique users
    user_ids = {e.user_id for e in events}
    assert len(user_ids) == ring_size, f"Expected {ring_size} unique users in the ring"

def test_fraud_ring_conversion_rate() -> None:
    """Verify that fraud agents always complete their purchases."""
    ring_size = 3
    ring = FraudRing(size=ring_size)
    events = ring.generate_all_events()

    # In our FraudAgent logic, conversion should be 100%
    success_payments = [
        e for e in events 
        if isinstance(e, PaymentEvent) and e.status == "success"
    ]
    
    assert len(success_payments) == ring_size, \
        f"Fraud ring of {ring_size} should have {ring_size} successful payments"