from simulator.agents.normal_user import NormalUser
from simulator.event_schema import (
    PageViewEvent, 
    UserSignupEvent,
    PaymentEvent,
    BaseEvent)
from simulator.agents.fraud_ring import FraudRing
from simulator.agents.churning_user import ChurningUser
from simulator.agents.bot_agent import BotAgent
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

def test_fraud_ring_coordination() -> None:
    """Verify that a fraud ring acts as a coordinated unit within a subnet."""
    # 1. Setup a ring of 5 attackers
    ring_size = 5
    ring = FraudRing(size=ring_size)
    events = ring.generate_all_events()

    # 2. Check: Shared Subnet (The new "Smoking Gun")
    # We grab the first three parts of the IP (e.g., "192.168.1")
    subnets = {'.'.join(e.properties.get("ip", "").split('.')[:3]) for e in events}
    
    assert len(subnets) == 1, "Fraud ring must share the same Class C subnet"
    
    # Optional: Verify they are using multiple different IPs within that subnet
    ips = {e.properties.get("ip") for e in events}
    assert len(ips) > 1, "Fraud ring should have multiple unique IPs within the subnet"

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
    

def test_churning_user_lifecycle_decay() -> None:
    """Check that a churning user eventually ghost us (fewer events over time)."""
    user = ChurningUser()
    lifecycle = user.generate_lifecycle()
    
    # Group events by session_id to count them
    sessions: dict[str, list[BaseEvent]] = {}
    for e in lifecycle:
        if e.session_id not in sessions:
            sessions[e.session_id] = []
        sessions[e.session_id].append(e)
    
    # 1. Check: Did we get multiple sessions?
    session_list = list(sessions.values())
    assert len(session_list) > 1, "Churning user should have multiple sessions"

    # 2. Check: Decay logic
    # The first session should generally be more active than the last session
    first_session_len = len(session_list[0])
    last_session_len = len(session_list[-1])
    
    assert first_session_len >= last_session_len, "Engagement should stay same or decrease"
    
    # 3. Check: Chronological Order
    # The last event of the last session should be much later than the first event
    assert session_list[-1][-1].timestamp > session_list[0][0].timestamp

def test_timestamps_have_realistic_gaps() -> None:
    """Normal users should have seconds between events, not microseconds."""
    user = NormalUser()
    session = user.generate_session()
    
    if len(session) < 2:
        return
    
    for i in range(len(session) - 1):
        gap = (session[i+1].timestamp - session[i].timestamp).total_seconds()
        assert gap >= 5.0, f"Gap too small: {gap}s — events should be 5-30s apart"
        assert gap <= 30.0, f"Gap too large: {gap}s"

def test_timestamps_have_realistic_gaps_bots() -> None:
    """Bots should have microseconds between events"""
    user = BotAgent()
    session = user.generate_session()
    
    if len(session) < 2:
        return
    
    for i in range(len(session) - 1):
        gap = (session[i+1].timestamp - session[i].timestamp).total_seconds()
        assert gap >= 0.5, f"Gap too small: {gap}s — events should be 5-30s apart"
        assert gap <= 1.2, f"Gap too large: {gap}s"