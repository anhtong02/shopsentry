import pytest
from datetime import timezone
from pydantic import ValidationError
from simulator.event_schema import EventType, PageViewEvent, SearchEvent, AddToCartEvent


def test_pageview_creation_minimal() -> None:
    # could i create a page view event with just user id and session id?
    event = PageViewEvent(user_id="u1", session_id="s1")
    assert event.user_id == "u1"
    assert event.session_id == "s1"
    assert event.event_type == EventType.PAGEVIEW


def test_event_id_auto_generates() -> None:
    """Does event_id auto-generate and stay unique?"""
    event1 = PageViewEvent(user_id="u1", session_id="s1")
    event2 = PageViewEvent(user_id="u1", session_id="s1")

    assert event1.event_id is not None
    assert event2.event_id is not None
    assert event1.event_id != event2.event_id


def test_timestamp_is_utc() -> None:
    """Does timestamp auto-set to UTC?"""
    event = PageViewEvent(user_id="u1", session_id="s1")
    assert event.timestamp.tzinfo == timezone.utc


def test_fails_without_user_id() -> None:
    """Does it fail if you leave out user_id?"""
    with pytest.raises(ValidationError):
        # We add '# type: ignore[call-arg]' to tell Mypy:
        # "I know what I'm doing, stop complaining about the missing argument here."
        PageViewEvent(session_id="s1")  # type: ignore[call-arg]


def test_kafka_serialization_roundtrip() -> None:
    """Can you roundtrip an event through to_kafka_value() and from_kafka_value()?"""
    original_event = PageViewEvent(
        user_id="u1", session_id="s1", url="https://test.com", properties={"source": "email"}
    )

    # Pack it into bytes
    kafka_bytes = original_event.to_kafka_value()
    assert isinstance(kafka_bytes, bytes)

    # Unpack it back into an object
    # Note: We call it on the specific class to ensure all fields are recovered
    recovered_event = PageViewEvent.from_kafka_value(kafka_bytes)
    assert isinstance(recovered_event, PageViewEvent)  # This "proves" it to Mypy
    assert recovered_event.event_id == original_event.event_id
    assert recovered_event.user_id == original_event.user_id
    assert recovered_event.url == original_event.url
    assert recovered_event.properties == {"source": "email"}


def test_subclass_event_types() -> None:
    """Does each event subclass have the correct event_type?"""
    pv = PageViewEvent(user_id="u1", session_id="s1")
    se = SearchEvent(user_id="u1", session_id="s1", query="shoes")
    atc = AddToCartEvent(user_id="u1", session_id="s1", product_id="p123", quantity=1)

    assert pv.event_type == EventType.PAGEVIEW
    assert se.event_type == EventType.SEARCH
    assert atc.event_type == EventType.ADD_TO_CART
