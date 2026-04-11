from confluent_kafka import Producer
from simulator.event_schema import PageViewEvent, BaseEvent


def create_producer() -> Producer:
    """
    Creates & returns kafka producer pointed at RedPanda
    """
    config = {"bootstrap.servers": "localhost:19092"}
    return Producer(config)


def send_event(producer: Producer, topic: str, event: BaseEvent) -> None:
    """
    Send a single event to a specific topic
    Key = user_id (helps kafka keeps users in order)
    Value = the bytes from the "to_kafka_value" method
    """
    producer.produce(topic=topic, key=event.user_id, value=event.to_kafka_value())
    print(f"Sent {event.event_type} event for user {event.user_id}")


if __name__ == "__main__":
    # set up
    p = create_producer()
    topic_name = "shop-events"

    # create one page view event
    event = PageViewEvent(user_id="u1", session_id="s1")

    # 3. Send it
    send_event(p, topic_name, event)

    # 4. Flush (Don't forget this! It makes sure the message actually leaves the building)
    p.flush()
