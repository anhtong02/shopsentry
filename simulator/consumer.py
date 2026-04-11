from confluent_kafka import Consumer, KafkaError
from simulator.event_schema import PageViewEvent


def create_consumer() -> Consumer:
    """
    Creates and returns a Kafka consumer.

    """
    config = {
        "bootstrap.servers": "localhost:19092",
        "group.id": "shop-monitor-group",  # The "Work Group" name
        "auto.offset.reset": "earliest",  # Start from the beginning of the list
    }
    return Consumer(config)


def consume_events(topic: str) -> None:
    consumer = create_consumer()
    consumer.subscribe([topic])
    print(f"Listening for events on topic: {topic}...")

    try:
        while True:
            # Wait up to 1 second for a new message
            msg = consumer.poll(1.0)

            if msg is None:
                continue
            err = msg.error()
            if err:
                if err.code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Consumer error: {msg.error()}")
                    break

            # 1. Get the bytes from Kafka
            raw_data = msg.value()
            if raw_data is None:
                continue
            # 2. Use your schema logic to turn bytes back into an Object
            # For now, we know it's a PageViewEvent.
            # Later, we can make this smarter to handle any event type!
            event = PageViewEvent.from_kafka_value(raw_data)

            if isinstance(event, PageViewEvent):
                print(f"RECEIVED: {event.event_type} | User: {event.user_id} | URL: {event.url}")
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up and close the connection
        consumer.close()


if __name__ == "__main__":
    consume_events("shop-events")
