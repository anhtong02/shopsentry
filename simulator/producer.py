import logging
from confluent_kafka import Producer
from simulator.event_schema import BaseEvent
from typing import Any
logger = logging.getLogger(__name__)

class EventProducer:
    def __init__(self, broker: str = "localhost:19092", topic: str = "shop-events") -> None:
        """
        Initializes the connection to Redpanda/Kafka.
        Note: Using port 19092 as per your previous setup!
        """
        self.topic = topic
        self.producer = Producer({"bootstrap.servers": broker})

    def _delivery_report(self, err: Any, msg: Any) -> None:
        """Called once for each message produced to indicate delivery result."""
        if err is not None:
            logger.error(f"❌ Message delivery failed: {err}")
        else:
            # Success! (Keeping it quiet so we don't spam the terminal)
            pass

    def send_event(self, event: BaseEvent) -> None:
        """
        Sends a single event. 
        Uses user_id as the key to ensure order for that specific user.
        """
        try:
            # Using your existing to_kafka_value() method
            self.producer.produce(
                topic=self.topic,
                key=event.user_id,
                value=event.to_kafka_value(),
                callback=self._delivery_report
            )
            # poll(0) serves delivery reports (callbacks) from previous produce calls
            self.producer.poll(0)
        except BufferError:
            logger.warning("Local producer queue is full, waiting...")
            self.producer.poll(1.0)
            self.send_event(event)

    def flush(self) -> None:
        """Blocks until all messages in the queue have been delivered."""
        logger.info("🧼 Flushing all remaining events to Redpanda...")
        self.producer.flush()