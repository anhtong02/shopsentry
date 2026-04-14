#----------PURPOSE-----------
#Acts as producer, take the events (such as clicks, page views, purchases,etc)
#and push them into Redpanda.
#----------------------------
#---------TECHNIQUES---------
#Its uses asynchronous micro batch. 
#The producer does not send data immediately, instead it puts the event
#into a local memory queue and starts 50ms later (hence linger.ms =50)
#Once 50ms hit or queue reaches 10,000 events, it compresses the data
#into much smaller file and sned it. Since the size is way smaller,
#it sends very quickly to red panda. This helps achieve 10,000 events/sec
#


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
        # PERFORMANCE TUNING: Enable batching and compression
        self.producer = Producer({
            "bootstrap.servers": broker,
            "linger.ms": 50,             # Wait up to 50ms to group messages into a batch
            "batch.num.messages": 10000, # Or until we hit 10k messages
            "compression.type": "lz4",   # Crucial for high throughput
            "queue.buffering.max.messages": 100000 # Larger internal queue
        })

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
            self.producer.produce(
                topic=self.topic,
                key=event.user_id,
                value=event.to_kafka_value(),
                callback=self._delivery_report
            )
            # REMOVED: self.producer.poll(0) from here!
        except BufferError:
            logger.warning("Local producer queue is full, waiting...")
            # If the queue actually fills up, 
            # wait a full second to let it drain
            self.producer.poll(1.0)
            self.send_event(event)

    def poll(self) -> None:
        """Expose poll so we can call it periodically in the main loop"""
        self.producer.poll(0)

    def flush(self) -> None:
        """Blocks until all messages in the queue have been delivered."""
        logger.info("🧼 Flushing all remaining events to Redpanda...")
        self.producer.flush()