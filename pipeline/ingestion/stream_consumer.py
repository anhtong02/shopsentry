#----------------------------
#----------PURPOSE-----------
#What this file does: It listens to Redpanda, 
#collects events [consuming part] into "buckets" (sessions), and once a bucket is full (the user exits), 
#it calculates their behavioral features and saves them to Redis.
#-------------DONE------------
#-------TECHNIQUES USED-------
#In-Memory Buffering: Using a Python dictionary to act as a temporary "waiting room" for events.
# Event-Driven Trigger: Using the "EXIT" pageview as a signal to stop collecting and start calculating.
# JSON Serialization: Converting Python dictionaries into strings to store them in Redis's Key-Value store.

import time
import json
import logging
import redis
from confluent_kafka import Consumer, KafkaError
from typing import Dict, List, Set
from simulator.event_schema import EVENT_MAP, BaseEvent
from pipeline.feature_engine.features import (
    calculate_events_per_minute,
    calculate_unique_pages_visited,
    calculate_avg_time_between_events,
    calculate_cart_to_purchase_ratio
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
KAFKA_CONF = {
    'bootstrap.servers': '127.0.0.1:19092',
    'group.id': 'feature-engine-group',
    'auto.offset.reset': 'earliest'
}

class StreamProcessor:
    
    def __init__(self) -> None:
        self.seen_sessions: Set[str] = set()      # every session we've ever seen
        self.stored_sessions: Set[str] = set()    # every session we've written to Redis
        self.consumer = Consumer(KAFKA_CONF) #Creates a Kafka consumer config
                                             #Why:A consumer can now read from Redpanda topics

        self.consumer.subscribe(['shop-events']) #Tells consumer which topic to read from
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)        # The 'Buffer': session_id -> List[BaseEvent]
        self.buffer: Dict[str, List[BaseEvent]] = {} # Where events hang out until their session is complete
        self.last_seen: Dict[str, float] = {}

    def process_session(self, session_id: str) -> None:
        """ Calculate features and writes to Redis """
        events = self.buffer[session_id]

        #compute features
        features = {
            "events_per_minute": calculate_events_per_minute(events),
            "unique_pages": calculate_unique_pages_visited(events),
            "avg_gap": calculate_avg_time_between_events(events),
            "cart_ratio": calculate_cart_to_purchase_ratio(events),
            "event_count": len(events)
        }

        #store in Redis as JSON
        self.redis_client.set(f"session:{session_id}", json.dumps(features))
        self.stored_sessions.add(session_id)

        logger.info(f"✅ Features stored for session {session_id[:8]}: {features}")

        #clear buffer for memory
        del self.buffer[session_id]

    def run(self) -> None:
        TIMEOUT_SECONDS=40
        logger.info("📡 Pipeline listening for events...")
        try:
            while True:
                msg = self.consumer.poll(1.0) #Ask Redpanda "Any new messages?" Wait up to 1.0 seconds
                                            #Why: Returns None if no message, 
                                            # returns Message object if there is one
                if msg is None: 
                    stale = [sid for sid, t in self.last_seen.items() 
                                        if time.time() - t > TIMEOUT_SECONDS]
                    for sid in stale:
                        if sid in self.buffer:
                            logger.info(f"⏰ Session {sid[:8]} timed out after {TIMEOUT_SECONDS}s")
                            self.process_session(sid)
                        del self.last_seen[sid]
                    continue
                err = msg.error()
                if err is not None:
                    if err.code() == KafkaError._PARTITION_EOF: 
                        continue 
                    else: 
                        logger.error(err)
                        break

                # 1. Deserialize raw bytes back into Pydantic models
                val = msg.value()
                if val is None:
                    continue
                raw_data = json.loads(val.decode('utf-8'))
                event_type = raw_data.get("event_type")
                event_obj = EVENT_MAP[event_type](**raw_data)

                # 2. Add to session buffer
                s_id = event_obj.session_id
                if s_id not in self.buffer:
                    self.buffer[s_id] = []
                self.buffer[s_id].append(event_obj)
                if s_id not in self.seen_sessions:
                    self.seen_sessions.add(s_id)
                    logger.info(f"🆕 New session: {s_id[:8]} (total seen: {len(self.seen_sessions)})")
                self.last_seen[s_id] = time.time()  # ← ADD THIS

                # 3. Check if session is "Done"
                # Logic: If they hit the exit URL or we've seen enough events
                is_exit = hasattr(event_obj, 'url') and "exit" in event_obj.url
                if is_exit or len(self.buffer[s_id]) >= 500:
                    self.process_session(s_id)
                    self.last_seen.pop(s_id, None)
        except KeyboardInterrupt:
            logger.info("🛑 Pipeline shutting down...")
        finally:
            # Flush everything still in the buffer
            for sid in list(self.buffer.keys()):
                self.process_session(sid)
            self.consumer.close()
            missed = self.seen_sessions - self.stored_sessions
            logger.info(f"📊 Seen: {len(self.seen_sessions)}, Stored: {len(self.stored_sessions)}, Never stored: {len(missed)}")
            for sid in missed:
                logger.warning(f"❌ Never stored: {sid}")
if __name__ == "__main__":
    processor = StreamProcessor()
    processor.run()