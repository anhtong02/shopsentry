import logging
import random
from typing import List
from simulator.producer import EventProducer
from simulator.agents.normal_user import NormalUser
from simulator.agents.bot_agent import BotAgent
from simulator.agents.fraud_ring import FraudRing
from simulator.agents.churning_user import ChurningUser
from simulator.event_schema import BaseEvent

# Setup logging to see the progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_simulation() -> None:
    producer = EventProducer()
    all_events: List[BaseEvent] = []

    logger.info("🎨 Generating agent populations...")

    # 1. Create 100 Normal Users
    for _ in range(100):
        user = NormalUser()
        all_events.extend(user.generate_session())

    # 2. Create 10 Bot Agents
    for _ in range(10):
        bot = BotAgent()
        all_events.extend(bot.generate_session())

    # 3. Create 2 Fraud Rings (approx 10-16 attackers total)
    for _ in range(2):
        ring = FraudRing(size=random.randint(5, 8))
        all_events.extend(ring.generate_all_events())
        
    # 4. Create 5 Churning Users (Life stories)
    for _ in range(5):
        churner = ChurningUser()
        all_events.extend(churner.generate_lifecycle())

    logger.info(f"📊 Total events generated: {len(all_events)}")

    # 5. THE MAGIC STEP: Interleaving
    # We sort the entire list so that events from different users 
    # are mixed together chronologically.
    logger.info("⏳ Interleaving events by timestamp...")
    all_events.sort(key=lambda x: x.timestamp)

    # 6. Send to Kafka
    logger.info("🚀 Pumping stream to Redpanda...")
    for i, event in enumerate(all_events):
        producer.send_event(event)
        if i % 100 == 0:
            logger.info(f"Sent {i} events...")

    producer.flush()
    logger.info("✅ Simulation complete! Check your Redpanda console.")
    unique_sessions = {e.session_id for e in all_events}
    logger.info(f"📋 Unique sessions: {len(unique_sessions)}")
if __name__ == "__main__":
    run_simulation()