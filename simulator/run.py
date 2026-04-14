import logging
import random
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any

from simulator.producer import EventProducer
from simulator.agents.normal_user import NormalUser
from simulator.agents.bot_agent import BotAgent
from simulator.agents.fraud_ring import FraudRing
from simulator.agents.churning_user import ChurningUser
from simulator.event_schema import BaseEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_scenario(scenario_name: str) -> Dict[str, Any]:
    """Loads the requested scenario from the YAML config."""
    config_path = Path(__file__).parent / "scenarios.yaml"
    
    with open(config_path, 'r') as file:
        scenarios: Dict[str, Any] = yaml.safe_load(file)
        
    if scenario_name not in scenarios:
        raise ValueError(f"❌ Scenario '{scenario_name}' not found in scenarios.yaml")
        
    # We wrap it in dict() to guarantee to Mypy that it's returning a dictionary
    return dict(scenarios[scenario_name])

def run_simulation(scenario_name: str) -> None:
    producer = EventProducer()
    all_events: List[BaseEvent] = []
    
    # 1. Load the traffic recipe
    logger.info(f"📜 Loading scenario: {scenario_name.upper()}")
    config = load_scenario(scenario_name)

    logger.info("🎨 Generating agent populations based on config...")

    # 2. Dynamically spawn agents based on YAML counts
    for _ in range(config.get('normal_users', 0)):
        all_events.extend(NormalUser().generate_session())

    for _ in range(config.get('bots', 0)):
        all_events.extend(BotAgent().generate_session())

    for _ in range(config.get('fraud_rings', 0)):
        ring = FraudRing(size=random.randint(5, 8))
        all_events.extend(ring.generate_all_events())
        
    for _ in range(config.get('churning_users', 0)):
        all_events.extend(ChurningUser().generate_lifecycle())

    logger.info(f"📊 Total events generated: {len(all_events)}")

    # 3. Interleave and Send
    logger.info("⏳ Interleaving events by timestamp...")
    all_events.sort(key=lambda x: x.timestamp)

    logger.info("🚀 Pumping stream to Redpanda...")
    for i, event in enumerate(all_events):
        producer.send_event(event)
        if i % 10000 == 0:
            producer.poll()
            logger.info(f"Sent {i} events...")

    producer.flush()
    logger.info("✅ Simulation complete! Check your Redpanda console.")

if __name__ == "__main__":
    # Set up the command line argument parser
    parser = argparse.ArgumentParser(description="StreamGuard Traffic Simulator")
    parser.add_argument(
        "--scenario", 
        type=str, 
        default="normal_day", 
        help="Which traffic scenario to run (e.g., normal_day, bot_attack, flash_sale)"
    )
    
    args = parser.parse_args()
    run_simulation(args.scenario)