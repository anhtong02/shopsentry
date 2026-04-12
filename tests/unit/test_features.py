from simulator.agents.normal_user import NormalUser
from simulator.agents.bot_agent import BotAgent
from pipeline.feature_engine.features import (
    calculate_events_per_minute,
    calculate_avg_time_between_events,
    calculate_cart_to_purchase_ratio
)

def test_feature_discrimination_bot_vs_human() -> None:
    """
    Purpose: Ensure bots and humans produce statistically different features.
    """
    human_session = NormalUser().generate_session()
    bot_session = BotAgent().generate_session()
    
    # Extract features
    human_epm = calculate_events_per_minute(human_session)
    bot_epm = calculate_events_per_minute(bot_session)
    
    # Assertions
    # In a perfect loop, the bot has many more events in the same time window
    assert bot_epm > human_epm, f"Bot EPM ({bot_epm}) should be higher than Human EPM ({human_epm})"

def test_empty_session_safety() -> None:
    """
    Purpose: Ensure functions don't crash (e.g. DivisionByZero) on empty data.
    """
    assert calculate_events_per_minute([]) == 0.0
    assert calculate_avg_time_between_events([]) == 0.0
    assert calculate_cart_to_purchase_ratio([]) == 0.0