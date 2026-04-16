#--------PURPOSE--------
# Register the schema contract with the Feast Registry
# This file does not process data. It does not run Spark. 
# It is purely the blueprint that tells Feast: 
#   "These are the columns I promise will exist in Redis, and this is how they are defined."
#-----------------------
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64, Float64

#1. Define user as entity.
#What it does: define primary key for feature store
#join_keys: This is the actual column name in the underlying data table that contains the ID.

session = Entity(name="session_id", 
                 join_keys=["session_id"],
                description="Entity to represent session-level features")

#2 dummy offline source that feast requires for internal logic
#event_timestamp_column: Tells Feast which column contains the time the event actually happened, used for time-travel queries.
stats_source = FileSource(
    path = "data/dummy.parquet",
    event_timestamp_column="event_timestamp",
)

#3 define the feature view (must match columns Spark calculates)
user_activity_v1 = FeatureView(
    name="session_features", #name for this group of features
    entities=[session], #	Tells Feast this data is organized per user. 
    ttl=timedelta(hours=1), #data expires after 5 minutes, critical for REDIS
    schema=[ #schema enforcement
        Field(name='events_per_minute', dtype=Float64), #calculate_events_per_minute
        Field(name='unique_pages_visited', dtype=Float64), #calculate_unique_pages_visited
        Field(name='avg_time_between_events', dtype=Float64), #calculate_avg_time_between_events

        Field(name='cart_to_purchase_ratio', dtype=Float64), #calculate_cart_to_purchase_ratio
        Field(name='session_duration_seconds', dtype=Float64), 
        Field(name='event_type_diversity', dtype=Int64), 

        Field(name='has_payment', dtype=Int64), 
        Field(name='signup_to_purchase_speed', dtype=Float64), 
        Field(name='page_revisit_ratio', dtype=Float64), 

    ],
    online = True,# Enables serving from Redis
    source=stats_source
)
