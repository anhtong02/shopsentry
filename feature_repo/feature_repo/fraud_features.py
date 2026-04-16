#--------PURPOSE--------
# Register the schema contract with the Feast Registry
# This file does not process data. It does not run Spark. 
# It is purely the blueprint that tells Feast: 
#   "These are the columns I promise will exist in Redis, and this is how they are defined."
#-----------------------
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64

#1. Define user as entity.
#What it does: define primary key for feature store
#join_keys: This is the actual column name in the underlying data table that contains the ID.

user = Entity(name="user_id", join_keys=["user_id"])

#2 dummy offline source that feast requires for internal logic
#event_timestamp_column: Tells Feast which column contains the time the event actually happened, used for time-travel queries.
stats_source = FileSource(
    path = "data/dummy.parquet",
    event_timestamp_column="event_timestamp",
)

#3 define the feature view (must match columns Spark calculates)
user_activity_v1 = FeatureView(
    name="user_activity", #name for this group of features
    entities=[user], #	Tells Feast this data is organized per user. 
    ttl=timedelta(minutes=5), #data expires after 5 minutes, critical for REDIS
    schema=[ #schema enforcement
        Field(name='click_velocity', dtype=Int64),
        Field(name='ip_entropy', dtype=Int64), 
    ],
    online = True,# Enables serving from Redis
    source=stats_source
)
