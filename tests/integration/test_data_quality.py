import pandas as pd
import great_expectations as gx

print("🔍 Booting up Great Expectations Health Inspector...")

# 1. Load the data using pure Pandas
df = pd.read_parquet("feature_repo/feature_repo/data/offline_features")
print(f"📊 Loaded {len(df)} rows from the Offline Store.")

# 2. Create a temporary, lightweight GX Context
context = gx.get_context(mode="ephemeral")

# 3. Connect Pandas to GX
data_source = context.data_sources.add_pandas("offline_store")
data_asset = data_source.add_dataframe_asset("user_activity_features")
batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_def")

# 4. Define our Rules (Expectations)
suite = context.suites.add(gx.ExpectationSuite(name="fraud_rules"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="session_id"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="events_per_minute", min_value=0, max_value=10000))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="unique_pages_visited", min_value=0, max_value=500))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="avg_time_between_events", min_value=0, max_value=3600))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="cart_to_purchase_ratio", min_value=0, max_value=1.001))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="session_duration_seconds", min_value=0, max_value=86400))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="event_type_diversity", min_value=1, max_value=8))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="has_payment", min_value=0, max_value=1))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="signup_to_purchase_speed", min_value=0, max_value=86400))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="page_revisit_ratio", min_value=0, max_value=1))

# 5. Run the Inspection!
validation_definition = context.validation_definitions.add(
    gx.ValidationDefinition(name="offline_validation", data=batch_definition, suite=suite)
)
results = validation_definition.run(batch_parameters={"dataframe": df})

# 6. Print the Scorecard
print("\n" + "="*50)
print("🛡️  DATA QUALITY SCORECARD")
print("="*50)
print(f"Overall Success: {'✅ PASSED' if results.success else '❌ FAILED'}")
print("-" * 50)
for result in results.results:
    rule_name = result.expectation_config.type if result.expectation_config else "Unknown Rule"    
    status = '✅' if result.success else '❌'
    print(f"{status} {rule_name}")
    print("="*50)