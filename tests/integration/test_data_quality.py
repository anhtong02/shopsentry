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
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="click_velocity", min_value=1, max_value=5000))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="ip_entropy", min_value=1, max_value=100))

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