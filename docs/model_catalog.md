# Phase 0: Proving Architecture (noiseless baseline)
## First approach: Isolation Forest
- Purpose: Used it as baseline, though in this case i have labels, but i want to see what if i dont. 
- The work-through: Since I have 112 bots out of 600 normal users, i set **contamination** to be **0.16**, it gave a strong precision, recall, and f1 for normal but for anomalies, it is bad. (0 is normal, 1 is anomaly)

|   | precision | recall | f1-score | support |   |   |
|---|-----------|--------|----------|---------|---|---|
| 0 | 0.91      | 0.81   | 0.86     | 600     |   |   |
| 1 | 0.37      | 0.58   | 0.45     | 112     |   |   |
|   |           |        |          |         |   |   |


- I pushed the contamination to **0.20**: 

|   | precision | recall | f1-score | support |   |   |
|---|-----------|--------|----------|---------|---|---|
| 0 | 0.92      | 0.87   | 0.89     | 600     |   |   |
| 1 | 0.45      | 0.57   | 0.50     | 112     |   |   |
|   |           |        |          |         |   |   |

It has some improvement, but you and I both know it's not getting better so lets stop here, lol. Primarily because Iso Forest can't learn complex boundaries. But it's fun to explore different models.
- Based on the recall of each agents:

   | fraud recall | bot recall | 
   |--------------|------------|
   | 0.24         | 0.98       | 

bot recall is good, catching all of them, mainly because bot's activities are too easy to be outliers. But a fraud ring spawns 5-8 agents that all hit the exact same product id, follow same routine (landing-view-add-checkout-payment) at same time, thus in feature space, it looks like a dense cluster. So to iso forest, it looks dense thus it must be normal, and then ignores them.

## Second approach: Autoencoder

|   | precision | recall | f1-score | support |   |
|---|-----------|--------|----------|---------|---|
| 0 | 1.00      | 0.93   | 0.97     | 120     |   |
| 1 | 0.93      | 1.00   | 0.97     | 112     |   |
|   |           |        |          |         |   |

Very good, but when found recall is too good to be true, i checked for fraud and bot recall:
   | fraud recall | bot recall | 
   |--------------|------------|
   | 1.0          | 1.0        | 

I splitted the data (80% train, 20% test), only trained on normal data (no anomalies), used standard scaler, I made sure there wasn't any leakage. But the reason why it's exactly 1 is because the data right now is too perfect, no noise in normal, everyone behaves the same so it's very easy to detect anomalies. Which is ok because this phase is to make everything works smoothly first.

