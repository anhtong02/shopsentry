## First approach: Isolation Forest
- Purpose: Used it as baseline, though in this case i have labels, but i want to see what if i dont. 
- The work-through: Since I have 112 bots out of 600 normal users, i set **contamination** to be **0.16**, it gave a strong precision, recall, and f1 for normal but for anomalies, it is bad. (0 is normal, 1 is anomaly)

|   | precision | recall | f1-score | support |   |   |
|---|-----------|--------|----------|---------|---|---|
| 0 | 0.91      | 0.81   | 0.86     | 600     |   |   |
| 1 | 0.37      | 0.58   | 0.45     | 112     |   |   |
|   |           |        |          |         |   |   |
|   |           |        |          |         |   |   |
|   |           |        |          |         |   |   |

- I pushed the contamination to **0.20**: 
|   | precision | recall | f1-score | support |   |   |
|---|-----------|--------|----------|---------|---|---|
| 0 | 0.92      | 0.87   | 0.89     | 600     |   |   |
| 1 | 0.45      | 0.57   | 0.50     | 112     |   |   |
|   |           |        |          |         |   |   |
|   |           |        |          |         |   |   |
|   |           |        |          |         |   |   |
It has some improvement, but you and I both know it's not getting better so lets stop here, lol.

