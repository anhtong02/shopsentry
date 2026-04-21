## events_per_minute
- Hypothesis: Bots generate events far faster than humans can click
- Computation: event_count / (session_duration_seconds / 60)
- Normal range: 2-15
- Suspicious: >50 (likely bot), >200 (definite bot)

## unique_pages_visited
- Hypothesis: Scrapers visit many but in a sequential pattern. Spammers visit almost none
- Computations: count disinct visited url per session
- Normal range: less than 50
- Suspicious: >100

## avg_time_between_events
- Hypothesis: shorter gap for bots
- Computations: total sec of a session / (count(event) - 1 +0.001)
- Normal range: 5-10s
- Suspicious < 1s

## cart_to_purchase_ratio
- Hypothesis: normal users sometimes purchase, bot never purchases, fraud rings purchases a lot
- Computation: succesful payments / total addToCarts
- Normal range:  0.1-0.3
- Suspicious: 1.0 

## session_duration_seconds
- Hypothesis: longer session = more engaged users, bots often have very short or unusually long session
- Computation: timestamp of last event minus timestamp of first event
- Normal range: minutes
- Suspicious: hours or 10-20 seconds

## event_type_diversity
- Hypothesis: Bots do one thing repeatedly (low diversity), humans do many things (high diversity)
- Computation: find all unique event types per session
- Normal range: > 4
- Suspicious: 1-2

## has_payment
- Hypothesis: bots never pay, fraud rings always pay, normal sometimes pay
- Comupation: check the payment event and status == "success"
- Normal range: mixed 0s and 1s across sessions
- Suspicious:  when combined with other signals (fast session + has_payment = fraud).

## sign_up_to_purchase
- Hypothesis: fraud rings sign up and buy within seconds using stolen cards before get detected
- Computation: time of purchase minus time of sign up
- Normal range: hours or none
- Suspicious: seconds.





