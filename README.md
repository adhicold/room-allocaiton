# RoomFinder - "The Campus Has Space. Students Don't."

Explainable, **conflict-free room recommendation**, timetable clash detection and utilisation
analytics for the Education / Campus Space datathon track.

> Product question: *given a room request, what is the best feasible room and what evidence
> supports that recommendation?*

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"
make test         # 38 tests, ~3 s
make evaluate     # runs every model, writes reports/ (~15 s, deterministic)
make app          # Streamlit UI
python -m roomfinder.cli recommend --date 2025-09-11 --start-hour 16 --capacity 107 --projector
```

The four CSVs live in `data/raw/` (file-name prefixes such as `1792..._rooms.csv` are tolerated).

## How it works

1. **Occupancy model** (`occupancy.py`): one integer counter per room per hour of the term.
   The weekly timetable is projected onto every week; dated events are added on an hourly axis
   (so events that cross midnight work). `0` = free, `>1` = double booking.
2. **Hard constraints** (`matcher.feasible_mask`): capacity, projector, computers, accessibility and
   being free for the *entire* interval (half-open intervals, so back-to-back slots are legal).
3. **Ranking** (`matcher.score_rooms`): weighted sum of capacity fit (`required / capacity`),
   not wasting projector/computer rooms, building preference and accessibility bonus.
   Weights are configurable (`config.Weights`); ties break on `room_id` => fully deterministic.
4. **Explanations**: every recommendation lists the evidence ("Fits 107 of 125 seats", "Projector
   available", "Free 16:00-17:00 on Thu 11 Sep, no clash").
5. **Fallback** (`suggest_alternatives`): nearest start times that work when nothing is feasible.
6. **Admin analytics** (`utilization.py`): utilisation heatmap, K-Means room clusters, and a greedy
   timetable repair that never creates a new clash and always seats the class.

## Evaluation (all numbers produced by `roomfinder evaluate`, see `reports/REPORT.md`)

The requests file has **no ground-truth room**, so allocation quality is measured by simulation:
requests are replayed in chronological order and each placement blocks its room.

| policy | placed | unmet (Waitlisted/Rejected) placed | mean wasted seats |
|---|---|---|---|
| first-fit | 96.8 % | 1,513 / 1,559 | 51.6 |
| random | 96.8 % | 1,515 / 1,559 | 60.2 |
| **best-fit (ours)** | **97.4 %** | **1,518 / 1,559** | **14.8** (-71 %) |

Data issues found and handled: 3,426 overlapping class pairs (3,630 rows), 24 event-event and
1,337 class-event overlaps, 40 % of classes larger than their room, 49 % of events outside
08:00-20:00, expected vs actual event attendance uncorrelated (r = 0.00, so never used as truth).
Greedy timetable repair cuts double-booked slots 1,990 -> 825 and over-crowded classes
2,009 -> 1,236.

### Learned models, reported honestly

| model | result | verdict |
|---|---|---|
| Poisson GLM, hourly demand (train Aug-Oct, test Nov) | MAE 1.401 vs 1.399 for a global-mean baseline | no gain |
| Logistic / boosted trees, unmet vs approved (chronological split) | ROC-AUC 0.51 / 0.49 | chance level |

Request attributes carry no usable signal in this synthetic data, so the project does **not**
claim predictive power it does not have. The decision value comes from the exact constraint-based
allocator; the learned models stay as audited, baseline-compared experiments
(`forecast.py`, `risk_model.py`) with chronological splits and no label leakage.

## Assumptions and limitations

- The weekly timetable repeats every week of the term (no holidays given).
- `request_date` is treated as the date of use.
- Cancelled requests are excluded from demand; Waitlisted/Rejected count as unmet demand.
- Timetable repair ignores dated events and facilities (the schedule has no facility needs).
- The data is synthetic; absolute numbers will differ on real campus data.

## Project layout

```
src/roomfinder/   data.py occupancy.py matcher.py allocation.py utilization.py
                  forecast.py risk_model.py evaluate.py cli.py config.py
app/              streamlit_app.py (Find a room / Admin dashboard / Evaluation)
tests/            38 tests: unit, regression on the real CSVs, Streamlit smoke tests
reports/          generated metrics.json, REPORT.md and CSV outputs
.github/          CI: ruff + pytest on every push
```
