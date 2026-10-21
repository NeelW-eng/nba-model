# Tip-Off Edge

NBA player-prop model for Underdog pick'em. Each day it scores every Underdog NBA line, posts a Top 10
(max 2 per game, 1 per player) and the 3 best picks in every game, and grades every pick from box scores.

## What runs, and when (Central time)

| When | Mode | What |
|---|---|---|
| ~9:45 AM | `am` | Retrain models, grade last night, pull lines + injuries, post the day's picks |
| ~1 hr before first tip | `pretip` | Rescore with the latest injury report; picks for games not yet started can change |
| 11:47 AM | backup | Runs `am` only if no board was made in the last 6 hours |

Claude scheduled tasks start each run by pushing `requests/run`, wait for it, then push `data/board.json`
to the dashboard's database. Manual run: **Actions → Tip-Off Edge run → Run workflow**.

No API keys or secrets are needed.

## Data (all free)

- **History:** ESPN box scores 2021-22 to 2025-26 via sportsdataverse (`data/hist`).
- **This season:** ESPN scoreboard + game summaries (`data/box_current.parquet`).
- **Injuries:** ESPN NBA injury feed. Out = removed and their production shifted to teammates; Day-To-Day = Questionable.
- **Lines:** Underdog's public pick'em feed, including each side's price and payout multiplier.

## How picks are chosen

1. **Projection.** One LightGBM model per stat (points, rebounds, assists, 3PM, steals, blocks, turnovers,
   the combos, fantasy points, double/triple-doubles) using recent form, per-minute rates, minutes trend,
   starter role, pace, opponent defense vs the position, rest, blowout risk and teammates out. Each becomes a
   negative-binomial outcome distribution, so it gives a probability at any line.
2. **Win chance.** Underdog's own two-sided price is de-vigged, then moved toward the model by up to 40%
   (less when they disagree a lot).
3. **Value** = win chance × payout multiplier. 3-pick standard breaks even at ~55% per pick, 4-pick ~56.2%.
4. Only lines priced on both sides at 0.6x–1.6x multipliers are pickable; the model must favour the side.

Backtest (`pipeline/backtest.py`): trained 2021-24, tuned 2024-25, tested on all of 2025-26 against lines
rebuilt from pre-game averages. The model's error is 2-15% lower than those lines on every stat and its
probabilities are calibrated. Real Underdog lines are sharper, so live results will be lower.

Every pick is logged in `data/picks_log.csv` and graded after the game (DNP = void).
