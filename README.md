# Financial RankNet
Pairwise ranking the stocks by predicting the future performance, the performance is determined by various forecasting indicators, such as return and volatility.
This repository implements RankNet which can generate the leaderboard according to the given question.

## Concept
Given stocks `[Si, Sj]` with timestep `[t-n, t-n+1, ..., t-1, t]` and the corresponding indicator `P` when `t+1`, to predict whether `P(Si) > P(Sj)` or not.

- For example, assume that `P` rerpresents the return of stock when `t+1`, the model is supposed to predict whether the return of `Si` is higher than `Sj` when `t+1`.

Furthermore, the model is supposed to predict the rank according to different `P`.
- For example, assume that `P` represents the volatility of stock when `t+1` then, the previous model is also capable of predicting whether the volatility of `Si` is higher than `Sj` when `t+1`.

**Note: this experiment is only an early prototype, the concept and architecture are still under developing.**
