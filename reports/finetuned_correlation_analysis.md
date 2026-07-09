## What was tested

- Whether earnings call tone (CEO/CFO sentiment, scored by a fine-tuned AI model) predicts how a stock moves the next day
  - Specifically the company-specific move, after stripping out broad market swings (1-day abnormal return, market-adjusted using SPY)
- Sample: 13,611 calls total
  - CEO and CFO sentiment averaged into one score per call, since they share the same return
  - About 12% of calls have only one speaker; those just use that speaker's score directly
  - 13,429 to 13,507 calls have valid return data, depending on which window is being measured

## What was found

- Positive-sounding calls tend to see somewhat better next-day stock performance
- Negative-sounding calls tend to see somewhat worse
- The effect is modest but real
  - Correlation (Spearman rho, a score from -1 to +1 measuring how consistently two things move together) = +0.10
  - 95% confidence interval (the likely range for the true value) of [+0.087, +0.120]
  - Confidence interval sits fully above zero, meaning the relationship is very unlikely to be chance
  - Minimum detectable effect (the smallest rho this sample size could reliably tell apart from zero) is about ±0.009, so a rho of +0.10 is well above the noise floor
- Holds up across conditions
  - 1-day window: rho = +0.10 (n = 13,429)
  - 5-day window: rho = +0.08 to +0.09 (n = 13,507)
  - Same pattern with and without market adjustment
  - All four variants statistically significant even after Benjamini-Hochberg correction (a check that raises the bar for significance when testing several related things at once, so we don't get a false positive by chance)

## Why I trust it

- Tested at the company level, not the call level
  - Used a ticker-clustered bootstrap (a resampling technique that treats each company, not each call, as one independent data point, since a company's calls tend to move together) with 10,000 resamples
- Ran a control using an untrained version of the model
  - Untrained model: rho = -0.002, 95% CI [-0.019, +0.015], essentially zero, as expected
  - Difference between trained and untrained: +0.106, 95% CI [+0.082, +0.129], p < 0.001 (p-value here is the probability the difference could show up by chance alone; under 0.001 means very unlikely)
  - Trained model finding a signal where the untrained one finds none is strong evidence the model is picking up something real, not an artifact of the return data

## What it means

- Call sentiment carries genuine predictive signal about post-earnings stock reaction
  - Small in size (rho around 0.10), but statistically reliable and above the noise floor for this sample size
- Best used as one input among many, not a standalone trading signal

## Caveats

- Market adjustment is simplified (assumes beta = 1, meaning we assume the stock moves exactly 1-for-1 with the market rather than fitting each company's actual sensitivity), not company-specific
- Asymptotic p-values (the standard textbook version of this significance test) ignore clustering and would overstate confidence on their own; the clustered bootstrap CI is the trustworthy number here
- This is the aggregate picture only
  - Not yet broken down by speaker (CEO vs. CFO) or subgroup
  - That breakdown is the next step
- A few hundred calls are missing return data and were dropped from the relevant comparisons