# @claude-code-skills/stats

MIT-level statistical analysis library with t-tests, regression, and effect size calculations.

## Features

- **T-tests**: Paired and independent samples t-tests
- **Effect Size**: Cohen's d calculation and interpretation
- **Regression**: Linear and log-log regression for complexity validation
- **Utilities**: Mean, SD, confidence intervals, outlier detection
- **Experiment Framework**: Complete before-after analysis

## Installation

```bash
npm install @claude-code-skills/stats
```

## Quick Start

```typescript
import { pairedTTest, runBeforeAfterExperiment } from '@claude-code-skills/stats';

// Simple t-test
const before = [12.5, 13.2, 11.8, 14.1, 12.9];
const after = [4.8, 5.2, 4.5, 5.5, 4.9];
const result = pairedTTest(before, after);

console.log(`p-value: ${result.p < 0.001 ? '<0.001' : result.p.toFixed(3)}`);
console.log(`Cohen's d: ${result.d.toFixed(2)}`);
console.log(`95% CI: [${result.ci[0].toFixed(2)}, ${result.ci[1].toFixed(2)}]`);

// Complete experiment with formatted output
const experiment = runBeforeAfterExperiment("Rendering time (ms)", before, after);
console.log(`Improvement: ${experiment.improvement.toFixed(1)}%`);
```

## API Documentation

See full TypeDoc documentation in `/docs/stats/`.

### T-Tests

#### `pairedTTest(before, after)`

Performs a paired t-test for dependent samples.

**Parameters:**
- `before: number[]` - Measurements before treatment
- `after: number[]` - Measurements after treatment

**Returns:** `TTestResult`
- `t`: t-statistic
- `df`: degrees of freedom
- `p`: two-tailed p-value
- `ci`: 95% confidence interval
- `d`: Cohen's d effect size

#### `independentTTest(group1, group2)`

Performs an independent samples t-test.

### Regression

#### `linearRegression(x, y)`

Simple linear regression: `y = slope * x + intercept`

**Returns:** `RegressionResult`
- `slope`: regression slope
- `intercept`: y-intercept
- `r2`: coefficient of determination
- `residuals`: array of residuals

#### `logLogRegression(n, time)`

Log-log regression for complexity validation.

**Example:**
```typescript
const n = [100, 200, 500, 1000];
const time = [2.1, 4.5, 12.3, 26.8];
const result = logLogRegression(n, time);
console.log(`Complexity: O(n^${result.slope.toFixed(2)})`);
console.log(`R² = ${result.r2.toFixed(4)}`);
```

### Utilities

- `mean(arr)` - Arithmetic mean
- `standardDeviation(arr)` - Sample standard deviation
- `confidenceInterval(arr, confidence=0.95)` - Confidence interval
- `median(arr)` - Median value
- `quartiles(arr)` - [Q1, Q2, Q3]
- `detectOutliers(arr, multiplier=1.5)` - IQR-based outlier detection

## Statistical Standards

This library follows rigorous statistical practices:

- **Sample size**: n ≥ 30 recommended
- **Tests**: Two-tailed by default
- **Variance**: Bessel's correction (n-1)
- **Confidence**: 95% intervals
- **Effect size**: Cohen's d interpretation

### Cohen's d Interpretation

- |d| < 0.2: negligible effect
- 0.2 ≤ |d| < 0.5: small effect
- 0.5 ≤ |d| < 0.8: medium effect
- 0.8 ≤ |d| < 1.2: large effect
- |d| ≥ 1.2: very large effect

## License

MIT
