/**
 * T-test implementations for hypothesis testing
 * @packageDocumentation
 */

import { tCDF, tInv } from './distributions.js';
import type { TTestResult } from './types.js';

/**
 * Performs a paired t-test (dependent samples)
 * @param before - Measurements before treatment
 * @param after - Measurements after treatment
 * @returns T-test results including t-statistic, p-value, Cohen's d, and 95% CI
 * @throws {Error} If arrays have different lengths
 * @throws {RangeError} If arrays have fewer than 2 elements
 * @public
 *
 * @remarks
 * The paired t-test compares the means of two related samples (e.g., before/after measurements).
 * It tests the null hypothesis that the mean difference is zero.
 *
 * Requirements:
 * - Paired observations (same subjects measured twice)
 * - Differences are approximately normally distributed
 * - n ≥ 2 (preferably n ≥ 30 for reliable results)
 *
 * @example
 * ```typescript
 * const before = [12.5, 13.2, 11.8, 14.1, 12.9];
 * const after = [4.8, 5.2, 4.5, 5.5, 4.9];
 * const result = pairedTTest(before, after);
 * console.log(`p-value: ${result.p < 0.001 ? '<0.001' : result.p.toFixed(3)}`);
 * console.log(`Cohen's d: ${result.d.toFixed(2)}`);
 * ```
 */
export function pairedTTest(before: number[], after: number[]): TTestResult {
  if (before.length !== after.length) {
    throw new Error('Arrays must have the same length');
  }
  if (before.length < 2) {
    throw new RangeError('Sample size must be at least 2');
  }

  const n = before.length;
  const diff = before.map((b, i) => b - after[i]);

  // Mean difference
  const meanDiff = diff.reduce((a, b) => a + b, 0) / n;

  // Standard deviation
  const variance = diff.reduce((sum, d) => sum + Math.pow(d - meanDiff, 2), 0) / (n - 1);
  const sd = Math.sqrt(variance);

  // Handle case where sd = 0
  if (sd === 0) {
    return {
      t: meanDiff === 0 ? 0 : Infinity,
      df: n - 1,
      p: meanDiff === 0 ? 1 : 0,
      ci: [meanDiff, meanDiff],
      meanDiff,
      sd,
      d: 0
    };
  }

  // T-statistic
  const t = meanDiff / (sd / Math.sqrt(n));

  // Degrees of freedom
  const df = n - 1;

  // P-value (two-tailed)
  const p = 2 * (1 - tCDF(Math.abs(t), df));

  // 95% confidence interval
  const tCrit = tInv(0.025, df);  // Two-tailed, α=0.05
  const margin = tCrit * (sd / Math.sqrt(n));
  const ci: [number, number] = [meanDiff - margin, meanDiff + margin];

  // Cohen's d
  const d = meanDiff / sd;

  return { t, df, p, ci, meanDiff, sd, d };
}

/**
 * Performs an independent samples t-test (two-sample t-test)
 * @param group1 - First group measurements
 * @param group2 - Second group measurements
 * @returns T-test results including t-statistic, p-value, Cohen's d, and 95% CI
 * @throws {RangeError} If either group has fewer than 2 elements
 * @public
 *
 * @remarks
 * The independent t-test compares the means of two unrelated samples.
 * It tests the null hypothesis that the two population means are equal.
 *
 * This implementation uses pooled variance (assumes equal variances).
 *
 * Requirements:
 * - Independent observations
 * - Both groups approximately normally distributed
 * - Equal variances (homoscedasticity)
 * - n ≥ 2 for each group (preferably n ≥ 30)
 *
 * @example
 * ```typescript
 * const control = [10.2, 11.5, 9.8, 10.9, 11.2];
 * const treatment = [8.1, 7.9, 8.5, 8.2, 7.8];
 * const result = independentTTest(control, treatment);
 * console.log(`Mean difference: ${result.meanDiff.toFixed(2)}`);
 * console.log(`95% CI: [${result.ci[0].toFixed(2)}, ${result.ci[1].toFixed(2)}]`);
 * ```
 */
export function independentTTest(group1: number[], group2: number[]): TTestResult {
  if (group1.length < 2 || group2.length < 2) {
    throw new RangeError('Each group must have at least 2 observations');
  }

  const n1 = group1.length;
  const n2 = group2.length;

  const mean1 = group1.reduce((a, b) => a + b, 0) / n1;
  const mean2 = group2.reduce((a, b) => a + b, 0) / n2;
  const meanDiff = mean1 - mean2;

  const var1 = group1.reduce((sum, x) => sum + Math.pow(x - mean1, 2), 0) / (n1 - 1);
  const var2 = group2.reduce((sum, x) => sum + Math.pow(x - mean2, 2), 0) / (n2 - 1);

  // Pooled variance
  const pooledVar = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
  const pooledSd = Math.sqrt(pooledVar);

  // Handle case where pooled sd = 0
  if (pooledSd === 0) {
    return {
      t: meanDiff === 0 ? 0 : Infinity,
      df: n1 + n2 - 2,
      p: meanDiff === 0 ? 1 : 0,
      ci: [meanDiff, meanDiff],
      meanDiff,
      sd: pooledSd,
      d: 0
    };
  }

  // T-statistic
  const t = meanDiff / (pooledSd * Math.sqrt(1 / n1 + 1 / n2));

  // Degrees of freedom
  const df = n1 + n2 - 2;

  // P-value (two-tailed)
  const p = 2 * (1 - tCDF(Math.abs(t), df));

  // 95% confidence interval
  const tCrit = tInv(0.025, df);
  const margin = tCrit * pooledSd * Math.sqrt(1 / n1 + 1 / n2);
  const ci: [number, number] = [meanDiff - margin, meanDiff + margin];

  // Cohen's d
  const d = meanDiff / pooledSd;

  return { t, df, p, ci, meanDiff, sd: pooledSd, d };
}

/**
 * Interprets Cohen's d effect size
 * @param d - Cohen's d value
 * @returns Interpretation string (English)
 * @public
 *
 * @remarks
 * Based on Cohen's (1988) guidelines:
 * - |d| < 0.2: negligible
 * - 0.2 ≤ |d| < 0.5: small
 * - 0.5 ≤ |d| < 0.8: medium
 * - 0.8 ≤ |d| < 1.2: large
 * - |d| ≥ 1.2: very large
 *
 * @example
 * ```typescript
 * const interpretation = interpretCohenD(0.85);
 * console.log(interpretation);  // "large effect"
 * ```
 */
export function interpretCohenD(d: number): string {
  const abs = Math.abs(d);
  if (abs < 0.2) return 'negligible effect';
  if (abs < 0.5) return 'small effect';
  if (abs < 0.8) return 'medium effect';
  if (abs < 1.2) return 'large effect';
  return 'very large effect';
}
