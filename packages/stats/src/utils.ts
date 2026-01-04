/**
 * Statistical utility functions
 * @packageDocumentation
 */

import { tInv } from './distributions.js';

/**
 * Calculates the arithmetic mean of an array
 * @param arr - Array of numbers
 * @returns Mean value
 * @throws {Error} If array is empty
 * @public
 *
 * @example
 * ```typescript
 * const avg = mean([1, 2, 3, 4, 5]);  // 3
 * ```
 */
export function mean(arr: number[]): number {
  if (arr.length === 0) {
    throw new Error('Array must not be empty');
  }
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

/**
 * Calculates the sample standard deviation
 * @param arr - Array of numbers
 * @returns Standard deviation (using n-1 degrees of freedom)
 * @throws {Error} If array has fewer than 2 elements
 * @public
 *
 * @remarks
 * Uses Bessel's correction (n-1) for unbiased estimation of population standard deviation.
 *
 * @example
 * ```typescript
 * const sd = standardDeviation([1, 2, 3, 4, 5]);
 * ```
 */
export function standardDeviation(arr: number[]): number {
  if (arr.length < 2) {
    throw new Error('Array must have at least 2 elements');
  }

  const m = mean(arr);
  const variance = arr.reduce((sum, x) => sum + Math.pow(x - m, 2), 0) / (arr.length - 1);
  return Math.sqrt(variance);
}

/**
 * Calculates the confidence interval for a sample mean
 * @param arr - Array of measurements
 * @param confidence - Confidence level (default: 0.95 for 95% CI)
 * @returns Confidence interval [lower, upper]
 * @throws {Error} If array has fewer than 2 elements
 * @throws {RangeError} If confidence is not in (0, 1)
 * @public
 *
 * @remarks
 * Uses the t-distribution to calculate the confidence interval for the population mean.
 * For n ≥ 30, the t-distribution approximates the normal distribution.
 *
 * @example
 * ```typescript
 * const data = [12.5, 13.2, 11.8, 14.1, 12.9];
 * const ci = confidenceInterval(data);  // 95% CI
 * console.log(`95% CI: [${ci[0].toFixed(2)}, ${ci[1].toFixed(2)}]`);
 * ```
 */
export function confidenceInterval(
  arr: number[],
  confidence: number = 0.95
): [number, number] {
  if (arr.length < 2) {
    throw new Error('Array must have at least 2 elements');
  }
  if (confidence <= 0 || confidence >= 1) {
    throw new RangeError('Confidence level must be in (0, 1)');
  }

  const m = mean(arr);
  const sd = standardDeviation(arr);
  const n = arr.length;
  const alpha = 1 - confidence;
  const tCrit = tInv(alpha / 2, n - 1);
  const margin = tCrit * (sd / Math.sqrt(n));
  return [m - margin, m + margin];
}

/**
 * Calculates the median of an array
 * @param arr - Array of numbers
 * @returns Median value
 * @throws {Error} If array is empty
 * @public
 *
 * @example
 * ```typescript
 * const med = median([1, 2, 3, 4, 5]);  // 3
 * ```
 */
export function median(arr: number[]): number {
  if (arr.length === 0) {
    throw new Error('Array must not be empty');
  }

  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);

  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

/**
 * Calculates quartiles (Q1, Q2/median, Q3) of an array
 * @param arr - Array of numbers
 * @returns Tuple [Q1, Q2, Q3]
 * @throws {Error} If array has fewer than 4 elements
 * @public
 *
 * @example
 * ```typescript
 * const [q1, q2, q3] = quartiles([1, 2, 3, 4, 5, 6, 7, 8, 9]);
 * ```
 */
export function quartiles(arr: number[]): [number, number, number] {
  if (arr.length < 4) {
    throw new Error('Array must have at least 4 elements');
  }

  const sorted = [...arr].sort((a, b) => a - b);
  const q2 = median(sorted);

  const mid = Math.floor(sorted.length / 2);
  const lower = sorted.length % 2 === 0
    ? sorted.slice(0, mid)
    : sorted.slice(0, mid);
  const upper = sorted.length % 2 === 0
    ? sorted.slice(mid)
    : sorted.slice(mid + 1);

  const q1 = median(lower);
  const q3 = median(upper);

  return [q1, q2, q3];
}

/**
 * Detects outliers using the IQR method
 * @param arr - Array of numbers
 * @param multiplier - IQR multiplier (default: 1.5)
 * @returns Indices of outliers
 * @throws {Error} If array has fewer than 4 elements
 * @public
 *
 * @remarks
 * Uses the IQR (Interquartile Range) method:
 * - Lower bound: Q1 - multiplier * IQR
 * - Upper bound: Q3 + multiplier * IQR
 *
 * Common multipliers:
 * - 1.5: standard outlier detection
 * - 3.0: extreme outlier detection
 *
 * @example
 * ```typescript
 * const data = [1, 2, 3, 4, 5, 100];
 * const outlierIndices = detectOutliers(data);  // [5]
 * ```
 */
export function detectOutliers(arr: number[], multiplier: number = 1.5): number[] {
  const [q1, , q3] = quartiles(arr);
  const iqr = q3 - q1;
  const lowerBound = q1 - multiplier * iqr;
  const upperBound = q3 + multiplier * iqr;

  const outliers: number[] = [];
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < lowerBound || arr[i] > upperBound) {
      outliers.push(i);
    }
  }

  return outliers;
}
