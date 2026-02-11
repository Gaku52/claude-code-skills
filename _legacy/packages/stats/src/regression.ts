/**
 * Regression analysis for complexity validation
 * @packageDocumentation
 */

import type { RegressionResult } from './types.js';

/**
 * Performs simple linear regression
 * @param x - Independent variable values
 * @param y - Dependent variable values
 * @returns Regression results including slope, intercept, R², and residuals
 * @throws {Error} If arrays have different lengths or fewer than 2 points
 * @public
 *
 * @remarks
 * Fits a line y = slope * x + intercept using least squares method.
 * R² (coefficient of determination) indicates how well the model fits the data:
 * - R² = 1: perfect fit
 * - R² = 0: no linear relationship
 *
 * @example
 * ```typescript
 * const x = [1, 2, 3, 4, 5];
 * const y = [2.1, 4.2, 5.9, 8.1, 10.0];
 * const result = linearRegression(x, y);
 * console.log(`y = ${result.slope.toFixed(2)}x + ${result.intercept.toFixed(2)}`);
 * console.log(`R² = ${result.r2.toFixed(4)}`);
 * ```
 */
export function linearRegression(x: number[], y: number[]): RegressionResult {
  if (x.length !== y.length) {
    throw new Error('x and y arrays must have the same length');
  }
  if (x.length < 2) {
    throw new Error('At least 2 data points are required');
  }

  const n = x.length;

  const meanX = x.reduce((a, b) => a + b, 0) / n;
  const meanY = y.reduce((a, b) => a + b, 0) / n;

  const ssXY = x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0);
  const ssXX = x.reduce((sum, xi) => sum + Math.pow(xi - meanX, 2), 0);

  // Handle case where all x values are identical
  if (ssXX === 0) {
    return {
      slope: 0,
      intercept: meanY,
      r2: 0,
      residuals: y.map(yi => yi - meanY)
    };
  }

  const slope = ssXY / ssXX;
  const intercept = meanY - slope * meanX;

  // Predicted values and residuals
  const predicted = x.map(xi => slope * xi + intercept);
  const residuals = y.map((yi, i) => yi - predicted[i]);

  // R² (coefficient of determination)
  const ssRes = residuals.reduce((sum, r) => sum + r * r, 0);
  const ssTot = y.reduce((sum, yi) => sum + Math.pow(yi - meanY, 2), 0);
  const r2 = ssTot === 0 ? 1 : 1 - (ssRes / ssTot);

  return { slope, intercept, r2, residuals };
}

/**
 * Performs log-log regression for complexity analysis
 * @param n - Input sizes
 * @param time - Execution times
 * @returns Regression results in log-log space
 * @throws {Error} If arrays have different lengths or invalid values
 * @public
 *
 * @remarks
 * Transforms data to log-log space and performs linear regression.
 * The slope indicates the complexity exponent:
 * - slope ≈ 1.0: O(n)
 * - slope ≈ 2.0: O(n²)
 * - slope ≈ 1.0: O(n log n) for algorithms with logarithmic component
 *
 * All input values must be positive (n > 0, time > 0).
 *
 * @example
 * ```typescript
 * const n = [100, 200, 500, 1000, 2000];
 * const time = [2.1, 4.5, 12.3, 26.8, 55.2];
 * const result = logLogRegression(n, time);
 * console.log(`Empirical complexity: O(n^${result.slope.toFixed(2)})`);
 * console.log(`R² = ${result.r2.toFixed(4)}`);
 * ```
 */
export function logLogRegression(n: number[], time: number[]): RegressionResult {
  if (n.some(x => x <= 0)) {
    throw new Error('All n values must be positive');
  }
  if (time.some(t => t <= 0)) {
    throw new Error('All time values must be positive');
  }

  const logN = n.map(x => Math.log(x));
  const logTime = time.map(t => Math.log(t));
  return linearRegression(logN, logTime);
}
