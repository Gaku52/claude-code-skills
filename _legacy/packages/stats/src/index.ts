/**
 * Statistical Analysis Library - MIT Master's Level
 *
 * A comprehensive statistical analysis library designed for rigorous research.
 * Implements hypothesis testing, effect size calculations, and complexity validation
 * following MIT master's thesis standards.
 *
 * @packageDocumentation
 *
 * @remarks
 * ## Features
 *
 * - **T-tests**: Paired and independent samples t-tests with p-values and confidence intervals
 * - **Effect Size**: Cohen's d calculation and interpretation
 * - **Regression**: Linear and log-log regression for complexity validation
 * - **Utilities**: Mean, standard deviation, confidence intervals, outlier detection
 * - **Experiment Framework**: Complete before-after analysis with formatted output
 *
 * ## Example Usage
 *
 * ```typescript
 * import { pairedTTest, runBeforeAfterExperiment } from '@claude-code-skills/stats';
 *
 * // Simple t-test
 * const before = [12.5, 13.2, 11.8, 14.1, 12.9];
 * const after = [4.8, 5.2, 4.5, 5.5, 4.9];
 * const result = pairedTTest(before, after);
 * console.log(`p-value: ${result.p < 0.001 ? '<0.001' : result.p.toFixed(3)}`);
 *
 * // Complete experiment
 * const experiment = runBeforeAfterExperiment("Rendering time", before, after);
 * console.log(`Improvement: ${experiment.improvement.toFixed(1)}%`);
 * ```
 *
 * ## Statistical Standards
 *
 * This library follows rigorous statistical practices:
 * - Recommended sample size: n ≥ 30
 * - Two-tailed tests by default
 * - Bessel's correction for unbiased variance
 * - 95% confidence intervals
 * - Cohen's d for effect size
 *
 * @see {@link https://github.com/Gaku52/claude-code-skills | GitHub Repository}
 */

// Type exports
export type { TTestResult, RegressionResult, ExperimentResult } from './types.js';

// Distribution functions
export {
  erf,
  normalCDF,
  normalInv,
  gamma,
  tCDF,
  tInv
} from './distributions.js';

// T-tests
export {
  pairedTTest,
  independentTTest,
  interpretCohenD
} from './ttest.js';

// Regression
export {
  linearRegression,
  logLogRegression
} from './regression.js';

// Utility functions
export {
  mean,
  standardDeviation,
  confidenceInterval,
  median,
  quartiles,
  detectOutliers
} from './utils.js';

// Experiment framework
export {
  runBeforeAfterExperiment,
  formatResults
} from './experiment.js';
