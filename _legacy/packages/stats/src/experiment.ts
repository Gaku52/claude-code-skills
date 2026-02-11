/**
 * Experiment framework for before-after analysis
 * @packageDocumentation
 */

import { pairedTTest, interpretCohenD } from './ttest.js';
import { mean, standardDeviation, confidenceInterval } from './utils.js';
import type { ExperimentResult } from './types.js';

/**
 * Runs a before-after experiment and performs statistical analysis
 * @param name - Name of the metric being measured
 * @param before - Measurements before treatment
 * @param after - Measurements after treatment
 * @returns Complete experiment results with statistical tests
 * @throws {Error} If arrays have different lengths or insufficient data
 * @public
 *
 * @remarks
 * Performs a comprehensive before-after analysis including:
 * - Descriptive statistics (mean, SD, 95% CI)
 * - Paired t-test for statistical significance
 * - Cohen's d effect size
 * - Improvement percentage
 *
 * Recommended sample size: n ≥ 30 for robust results
 *
 * @example
 * ```typescript
 * const before = [12.5, 13.2, 11.8, 14.1, 12.9];
 * const after = [4.8, 5.2, 4.5, 5.5, 4.9];
 * const result = runBeforeAfterExperiment("Rendering time (ms)", before, after);
 *
 * console.log(`Improvement: ${result.improvement.toFixed(1)}%`);
 * console.log(`p-value: ${result.tTest.p < 0.001 ? '<0.001' : result.tTest.p.toFixed(3)}`);
 * console.log(`Cohen's d: ${result.tTest.d.toFixed(2)}`);
 * ```
 */
export function runBeforeAfterExperiment(
  name: string,
  before: number[],
  after: number[]
): ExperimentResult {
  const beforeMean = mean(before);
  const beforeSd = standardDeviation(before);
  const beforeCI = confidenceInterval(before);

  const afterMean = mean(after);
  const afterSd = standardDeviation(after);
  const afterCI = confidenceInterval(after);

  const improvement = ((beforeMean - afterMean) / beforeMean) * 100;

  const tTest = pairedTTest(before, after);

  return {
    name,
    before: { mean: beforeMean, sd: beforeSd, ci: beforeCI },
    after: { mean: afterMean, sd: afterSd, ci: afterCI },
    improvement,
    tTest
  };
}

/**
 * Formats experiment results as Markdown table
 * @param results - Array of experiment results
 * @returns Markdown-formatted string
 * @public
 *
 * @remarks
 * Generates a publication-ready Markdown table with:
 * - Descriptive statistics (mean ± SD)
 * - Improvement percentage
 * - Statistical test results (t-value, p-value)
 * - Effect size (Cohen's d)
 *
 * @example
 * ```typescript
 * const results = [
 *   runBeforeAfterExperiment("Time", before1, after1),
 *   runBeforeAfterExperiment("Memory", before2, after2)
 * ];
 * console.log(formatResults(results));
 * ```
 */
export function formatResults(results: ExperimentResult[]): string {
  let output = '## Experiment Results\n\n';
  output += '### Statistical Test Results\n\n';
  output += '| Metric | Before | After | Improvement | t-value | p-value | Cohen\'s d |\n';
  output += '|--------|--------|-------|-------------|---------|---------|------------|\n';

  for (const result of results) {
    const before = `${result.before.mean.toFixed(2)} (±${result.before.sd.toFixed(2)})`;
    const after = `${result.after.mean.toFixed(2)} (±${result.after.sd.toFixed(2)})`;
    const improvement = `${result.improvement > 0 ? '-' : '+'}${Math.abs(result.improvement).toFixed(1)}%`;
    const t = `t(${result.tTest.df})=${result.tTest.t.toFixed(1)}`;
    const p = result.tTest.p < 0.001 ? '<0.001' : result.tTest.p.toFixed(3);
    const d = `d=${result.tTest.d.toFixed(1)}`;

    output += `| ${result.name} | ${before} | ${after} | ${improvement} | ${t} | ${p} | ${d} |\n`;
  }

  output += '\n### Detailed Analysis\n\n';
  for (const result of results) {
    output += `**${result.name}**:\n`;
    output += `- Before: ${result.before.mean.toFixed(2)} (SD=${result.before.sd.toFixed(2)}, 95% CI [${result.before.ci[0].toFixed(2)}, ${result.before.ci[1].toFixed(2)}])\n`;
    output += `- After: ${result.after.mean.toFixed(2)} (SD=${result.after.sd.toFixed(2)}, 95% CI [${result.after.ci[0].toFixed(2)}, ${result.after.ci[1].toFixed(2)}])\n`;
    output += `- Improvement: ${result.improvement.toFixed(1)}%\n`;
    output += `- Statistical test: t(${result.tTest.df}) = ${result.tTest.t.toFixed(2)}, p ${result.tTest.p < 0.001 ? '<' : '='} ${result.tTest.p < 0.001 ? '0.001' : result.tTest.p.toFixed(3)}\n`;
    output += `- Cohen's d: ${result.tTest.d.toFixed(2)} (${interpretCohenD(result.tTest.d)})\n\n`;
  }

  return output;
}
