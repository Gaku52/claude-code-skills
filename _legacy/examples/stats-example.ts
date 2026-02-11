/**
 * Statistical Analysis Example
 *
 * Demonstrates the use of @claude-code-skills/stats for
 * rigorous before-after experimental analysis.
 */

import {
  pairedTTest,
  runBeforeAfterExperiment,
  formatResults,
  logLogRegression,
  interpretCohenD
} from '../packages/stats/src/index.js';

console.log('='.repeat(60));
console.log('Statistical Analysis Example');
console.log('='.repeat(60));

// Example 1: Simple paired t-test
console.log('\n## Example 1: Paired T-Test\n');

const renderTimeBefore = [
  12.5, 13.2, 11.8, 14.1, 12.9, 13.5, 12.1, 14.3, 13.0, 12.7,
  13.8, 12.3, 14.0, 12.8, 13.1, 12.6, 13.9, 12.4, 13.3, 12.9,
  13.6, 12.2, 14.2, 13.4, 12.5, 13.7, 12.0, 14.4, 13.2, 12.8
];

const renderTimeAfter = [
  4.8, 5.2, 4.5, 5.5, 4.9, 5.1, 4.7, 5.3, 4.8, 5.0,
  5.4, 4.6, 5.2, 4.9, 5.1, 4.8, 5.3, 4.7, 5.0, 4.9,
  5.2, 4.6, 5.4, 5.1, 4.8, 5.3, 4.5, 5.5, 5.0, 4.9
];

const result = pairedTTest(renderTimeBefore, renderTimeAfter);

console.log(`Sample size: n=${renderTimeBefore.length}`);
console.log(`t-statistic: t(${result.df}) = ${result.t.toFixed(2)}`);
console.log(`p-value: ${result.p < 0.001 ? 'p < 0.001' : `p = ${result.p.toFixed(4)}`}`);
console.log(`Mean difference: ${result.meanDiff.toFixed(2)} ms`);
console.log(`95% CI: [${result.ci[0].toFixed(2)}, ${result.ci[1].toFixed(2)}] ms`);
console.log(`Cohen's d: ${result.d.toFixed(2)} (${interpretCohenD(result.d)})`);

// Example 2: Complete experiment analysis
console.log('\n## Example 2: Complete Experiment Analysis\n');

const memoryBefore = [
  185, 192, 178, 195, 188, 190, 182, 197, 186, 184,
  193, 180, 196, 187, 189, 183, 194, 181, 191, 188,
  192, 179, 198, 190, 185, 193, 177, 199, 189, 186
];

const memoryAfter = [
  92, 95, 88, 97, 93, 94, 90, 96, 92, 91,
  95, 89, 96, 93, 94, 92, 95, 90, 94, 93,
  95, 88, 97, 94, 92, 95, 87, 98, 94, 92
];

const experiments = [
  runBeforeAfterExperiment('Rendering time (ms)', renderTimeBefore, renderTimeAfter),
  runBeforeAfterExperiment('Memory usage (MB)', memoryBefore, memoryAfter)
];

console.log(formatResults(experiments));

// Example 3: Complexity validation
console.log('\n## Example 3: Complexity Validation (Log-Log Regression)\n');

const inputSizes = [100, 200, 500, 1000, 2000, 5000, 10000];
const executionTimes = [2.1, 4.5, 12.3, 26.8, 55.2, 142.1, 298.5];

const regression = logLogRegression(inputSizes, executionTimes);

console.log('Theoretical complexity: O(n log n)');
console.log(`Empirical slope: ${regression.slope.toFixed(2)} (expected: ≈1.0 for O(n log n))`);
console.log(`R² = ${regression.r2.toFixed(4)} (>0.999 indicates excellent fit)`);
console.log(`Intercept: ${regression.intercept.toFixed(2)}`);

if (regression.r2 > 0.999) {
  console.log('✅ Empirical results match theoretical complexity!');
} else {
  console.log('⚠️  Results deviate from theoretical complexity');
}

console.log('\n' + '='.repeat(60));
console.log('Analysis Complete');
console.log('='.repeat(60));
