/**
 * Statistical analysis types
 * @packageDocumentation
 */

/**
 * Result of a t-test analysis
 * @public
 */
export interface TTestResult {
  /** t-statistic value */
  t: number;
  /** Degrees of freedom */
  df: number;
  /** Two-tailed p-value */
  p: number;
  /** 95% confidence interval [lower, upper] */
  ci: [number, number];
  /** Mean difference between groups */
  meanDiff: number;
  /** Standard deviation */
  sd: number;
  /** Cohen's d effect size */
  d: number;
}

/**
 * Result of a linear regression analysis
 * @public
 */
export interface RegressionResult {
  /** Slope of the regression line */
  slope: number;
  /** Y-intercept of the regression line */
  intercept: number;
  /** Coefficient of determination (R²) */
  r2: number;
  /** Residuals for each data point */
  residuals: number[];
}

/**
 * Result of a before-after experiment
 * @public
 */
export interface ExperimentResult {
  /** Name of the metric being measured */
  name: string;
  /** Before measurements summary */
  before: {
    /** Mean value */
    mean: number;
    /** Standard deviation */
    sd: number;
    /** 95% confidence interval */
    ci: [number, number];
  };
  /** After measurements summary */
  after: {
    /** Mean value */
    mean: number;
    /** Standard deviation */
    sd: number;
    /** 95% confidence interval */
    ci: [number, number];
  };
  /** Percentage improvement (negative means worse) */
  improvement: number;
  /** T-test results */
  tTest: TTestResult;
}
