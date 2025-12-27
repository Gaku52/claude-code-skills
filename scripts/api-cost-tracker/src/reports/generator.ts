import { db } from '../database';
import { config } from '../config';

export interface Report {
  period: string;
  startDate: Date;
  endDate: Date;
  totalCost: number;
  modelBreakdown: Record<
    string,
    {
      requests: number;
      inputTokens: number;
      outputTokens: number;
      totalCost: number;
    }
  >;
  thresholdExceeded?: boolean;
  threshold?: number;
}

export function generateDailyReport(): Report {
  const now = new Date();
  const startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0);
  const endDate = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 23, 59, 59);

  const totalCost = db.getTotalCostByDateRange(startDate, endDate);
  const modelBreakdown = db.getUsageByModel(startDate, endDate);

  return {
    period: `Daily Report - ${startDate.toLocaleDateString()}`,
    startDate,
    endDate,
    totalCost,
    modelBreakdown,
    thresholdExceeded: totalCost > config.thresholds.daily,
    threshold: config.thresholds.daily,
  };
}

export function generateWeeklyReport(): Report {
  const now = new Date();
  const dayOfWeek = now.getDay();
  const diff = now.getDate() - dayOfWeek + (dayOfWeek === 0 ? -6 : 1); // Adjust when day is Sunday
  const startDate = new Date(now.getFullYear(), now.getMonth(), diff, 0, 0, 0);
  const endDate = new Date(now.getFullYear(), now.getMonth(), diff + 6, 23, 59, 59);

  const totalCost = db.getTotalCostByDateRange(startDate, endDate);
  const modelBreakdown = db.getUsageByModel(startDate, endDate);

  return {
    period: `Weekly Report - Week of ${startDate.toLocaleDateString()}`,
    startDate,
    endDate,
    totalCost,
    modelBreakdown,
    thresholdExceeded: totalCost > config.thresholds.weekly,
    threshold: config.thresholds.weekly,
  };
}

export function generateMonthlyReport(): Report {
  const now = new Date();
  const startDate = new Date(now.getFullYear(), now.getMonth(), 1, 0, 0, 0);
  const endDate = new Date(now.getFullYear(), now.getMonth() + 1, 0, 23, 59, 59);

  const totalCost = db.getTotalCostByDateRange(startDate, endDate);
  const modelBreakdown = db.getUsageByModel(startDate, endDate);

  return {
    period: `Monthly Report - ${startDate.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}`,
    startDate,
    endDate,
    totalCost,
    modelBreakdown,
    thresholdExceeded: totalCost > config.thresholds.monthly,
    threshold: config.thresholds.monthly,
  };
}

export function formatReportAsText(report: Report): string {
  let text = `
Claude API Usage Report
${report.period}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

`;

  if (Object.keys(report.modelBreakdown).length === 0) {
    text += 'No API usage during this period.\n';
  } else {
    for (const [modelId, stats] of Object.entries(report.modelBreakdown)) {
      text += `
Model: ${modelId}
Requests: ${stats.requests.toLocaleString()}

Input Tokens:  ${stats.inputTokens.toLocaleString()}
Output Tokens: ${stats.outputTokens.toLocaleString()}
Cost:          $${stats.totalCost.toFixed(4)}

`;
    }
  }

  text += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Cost: $${report.totalCost.toFixed(4)}
`;

  if (report.threshold) {
    text += `Threshold:  $${report.threshold.toFixed(2)}
`;
    if (report.thresholdExceeded) {
      text += `
⚠️  WARNING: Threshold exceeded by $${(report.totalCost - report.threshold).toFixed(4)}
`;
    }
  }

  text += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`;

  return text;
}

export function formatReportAsHTML(report: Report): string {
  let modelRows = '';

  if (Object.keys(report.modelBreakdown).length === 0) {
    modelRows = '<tr><td colspan="4">No API usage during this period.</td></tr>';
  } else {
    for (const [modelId, stats] of Object.entries(report.modelBreakdown)) {
      modelRows += `
      <tr>
        <td>${modelId}</td>
        <td>${stats.requests.toLocaleString()}</td>
        <td>${stats.inputTokens.toLocaleString()} / ${stats.outputTokens.toLocaleString()}</td>
        <td>$${stats.totalCost.toFixed(4)}</td>
      </tr>
      `;
    }
  }

  const thresholdWarning = report.thresholdExceeded
    ? `
    <div style="background-color: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 5px; margin-top: 20px;">
      <strong>⚠️ Warning:</strong> Threshold exceeded by $${(report.totalCost - (report.threshold || 0)).toFixed(4)}
    </div>
  `
    : '';

  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
    h1 { color: #333; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background-color: #f8f9fa; font-weight: 600; }
    .total { font-size: 18px; font-weight: 600; margin: 20px 0; }
  </style>
</head>
<body>
  <h1>Claude API Usage Report</h1>
  <p><strong>${report.period}</strong></p>

  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Requests</th>
        <th>Input / Output Tokens</th>
        <th>Cost</th>
      </tr>
    </thead>
    <tbody>
      ${modelRows}
    </tbody>
  </table>

  <div class="total">
    Total Cost: $${report.totalCost.toFixed(4)}
    ${report.threshold ? `<br>Threshold: $${report.threshold.toFixed(2)}` : ''}
  </div>

  ${thresholdWarning}

  <p style="margin-top: 30px; color: #666; font-size: 14px;">
    Generated at ${new Date().toLocaleString()}
  </p>
</body>
</html>
  `;
}
