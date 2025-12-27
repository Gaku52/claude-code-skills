import { emailSender } from './email/sender';
import { generateDailyReport, generateWeeklyReport, generateMonthlyReport, formatReportAsText, formatReportAsHTML } from './reports/generator';
import { config } from './config';

export async function sendDailyReport(): Promise<void> {
  const report = generateDailyReport();

  await emailSender.send({
    to: config.email.notifyEmail,
    subject: `${report.thresholdExceeded ? '⚠️ ' : ''}Claude API Daily Report - ${new Date().toLocaleDateString()}`,
    text: formatReportAsText(report),
    html: formatReportAsHTML(report),
  });

  console.log('[Notifier] Daily report sent');
}

export async function sendWeeklyReport(): Promise<void> {
  const report = generateWeeklyReport();

  await emailSender.send({
    to: config.email.notifyEmail,
    subject: `Claude API Weekly Report - Week of ${report.startDate.toLocaleDateString()}`,
    text: formatReportAsText(report),
    html: formatReportAsHTML(report),
  });

  console.log('[Notifier] Weekly report sent');
}

export async function sendMonthlyReport(): Promise<void> {
  const report = generateMonthlyReport();

  await emailSender.send({
    to: config.email.notifyEmail,
    subject: `Claude API Monthly Report - ${report.startDate.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}`,
    text: formatReportAsText(report),
    html: formatReportAsHTML(report),
  });

  console.log('[Notifier] Monthly report sent');
}

export async function sendThresholdAlert(period: 'daily' | 'weekly' | 'monthly', currentCost: number, threshold: number): Promise<void> {
  const subject = `⚠️ Claude API Cost Alert: ${period.charAt(0).toUpperCase() + period.slice(1)} Threshold Exceeded`;
  const text = `
Claude API Cost Alert
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Period: ${period.toUpperCase()}
Current Cost: $${currentCost.toFixed(4)}
Threshold: $${threshold.toFixed(2)}
Exceeded by: $${(currentCost - threshold).toFixed(4)}

Please review your API usage.

View detailed report at: https://console.anthropic.com

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  `;

  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
    .alert { background-color: #fff3cd; border: 2px solid #ffc107; padding: 20px; border-radius: 5px; }
    .alert h2 { margin-top: 0; color: #856404; }
    .stats { margin: 20px 0; }
    .stat { margin: 10px 0; font-size: 16px; }
    .stat strong { display: inline-block; width: 150px; }
  </style>
</head>
<body>
  <div class="alert">
    <h2>⚠️ Cost Alert: ${period.charAt(0).toUpperCase() + period.slice(1)} Threshold Exceeded</h2>
    <div class="stats">
      <div class="stat"><strong>Period:</strong> ${period.toUpperCase()}</div>
      <div class="stat"><strong>Current Cost:</strong> $${currentCost.toFixed(4)}</div>
      <div class="stat"><strong>Threshold:</strong> $${threshold.toFixed(2)}</div>
      <div class="stat"><strong>Exceeded by:</strong> $${(currentCost - threshold).toFixed(4)}</div>
    </div>
    <p>Please review your API usage.</p>
    <p><a href="https://console.anthropic.com">View detailed report at Anthropic Console</a></p>
  </div>
</body>
</html>
  `;

  await emailSender.send({
    to: config.email.notifyEmail,
    subject,
    text,
    html,
  });

  console.log(`[Notifier] ${period} threshold alert sent`);
}
