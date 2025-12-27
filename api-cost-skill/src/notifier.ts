import { emailSender } from './email/sender';
import { generateDailyReport, generateWeeklyReport, generateMonthlyReport, formatReportAsText, formatReportAsHTML } from './reports/generator';
import { config } from './config';

export async function sendDailyReport(): Promise<void> {
  const report = generateDailyReport();

  await emailSender.send({
    to: config.email.notifyEmail,
    subject: `${report.thresholdExceeded ? '⚠️ ' : ''}[claude-code-skills] Claude API 日次レポート - ${new Date().toLocaleDateString('ja-JP')}`,
    text: formatReportAsText(report),
    html: formatReportAsHTML(report),
  });

  console.log('[Notifier] Daily report sent');
}

export async function sendWeeklyReport(): Promise<void> {
  const report = generateWeeklyReport();

  await emailSender.send({
    to: config.email.notifyEmail,
    subject: `[claude-code-skills] Claude API 週次レポート - ${report.startDate.toLocaleDateString('ja-JP')}の週`,
    text: formatReportAsText(report),
    html: formatReportAsHTML(report),
  });

  console.log('[Notifier] Weekly report sent');
}

export async function sendMonthlyReport(): Promise<void> {
  const report = generateMonthlyReport();

  await emailSender.send({
    to: config.email.notifyEmail,
    subject: `[claude-code-skills] Claude API 月次レポート - ${report.startDate.toLocaleDateString('ja-JP', { year: 'numeric', month: 'long' })}`,
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

export async function sendWorkflowCompletionNotification(
  workflowName: string,
  status: 'success' | 'failure',
  details?: string
): Promise<void> {
  const statusEmoji = status === 'success' ? '✅' : '❌';
  const subject = `${statusEmoji} GitHub Workflow: ${workflowName}`;

  const text = `
GitHub Workflow Notification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Workflow: ${workflowName}
Status: ${status.toUpperCase()}
Time: ${new Date().toLocaleString('ja-JP', { timeZone: 'Asia/Tokyo' })}

${details || 'Workflow completed.'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  `;

  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
    .notification {
      background-color: ${status === 'success' ? '#d4edda' : '#f8d7da'};
      border: 2px solid ${status === 'success' ? '#28a745' : '#dc3545'};
      padding: 20px;
      border-radius: 5px;
    }
    .notification h2 {
      margin-top: 0;
      color: ${status === 'success' ? '#155724' : '#721c24'};
    }
    .stats { margin: 20px 0; }
    .stat { margin: 10px 0; font-size: 16px; }
    .stat strong { display: inline-block; width: 150px; }
    .details {
      background-color: #f8f9fa;
      padding: 15px;
      border-radius: 5px;
      margin-top: 15px;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <div class="notification">
    <h2>${statusEmoji} Workflow: ${workflowName}</h2>
    <div class="stats">
      <div class="stat"><strong>Status:</strong> ${status.toUpperCase()}</div>
      <div class="stat"><strong>Time:</strong> ${new Date().toLocaleString('ja-JP', { timeZone: 'Asia/Tokyo' })}</div>
    </div>
    ${details ? `<div class="details">${details}</div>` : '<p>Workflow completed.</p>'}
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

  console.log(`[Notifier] Workflow completion notification sent for: ${workflowName}`);
}
