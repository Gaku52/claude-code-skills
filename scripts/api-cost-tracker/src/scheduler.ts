import cron from 'node-cron';
import { sendDailyReport, sendWeeklyReport } from './notifier';
import { config } from './config';

console.log('📅 Scheduler started');
console.log(`Daily report scheduled at: ${config.schedule.dailyReportTime}`);
console.log(`Weekly report scheduled on: ${config.schedule.weeklyReportDay}`);

// 日次レポート（毎日 09:00）
const [dailyHour, dailyMinute] = config.schedule.dailyReportTime.split(':');
cron.schedule(`${dailyMinute} ${dailyHour} * * *`, async () => {
  console.log('[Scheduler] Running daily report...');
  try {
    await sendDailyReport();
  } catch (error) {
    console.error('[Scheduler] Failed to send daily report:', error);
  }
});

// 週次レポート（毎週月曜日 09:00）
const weekdayMap: Record<string, number> = {
  sunday: 0,
  monday: 1,
  tuesday: 2,
  wednesday: 3,
  thursday: 4,
  friday: 5,
  saturday: 6,
};

const weeklyDay = weekdayMap[config.schedule.weeklyReportDay.toLowerCase()] || 1;
cron.schedule(`${dailyMinute} ${dailyHour} * * ${weeklyDay}`, async () => {
  console.log('[Scheduler] Running weekly report...');
  try {
    await sendWeeklyReport();
  } catch (error) {
    console.error('[Scheduler] Failed to send weekly report:', error);
  }
});

// 月次レポート（毎月1日 09:00）
cron.schedule(`${dailyMinute} ${dailyHour} 1 * *`, async () => {
  console.log('[Scheduler] Running monthly report...');
  try {
    const { sendMonthlyReport } = await import('./notifier');
    await sendMonthlyReport();
  } catch (error) {
    console.error('[Scheduler] Failed to send monthly report:', error);
  }
});

console.log('✅ All schedules initialized');

// プロセスを維持
process.on('SIGINT', () => {
  console.log('\n👋 Scheduler stopped');
  process.exit(0);
});
