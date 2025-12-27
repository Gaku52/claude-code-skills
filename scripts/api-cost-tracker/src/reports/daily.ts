#!/usr/bin/env node

import { sendDailyReport } from '../notifier';

async function main() {
  try {
    console.log('[Daily Report] Sending daily report...');
    await sendDailyReport();
    console.log('[Daily Report] Report sent successfully');
  } catch (error) {
    console.error('[Daily Report] Failed to send report:', error);
    process.exit(1);
  }
}

main();
