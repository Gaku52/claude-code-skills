#!/usr/bin/env node

import { sendWeeklyReport } from '../notifier';

async function main() {
  try {
    console.log('[Weekly Report] Sending weekly report...');
    await sendWeeklyReport();
    console.log('[Weekly Report] Report sent successfully');
  } catch (error) {
    console.error('[Weekly Report] Failed to send report:', error);
    process.exit(1);
  }
}

main();
