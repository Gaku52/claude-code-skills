#!/usr/bin/env node

import { sendMonthlyReport } from '../notifier';

async function main() {
  try {
    console.log('[Monthly Report] Sending monthly report...');
    await sendMonthlyReport();
    console.log('[Monthly Report] Report sent successfully');
  } catch (error) {
    console.error('[Monthly Report] Failed to send report:', error);
    process.exit(1);
  }
}

main();
