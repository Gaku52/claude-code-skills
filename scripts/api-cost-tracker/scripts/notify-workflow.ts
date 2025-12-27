#!/usr/bin/env tsx

import { sendWorkflowCompletionNotification } from '../src/notifier';

async function main() {
  const workflowName = process.argv[2];
  const status = process.argv[3] as 'success' | 'failure';
  const details = process.argv[4];

  if (!workflowName || !status) {
    console.error('Usage: npm run notify:workflow <workflow-name> <success|failure> [details]');
    process.exit(1);
  }

  if (status !== 'success' && status !== 'failure') {
    console.error('Status must be either "success" or "failure"');
    process.exit(1);
  }

  try {
    await sendWorkflowCompletionNotification(workflowName, status, details);
    console.log('✅ Workflow notification sent successfully');
  } catch (error) {
    console.error('❌ Failed to send workflow notification:', error);
    process.exit(1);
  }
}

main();
