import * as dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.join(__dirname, '../.env') });

export const config = {
  email: {
    provider: process.env.EMAIL_PROVIDER || 'smtp',
    smtp: {
      host: process.env.SMTP_HOST || 'smtp.gmail.com',
      port: parseInt(process.env.SMTP_PORT || '587'),
      user: process.env.SMTP_USER || '',
      pass: process.env.SMTP_PASS || '',
      from: process.env.SMTP_FROM || process.env.SMTP_USER || '',
    },
    aws: {
      region: process.env.AWS_REGION || 'us-east-1',
      accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
      from: process.env.AWS_SES_FROM || '',
    },
    notifyEmail: process.env.NOTIFY_EMAIL || '',
  },
  thresholds: {
    daily: parseFloat(process.env.COST_THRESHOLD_DAILY || '10'),
    weekly: parseFloat(process.env.COST_THRESHOLD_WEEKLY || '50'),
    monthly: parseFloat(process.env.COST_THRESHOLD_MONTHLY || '200'),
  },
  schedule: {
    dailyReportTime: process.env.DAILY_REPORT_TIME || '09:00',
    weeklyReportDay: process.env.WEEKLY_REPORT_DAY || 'monday',
  },
  database: {
    path: path.join(__dirname, '../data/usage.db'),
  },
};

// Pricing (per 1M tokens)
export const PRICING = {
  'claude-sonnet-4-20250514': {
    input: 3.0,
    output: 15.0,
  },
  'claude-3-5-sonnet-20241022': {
    input: 3.0,
    output: 15.0,
  },
  'claude-3-5-haiku-20241022': {
    input: 0.8,
    output: 4.0,
  },
  'claude-3-opus-20240229': {
    input: 15.0,
    output: 75.0,
  },
} as const;

export type ModelId = keyof typeof PRICING;
