import nodemailer from 'nodemailer';
import { SESClient, SendEmailCommand } from '@aws-sdk/client-ses';
import { config } from '../config';

export interface EmailOptions {
  to: string;
  subject: string;
  text?: string;
  html?: string;
}

abstract class EmailSender {
  abstract send(options: EmailOptions): Promise<void>;
}

class SMTPEmailSender extends EmailSender {
  private transporter: nodemailer.Transporter;

  constructor() {
    super();
    this.transporter = nodemailer.createTransport({
      host: config.email.smtp.host,
      port: config.email.smtp.port,
      secure: false, // TLS
      auth: {
        user: config.email.smtp.user,
        pass: config.email.smtp.pass,
      },
    });
  }

  async send(options: EmailOptions): Promise<void> {
    await this.transporter.sendMail({
      from: config.email.smtp.from,
      to: options.to,
      subject: options.subject,
      text: options.text,
      html: options.html,
    });

    console.log(`[Email] Sent to ${options.to}: ${options.subject}`);
  }
}

class AWSEmailSender extends EmailSender {
  private client: SESClient;

  constructor() {
    super();
    this.client = new SESClient({
      region: config.email.aws.region,
      credentials: {
        accessKeyId: config.email.aws.accessKeyId,
        secretAccessKey: config.email.aws.secretAccessKey,
      },
    });
  }

  async send(options: EmailOptions): Promise<void> {
    const command = new SendEmailCommand({
      Source: config.email.aws.from,
      Destination: {
        ToAddresses: [options.to],
      },
      Message: {
        Subject: {
          Data: options.subject,
        },
        Body: {
          Text: options.text ? { Data: options.text } : undefined,
          Html: options.html ? { Data: options.html } : undefined,
        },
      },
    });

    await this.client.send(command);

    console.log(`[Email/AWS SES] Sent to ${options.to}: ${options.subject}`);
  }
}

function createEmailSender(): EmailSender {
  if (config.email.provider === 'aws-ses') {
    return new AWSEmailSender();
  }
  return new SMTPEmailSender();
}

export const emailSender = createEmailSender();

// テスト用関数
export async function testEmail(): Promise<void> {
  try {
    await emailSender.send({
      to: config.email.notifyEmail,
      subject: 'Test Email from Claude API Cost Tracker',
      text: 'This is a test email. If you receive this, email configuration is working correctly!',
      html: '<p>This is a test email. If you receive this, <strong>email configuration is working correctly!</strong></p>',
    });

    console.log('✅ Test email sent successfully!');
  } catch (error) {
    console.error('❌ Failed to send test email:', error);
    throw error;
  }
}
