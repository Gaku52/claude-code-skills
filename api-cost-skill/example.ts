/**
 * Claude API Cost Tracker 使用例
 */

import Anthropic from '@anthropic-ai/sdk';
import { trackUsage } from './src/tracker';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY || '',
});

async function main() {
  console.log('Claude API呼び出し開始...\n');

  // Claude API呼び出し
  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 1024,
    messages: [
      {
        role: 'user',
        content: 'こんにちは！Next.jsの最適化について教えてください。',
      },
    ],
  });

  console.log('レスポンス:', response.content[0].text.substring(0, 200) + '...\n');

  // 使用量を記録
  await trackUsage({
    modelId: response.model,
    inputTokens: response.usage.input_tokens,
    outputTokens: response.usage.output_tokens,
    timestamp: new Date(),
    requestId: response.id,
    metadata: {
      task: 'nextjs_optimization_question',
      user: 'example_user',
    },
  });

  console.log('✅ 使用量が記録されました');
  console.log(`入力トークン: ${response.usage.input_tokens.toLocaleString()}`);
  console.log(`出力トークン: ${response.usage.output_tokens.toLocaleString()}`);
}

main().catch(console.error);
