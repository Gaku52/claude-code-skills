#!/usr/bin/env node

import 'dotenv/config';
import { trackUsage } from './src/tracker';
import { db } from './src/database/supabase';

async function main() {
  try {
    console.log('[Test] テスト使用記録を挿入します...');

    // テストデータを挿入
    const recordId = await trackUsage({
      modelId: 'claude-sonnet-4-20250514',
      inputTokens: 1000,
      outputTokens: 500,
      timestamp: new Date(),
      requestId: 'test_' + Date.now(),
      metadata: {
        test: true,
        description: 'Supabase migration test'
      }
    });

    console.log(`[Test] 記録ID: ${recordId}`);

    // 今日の使用状況を取得
    const now = new Date();
    const startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0);
    const endDate = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 23, 59, 59);

    console.log('\n[Test] 今日の使用状況を取得...');
    const totalCost = await db.getTotalCostByDateRange(startDate, endDate);
    console.log(`[Test] 今日の合計コスト: $${totalCost.toFixed(4)}`);

    const modelBreakdown = await db.getUsageByModel(startDate, endDate);
    console.log('\n[Test] モデル別内訳:');
    for (const [modelId, stats] of Object.entries(modelBreakdown)) {
      console.log(`  ${modelId}:`);
      console.log(`    リクエスト: ${stats.requests}`);
      console.log(`    入力トークン: ${stats.inputTokens.toLocaleString()}`);
      console.log(`    出力トークン: ${stats.outputTokens.toLocaleString()}`);
      console.log(`    コスト: $${stats.totalCost.toFixed(4)}`);
    }

    // 今月の合計使用量を取得
    const monthStart = new Date(now.getFullYear(), now.getMonth(), 1, 0, 0, 0);
    const monthToDate = await db.getTotalCostByDateRange(monthStart, endDate);
    const creditBalance = parseFloat(process.env.CREDIT_BALANCE || '0');
    const remainingBalance = creditBalance - monthToDate;

    console.log(`\n[Test] 今月の合計コスト: $${monthToDate.toFixed(4)}`);
    console.log(`[Test] 初期残高: $${creditBalance.toFixed(2)}`);
    console.log(`[Test] 残りクレジット: $${remainingBalance.toFixed(2)}`);

    console.log('\n[Test] テスト完了！');
  } catch (error) {
    console.error('[Test] エラー:', error);
    process.exit(1);
  }
}

main();
