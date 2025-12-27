import { db } from './database';
import { PRICING, ModelId } from './config';

export interface TrackUsageInput {
  modelId: string;
  inputTokens: number;
  outputTokens: number;
  timestamp: Date;
  requestId: string;
  metadata?: Record<string, any>;
}

export function calculateCost(modelId: string, inputTokens: number, outputTokens: number): {
  inputCost: number;
  outputCost: number;
  totalCost: number;
} {
  const pricing = PRICING[modelId as ModelId];

  if (!pricing) {
    throw new Error(`Unknown model: ${modelId}`);
  }

  // Cost per million tokens
  const inputCost = (inputTokens / 1_000_000) * pricing.input;
  const outputCost = (outputTokens / 1_000_000) * pricing.output;
  const totalCost = inputCost + outputCost;

  return {
    inputCost: parseFloat(inputCost.toFixed(6)),
    outputCost: parseFloat(outputCost.toFixed(6)),
    totalCost: parseFloat(totalCost.toFixed(6)),
  };
}

export async function trackUsage(input: TrackUsageInput): Promise<number> {
  const { inputCost, outputCost, totalCost } = calculateCost(
    input.modelId,
    input.inputTokens,
    input.outputTokens
  );

  const recordId = db.insertUsage({
    request_id: input.requestId,
    model_id: input.modelId,
    input_tokens: input.inputTokens,
    output_tokens: input.outputTokens,
    input_cost: inputCost,
    output_cost: outputCost,
    total_cost: totalCost,
    timestamp: input.timestamp,
    metadata: input.metadata ? JSON.stringify(input.metadata) : undefined,
  });

  console.log(`[Tracker] Recorded usage: $${totalCost.toFixed(4)} (ID: ${recordId})`);

  return recordId;
}

// 使用例
export async function exampleUsage() {
  // Claude API呼び出し後の例
  const response = {
    id: 'msg_123abc',
    model: 'claude-sonnet-4-20250514',
    usage: {
      input_tokens: 1000,
      output_tokens: 500,
    },
  };

  await trackUsage({
    modelId: response.model,
    inputTokens: response.usage.input_tokens,
    outputTokens: response.usage.output_tokens,
    timestamp: new Date(),
    requestId: response.id,
    metadata: {
      // 追加情報（オプション）
      task: 'code_generation',
      user: 'user_123',
    },
  });
}
