import { createClient, SupabaseClient } from '@supabase/supabase-js';

let supabase: SupabaseClient;

function getSupabase(): SupabaseClient {
  if (!supabase) {
    supabase = createClient(
      process.env.SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_KEY!
    );
  }
  return supabase;
}

export interface UsageRecord {
  id?: string;
  request_id: string;
  model_id: string;
  input_tokens: number;
  output_tokens: number;
  input_cost: number;
  output_cost: number;
  total_cost: number;
  timestamp: Date;
  metadata?: string | null;
  created_at?: Date;
}

class DatabaseManager {
  async insertUsage(record: Omit<UsageRecord, 'id' | 'created_at'>): Promise<string> {
    const { data, error } = await getSupabase()
      .from('claude_usage_history')
      .insert({
        request_id: record.request_id,
        model_id: record.model_id,
        input_tokens: record.input_tokens,
        output_tokens: record.output_tokens,
        input_cost: record.input_cost,
        output_cost: record.output_cost,
        total_cost: record.total_cost,
        timestamp: record.timestamp.toISOString(),
        operation_type: 'api_call',
        metadata: record.metadata ? JSON.parse(record.metadata) : null,
      })
      .select('id')
      .single();

    if (error) {
      throw new Error(`Failed to insert usage: ${error.message}`);
    }

    return data.id;
  }

  async getUsageByDateRange(startDate: Date, endDate: Date): Promise<UsageRecord[]> {
    const { data, error } = await getSupabase()
      .from('claude_usage_history')
      .select('*')
      .gte('timestamp', startDate.toISOString())
      .lte('timestamp', endDate.toISOString())
      .order('timestamp', { ascending: false });

    if (error) {
      throw new Error(`Failed to get usage: ${error.message}`);
    }

    return (data || []).map((row) => ({
      id: row.id,
      request_id: row.request_id,
      model_id: row.model_id,
      input_tokens: row.input_tokens,
      output_tokens: row.output_tokens,
      input_cost: Number(row.input_cost),
      output_cost: Number(row.output_cost),
      total_cost: Number(row.total_cost),
      timestamp: new Date(row.timestamp),
      metadata: row.metadata ? JSON.stringify(row.metadata) : null,
      created_at: new Date(row.created_at),
    }));
  }

  async getTotalCostByDateRange(startDate: Date, endDate: Date): Promise<number> {
    const { data, error } = await getSupabase()
      .from('claude_usage_history')
      .select('total_cost')
      .gte('timestamp', startDate.toISOString())
      .lte('timestamp', endDate.toISOString());

    if (error) {
      throw new Error(`Failed to get total cost: ${error.message}`);
    }

    return (data || []).reduce((sum, record) => sum + Number(record.total_cost), 0);
  }

  async getUsageByModel(
    startDate: Date,
    endDate: Date
  ): Promise<
    Record<
      string,
      {
        requests: number;
        inputTokens: number;
        outputTokens: number;
        totalCost: number;
      }
    >
  > {
    const { data, error } = await getSupabase()
      .from('claude_usage_history')
      .select('*')
      .gte('timestamp', startDate.toISOString())
      .lte('timestamp', endDate.toISOString());

    if (error) {
      throw new Error(`Failed to get usage by model: ${error.message}`);
    }

    const breakdown: Record<
      string,
      {
        requests: number;
        inputTokens: number;
        outputTokens: number;
        totalCost: number;
      }
    > = {};

    for (const record of data || []) {
      const modelId = record.model_id;
      if (!breakdown[modelId]) {
        breakdown[modelId] = {
          requests: 0,
          inputTokens: 0,
          outputTokens: 0,
          totalCost: 0,
        };
      }
      breakdown[modelId].requests++;
      breakdown[modelId].inputTokens += record.input_tokens;
      breakdown[modelId].outputTokens += record.output_tokens;
      breakdown[modelId].totalCost += Number(record.total_cost);
    }

    return breakdown;
  }

  // 不要（Supabaseは接続プール管理）
  close() {
    // No-op for Supabase
  }
}

export const db = new DatabaseManager();
