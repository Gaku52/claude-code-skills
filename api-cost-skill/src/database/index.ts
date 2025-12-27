import Database from 'better-sqlite3';
import { config } from '../config';
import { SCHEMA, UsageRecord } from './schema';
import fs from 'fs';
import path from 'path';

class DatabaseManager {
  private db: Database.Database;

  constructor() {
    // データベースディレクトリの作成
    const dbDir = path.dirname(config.database.path);
    if (!fs.existsSync(dbDir)) {
      fs.mkdirSync(dbDir, { recursive: true });
    }

    this.db = new Database(config.database.path);
    this.initialize();
  }

  private initialize() {
    this.db.exec(SCHEMA);
  }

  insertUsage(record: Omit<UsageRecord, 'id' | 'created_at'>): number {
    const stmt = this.db.prepare(`
      INSERT INTO usage_history (
        request_id, model_id, input_tokens, output_tokens,
        input_cost, output_cost, total_cost, timestamp, metadata
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const result = stmt.run(
      record.request_id,
      record.model_id,
      record.input_tokens,
      record.output_tokens,
      record.input_cost,
      record.output_cost,
      record.total_cost,
      record.timestamp.toISOString(),
      record.metadata || null
    );

    return result.lastInsertRowid as number;
  }

  getUsageByDateRange(startDate: Date, endDate: Date): UsageRecord[] {
    const stmt = this.db.prepare(`
      SELECT * FROM usage_history
      WHERE timestamp >= ? AND timestamp <= ?
      ORDER BY timestamp DESC
    `);

    const rows = stmt.all(startDate.toISOString(), endDate.toISOString()) as any[];

    return rows.map((row) => ({
      ...row,
      timestamp: new Date(row.timestamp),
      created_at: new Date(row.created_at),
    }));
  }

  getTotalCostByDateRange(startDate: Date, endDate: Date): number {
    const stmt = this.db.prepare(`
      SELECT SUM(total_cost) as total
      FROM usage_history
      WHERE timestamp >= ? AND timestamp <= ?
    `);

    const result = stmt.get(startDate.toISOString(), endDate.toISOString()) as { total: number | null };
    return result.total || 0;
  }

  getUsageByModel(startDate: Date, endDate: Date): Record<string, {
    requests: number;
    inputTokens: number;
    outputTokens: number;
    totalCost: number;
  }> {
    const stmt = this.db.prepare(`
      SELECT
        model_id,
        COUNT(*) as requests,
        SUM(input_tokens) as input_tokens,
        SUM(output_tokens) as output_tokens,
        SUM(total_cost) as total_cost
      FROM usage_history
      WHERE timestamp >= ? AND timestamp <= ?
      GROUP BY model_id
    `);

    const rows = stmt.all(startDate.toISOString(), endDate.toISOString()) as any[];

    return rows.reduce((acc, row) => {
      acc[row.model_id] = {
        requests: row.requests,
        inputTokens: row.input_tokens,
        outputTokens: row.output_tokens,
        totalCost: row.total_cost,
      };
      return acc;
    }, {} as Record<string, any>);
  }

  close() {
    this.db.close();
  }
}

export const db = new DatabaseManager();
