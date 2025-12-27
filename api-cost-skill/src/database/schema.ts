export const SCHEMA = `
CREATE TABLE IF NOT EXISTS usage_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  input_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  input_cost REAL NOT NULL,
  output_cost REAL NOT NULL,
  total_cost REAL NOT NULL,
  timestamp DATETIME NOT NULL,
  metadata TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_timestamp ON usage_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_model_id ON usage_history(model_id);
CREATE INDEX IF NOT EXISTS idx_request_id ON usage_history(request_id);
`;

export interface UsageRecord {
  id?: number;
  request_id: string;
  model_id: string;
  input_tokens: number;
  output_tokens: number;
  input_cost: number;
  output_cost: number;
  total_cost: number;
  timestamp: Date;
  metadata?: string;
  created_at?: Date;
}
