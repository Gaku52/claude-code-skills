/**
 * API テストのセットアップファイル
 * 全てのAPIテスト実行前に実行されます
 */

// テスト環境変数の設定
process.env.NODE_ENV = 'test';
process.env.PORT = '0'; // ランダムポート

// データベース接続の例（使用するDBに応じて変更）
// import mongoose from 'mongoose';
// import { Pool } from 'pg';

/**
 * テスト用データベースの接続情報
 */
const TEST_DB_CONFIG = {
  // MongoDB の例
  // mongoUri: process.env.MONGO_TEST_URL || 'mongodb://localhost:27017/test_db',

  // PostgreSQL の例
  // postgres: {
  //   host: 'localhost',
  //   port: 5432,
  //   database: 'test_db',
  //   user: 'test_user',
  //   password: 'test_password',
  // },
};

/**
 * 全テスト開始前の処理
 */
beforeAll(async () => {
  console.log('🚀 Starting API tests...');

  // データベース接続
  // MongoDB の例
  // await mongoose.connect(TEST_DB_CONFIG.mongoUri);

  // PostgreSQL の例
  // global.db = new Pool(TEST_DB_CONFIG.postgres);
  // await global.db.query('SELECT 1'); // 接続確認
});

/**
 * 全テスト終了後の処理
 */
afterAll(async () => {
  console.log('✅ API tests completed');

  // データベース切断
  // MongoDB の例
  // await mongoose.connection.close();

  // PostgreSQL の例
  // await global.db.end();
});

/**
 * 各テストスイート開始前の処理
 */
beforeEach(async () => {
  // テストデータのクリーンアップ
  // await clearTestData();
});

/**
 * 各テストスイート終了後の処理
 */
afterEach(async () => {
  // 必要に応じてクリーンアップ
  // jest.clearAllMocks();
});

/**
 * テストデータのクリーンアップ
 */
async function clearTestData() {
  // MongoDB の例
  // const collections = await mongoose.connection.db.collections();
  // for (const collection of collections) {
  //   await collection.deleteMany({});
  // }

  // PostgreSQL の例
  // await global.db.query('TRUNCATE TABLE users CASCADE');
  // await global.db.query('TRUNCATE TABLE posts CASCADE');
}

/**
 * TypeScript グローバル型定義
 */
declare global {
  namespace NodeJS {
    interface Global {
      db: any; // データベースインスタンス
    }
  }
}

export {};
