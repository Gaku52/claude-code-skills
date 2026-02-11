/** @type {import('jest').Config} */
module.exports = {
  // プリセット: TypeScript + Jest
  preset: 'ts-jest',

  // テスト環境
  testEnvironment: 'jsdom',

  // セットアップファイル
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],

  // モジュールパスのエイリアス
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '\\.(jpg|jpeg|png|gif|svg)$': '<rootDir>/__mocks__/fileMock.js',
  },

  // テストファイルのパターン
  testMatch: [
    '**/__tests__/**/*.(test|spec).(ts|tsx|js|jsx)',
    '**/*.(test|spec).(ts|tsx|js|jsx)',
  ],

  // カバレッジ収集対象
  collectCoverageFrom: [
    'src/**/*.{ts,tsx,js,jsx}',
    '!src/**/*.d.ts',
    '!src/**/*.stories.{ts,tsx}',
    '!src/**/__tests__/**',
    '!src/main.tsx',
    '!src/vite-env.d.ts',
  ],

  // カバレッジ除外パターン
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/build/',
    '/.next/',
  ],

  // カバレッジ最低基準
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
    // クリティカルなディレクトリは高い基準
    './src/core/**': {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90,
    },
  },

  // カバレッジレポート形式
  coverageReporters: [
    'text',        // ターミナル出力
    'text-summary', // サマリー
    'lcov',        // CI用
    'html',        // ブラウザで確認
  ],

  // Transform設定
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: {
        jsx: 'react-jsx',
      },
    }],
  },

  // モジュールファイル拡張子
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json'],

  // テストタイムアウト（ミリ秒）
  testTimeout: 5000,

  // 並列実行の最大ワーカー数
  maxWorkers: '50%',

  // クリアモック設定
  clearMocks: true,
  resetMocks: false,
  restoreMocks: true,

  // verbose モード（詳細出力）
  verbose: true,

  // エラー時のスタックトレース表示
  errorOnDeprecated: true,

  // グローバル変数
  globals: {
    'ts-jest': {
      isolatedModules: true,
    },
  },
};
