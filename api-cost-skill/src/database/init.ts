#!/usr/bin/env node

import { db } from './index';

// データベースはインスタンス作成時に自動的に初期化される
// dbをインポートするだけで初期化が完了する
console.log('[Init] Database initialized successfully');
