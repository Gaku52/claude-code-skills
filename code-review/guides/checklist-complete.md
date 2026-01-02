# コードレビューチェックリスト 完全ガイド
**作成日**: 2025年1月
**対象**: 全エンジニア
**カバレッジ**: TypeScript, Python, Swift, Go, 一般的な観点

---

## 目次

1. [総合チェックリスト](#1-総合チェックリスト)
2. [TypeScript/JavaScriptチェックリスト](#2-typescriptjavascriptチェックリスト)
3. [Pythonチェックリスト](#3-pythonチェックリスト)
4. [Swiftチェックリスト](#4-swiftチェックリスト)
5. [Goチェックリスト](#5-goチェックリスト)
6. [セキュリティチェックリスト](#6-セキュリティチェックリスト)
7. [パフォーマンスチェックリスト](#7-パフォーマンスチェックリスト)
8. [テストチェックリスト](#8-テストチェックリスト)
9. [アーキテクチャチェックリスト](#9-アーキテクチャチェックリスト)
10. [ドキュメントチェックリスト](#10-ドキュメントチェックリスト)

---

## 1. 総合チェックリスト

### 1.1 機能性チェックリスト

```markdown
## 機能性レビュー項目

### 要件充足
- [ ] 全ての要件を満たしているか
- [ ] 受け入れ基準を満たしているか
- [ ] ユーザーストーリーに沿っているか
- [ ] 設計書/仕様書と一致しているか

### エッジケース
- [ ] 空データ/null/undefined の処理
- [ ] 大量データの処理
- [ ] 境界値の処理（最小値、最大値）
- [ ] 不正な入力の処理
- [ ] タイムアウト/ネットワークエラーの処理

### エラーハンドリング
- [ ] 全てのエラーケースを処理しているか
- [ ] エラーメッセージが明確か
- [ ] エラーログが適切に記録されるか
- [ ] ユーザーにわかりやすいエラー表示か
- [ ] リカバリー処理が実装されているか

### データ整合性
- [ ] データバリデーションが実装されているか
- [ ] データの一貫性が保たれるか
- [ ] トランザクション処理が適切か
- [ ] ロールバック処理が実装されているか
```

### 1.2 コード品質チェックリスト

```markdown
## コード品質レビュー項目

### 可読性
- [ ] 変数名が明確で意味が分かるか
- [ ] 関数名が動作を表しているか
- [ ] クラス名が責務を表しているか
- [ ] マジックナンバーを避けているか
- [ ] ネストが深すぎないか（4階層以下）
- [ ] 1行が長すぎないか（80-120文字）

### 関数設計
- [ ] 関数が単一責任を持っているか
- [ ] 関数が短い（20-50行程度）か
- [ ] 引数の数が適切か（3個以下推奨）
- [ ] 関数の戻り値が明確か
- [ ] 副作用が最小限か

### クラス設計
- [ ] 単一責任原則を守っているか
- [ ] 開放/閉鎖原則を守っているか
- [ ] インターフェース分離原則を守っているか
- [ ] 依存性逆転原則を守っているか

### DRY原則
- [ ] 重複コードがないか
- [ ] 共通処理が抽出されているか
- [ ] ユーティリティ関数が適切に使われているか

### コメント
- [ ] Whyを説明するコメントがあるか
- [ ] 複雑なロジックに説明があるか
- [ ] 古いコメントが削除されているか
- [ ] TODOコメントに期限と担当者があるか
```

### 1.3 保守性チェックリスト

```markdown
## 保守性レビュー項目

### 拡張性
- [ ] 新機能追加が容易か
- [ ] 変更の影響範囲が限定的か
- [ ] 適切な抽象化がされているか
- [ ] プラグイン/拡張ポイントがあるか

### モジュール性
- [ ] 適切にモジュール分割されているか
- [ ] モジュール間の依存関係が明確か
- [ ] 循環依存がないか
- [ ] 凝集度が高く、結合度が低いか

### 設定の外部化
- [ ] ハードコードされた設定がないか
- [ ] 環境変数が適切に使われているか
- [ ] 設定ファイルが適切か
- [ ] 秘密情報が環境変数化されているか

### バージョン管理
- [ ] 破壊的変更が文書化されているか
- [ ] APIバージョニングが適切か
- [ ] マイグレーションパスが明確か
```

---

## 2. TypeScript/JavaScriptチェックリスト

### 2.1 TypeScript型安全性チェックリスト

```typescript
// TypeScript 型安全性チェックリスト

// ✅ チェック項目
const typeScriptChecks = {
  '型定義': [
    '`any` を避けているか',
    '`unknown` を適切に使っているか',
    'Type guards を実装しているか',
    'Union types を適切に使っているか',
    'Intersection types を適切に使っているか',
  ],

  'Generics': [
    'Generics が適切に使われているか',
    '型制約が適切か',
    'デフォルト型パラメータが適切か',
  ],

  'Utility Types': [
    'Partial, Required, Readonly などを活用しているか',
    'Pick, Omit を適切に使っているか',
    'ReturnType, Parameters を活用しているか',
  ],

  'Null Safety': [
    'null/undefined チェックが適切か',
    'Optional chaining (?.) を使っているか',
    'Nullish coalescing (??) を使っているか',
    'Non-null assertion (!) を避けているか',
  ],
};

// ❌ 悪い例
function processData(data: any) {
  return data.value; // any を使用
}

const name = user!.name; // Non-null assertion

// ✅ 良い例
interface Data {
  value: string;
}

function processData(data: Data): string {
  return data.value;
}

const name = user?.name ?? 'Unknown';
```

### 2.2 React/JSXチェックリスト

```tsx
// React/JSX チェックリスト

const reactChecks = {
  'Hooks': [
    'useState の初期値が適切か',
    'useEffect の依存配列が正しいか',
    'useMemo でメモ化すべきか',
    'useCallback でメモ化すべきか',
    'カスタムフックが再利用可能か',
  ],

  'Component Design': [
    'コンポーネントが小さく保たれているか',
    'Props の型が定義されているか',
    'デフォルトPropsが適切か',
    'コンポーネントが純粋関数か',
  ],

  'Performance': [
    '不要な再レンダリングがないか',
    'React.memo が適切に使われているか',
    'key プロパティが適切か',
    '仮想化（virtualization）が必要か',
  ],

  'Accessibility': [
    'セマンティックHTMLを使っているか',
    'ARIA属性が適切か',
    'キーボード操作が可能か',
    'スクリーンリーダー対応されているか',
  ],
};

// ❌ 悪い例
function UserList({ users }) {
  // 毎回再計算
  const sortedUsers = users.sort((a, b) => a.name.localeCompare(b.name));

  return (
    <div>
      {sortedUsers.map(user => (
        <div>{user.name}</div> // key がない
      ))}
    </div>
  );
}

// ✅ 良い例
interface UserListProps {
  users: User[];
}

function UserList({ users }: UserListProps) {
  // useMemo でメモ化
  const sortedUsers = useMemo(
    () => users.sort((a, b) => a.name.localeCompare(b.name)),
    [users]
  );

  return (
    <ul>
      {sortedUsers.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

### 2.3 非同期処理チェックリスト

```typescript
// 非同期処理チェックリスト

const asyncChecks = {
  'Promise': [
    'async/await を使っているか',
    'エラーハンドリングがあるか（try-catch）',
    'Promise.all で並行処理しているか',
    'Promise.allSettled を適切に使っているか',
  ],

  'エラーハンドリング': [
    '全ての非同期処理にエラーハンドリングがあるか',
    'エラーを適切に伝搬しているか',
    'リトライロジックが必要か',
    'タイムアウト処理があるか',
  ],
};

// ❌ 悪い例
async function fetchData() {
  const data = await fetch('/api/data'); // エラーハンドリングなし
  return data.json();
}

// Promise チェーン（読みにくい）
getData()
  .then(data => processData(data))
  .then(result => saveResult(result))
  .catch(err => console.error(err));

// ✅ 良い例
async function fetchData(): Promise<Data> {
  try {
    const response = await fetch('/api/data');

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    logger.error('Failed to fetch data', { error });
    throw new FetchError('Data fetch failed', { cause: error });
  }
}

// async/await（読みやすい）
async function processFlow() {
  try {
    const data = await getData();
    const result = await processData(data);
    await saveResult(result);
  } catch (error) {
    logger.error('Processing failed', { error });
    throw error;
  }
}

// 並行処理
async function fetchMultiple() {
  try {
    const [users, posts, comments] = await Promise.all([
      fetchUsers(),
      fetchPosts(),
      fetchComments(),
    ]);

    return { users, posts, comments };
  } catch (error) {
    logger.error('Parallel fetch failed', { error });
    throw error;
  }
}
```

### 2.4 Node.js バックエンドチェックリスト

```typescript
// Node.js バックエンドチェックリスト

const nodeChecks = {
  'セキュリティ': [
    '入力バリデーションが実装されているか',
    'SQLインジェクション対策があるか',
    'XSS対策があるか',
    'CSRF対策があるか',
    'レート制限が実装されているか',
  ],

  'エラーハンドリング': [
    'グローバルエラーハンドラーがあるか',
    'エラーログが適切に記録されるか',
    'エラーレスポンスが統一されているか',
    '本番環境でスタックトレースを隠しているか',
  ],

  'パフォーマンス': [
    'N+1クエリを避けているか',
    'コネクションプールを使っているか',
    'キャッシュを適切に使っているか',
    '非同期処理を活用しているか',
  ],
};

// ✅ Express.js エラーハンドリング例
import express, { Request, Response, NextFunction } from 'express';

// カスタムエラークラス
class AppError extends Error {
  constructor(
    public statusCode: number,
    message: string,
    public isOperational = true
  ) {
    super(message);
    Object.setPrototypeOf(this, AppError.prototype);
  }
}

// グローバルエラーハンドラー
function errorHandler(
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction
) {
  // ログ記録
  logger.error('Error occurred', {
    error: err.message,
    stack: err.stack,
    url: req.url,
    method: req.method,
  });

  // エラーレスポンス
  if (err instanceof AppError && err.isOperational) {
    // 運用エラー（予期されたエラー）
    res.status(err.statusCode).json({
      status: 'error',
      message: err.message,
    });
  } else {
    // プログラミングエラー（予期しないエラー）
    res.status(500).json({
      status: 'error',
      message: process.env.NODE_ENV === 'production'
        ? 'Internal server error'
        : err.message,
    });
  }
}

// 使用例
app.get('/users/:id', async (req, res, next) => {
  try {
    const user = await userService.findById(req.params.id);

    if (!user) {
      throw new AppError(404, 'User not found');
    }

    res.json(user);
  } catch (error) {
    next(error);
  }
});

app.use(errorHandler);
```

---

## 3. Pythonチェックリスト

### 3.1 Python型ヒントチェックリスト

```python
# Python 型ヒントチェックリスト

from typing import Optional, List, Dict, Union, TypeVar, Generic

# ✅ チェック項目
python_type_checks = {
    '型ヒント': [
        '関数の引数に型ヒントがあるか',
        '関数の戻り値に型ヒントがあるか',
        'Optional が適切に使われているか',
        'Union types が適切に使われているか',
    ],
    'mypy': [
        'mypy でチェックしているか',
        'strict モードを使っているか',
        '型エラーがないか',
    ],
}

# ❌ 悪い例
def get_user(user_id):  # 型ヒントなし
    return database.get(user_id)

def process_data(data):
    return data.upper()

# ✅ 良い例
def get_user(user_id: int) -> Optional[User]:
    """ユーザーIDからユーザーを取得

    Args:
        user_id: ユーザーID

    Returns:
        User object or None if not found
    """
    return database.get(user_id)

def process_data(data: str) -> str:
    """データを大文字に変換

    Args:
        data: 入力文字列

    Returns:
        大文字に変換された文字列
    """
    return data.upper()

# Generics の使用
T = TypeVar('T')

class Repository(Generic[T]):
    def __init__(self, model: type[T]) -> None:
        self.model = model

    def find_by_id(self, id: int) -> Optional[T]:
        # 実装
        pass

    def find_all(self) -> List[T]:
        # 実装
        pass
```

### 3.2 Pythonエラーハンドリングチェックリスト

```python
# Pythonエラーハンドリングチェックリスト

from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ✅ チェック項目
error_handling_checks = {
    '例外処理': [
        '具体的な例外をキャッチしているか',
        '例外を握りつぶしていないか',
        'カスタム例外を定義しているか',
        '例外チェーンを使っているか',
    ],
    'ログ': [
        '適切なログレベルを使っているか',
        'エラー情報が十分に記録されるか',
        'スタックトレースが記録されるか',
    ],
}

# ❌ 悪い例
def process_file(filename):
    try:
        with open(filename) as f:
            data = f.read()
            return data
    except:  # 全ての例外をキャッチ
        pass  # 例外を握りつぶす

# ✅ 良い例
class FileProcessingError(Exception):
    """ファイル処理エラー"""
    pass

def process_file(filename: str) -> Optional[str]:
    """ファイルを処理

    Args:
        filename: ファイルパス

    Returns:
        ファイル内容

    Raises:
        FileProcessingError: ファイル処理に失敗した場合
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = f.read()
            return data
    except FileNotFoundError as e:
        logger.error(f"File not found: {filename}", exc_info=True)
        raise FileProcessingError(f"File not found: {filename}") from e
    except PermissionError as e:
        logger.error(f"Permission denied: {filename}", exc_info=True)
        raise FileProcessingError(f"Permission denied: {filename}") from e
    except Exception as e:
        logger.error(f"Unexpected error processing file: {filename}", exc_info=True)
        raise FileProcessingError(f"Failed to process file: {filename}") from e

# コンテキストマネージャーの使用
class DatabaseConnection:
    def __enter__(self):
        self.conn = connect_to_database()
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
        # 例外を再発生させる
        return False

# 使用例
try:
    with DatabaseConnection() as conn:
        result = conn.execute("SELECT * FROM users")
except DatabaseError as e:
    logger.error("Database query failed", exc_info=True)
    raise
```

### 3.3 Pythonコードスタイルチェックリスト

```python
# Python コードスタイルチェックリスト (PEP 8)

# ✅ チェック項目
style_checks = {
    '命名規則': [
        '関数名は snake_case か',
        'クラス名は PascalCase か',
        '定数名は UPPER_CASE か',
        'プライベート変数は _leading_underscore か',
    ],
    'フォーマット': [
        'インデントは4スペースか',
        '1行は79文字以内か',
        '空行が適切に配置されているか',
        'import文が適切に整理されているか',
    ],
    'Docstring': [
        'モジュールに docstring があるか',
        'クラスに docstring があるか',
        '関数に docstring があるか',
        'Google/NumPy スタイルに従っているか',
    ],
}

# ❌ 悪い例
def processData(UserID):  # キャメルケース、引数もパスカルケース
    result=database.query(UserID)  # スペースなし
    return result

class user_service:  # スネークケース（クラスはパスカルケース）
    def GetUser(self,id):  # キャメルケース、スペースなし
        pass

# ✅ 良い例
from typing import Optional

MAX_RETRY_COUNT = 3  # 定数は大文字

class UserService:
    """ユーザーサービスクラス

    ユーザー関連の操作を提供します。

    Attributes:
        _database: データベース接続
    """

    def __init__(self, database: Database) -> None:
        """初期化

        Args:
            database: データベース接続
        """
        self._database = database  # プライベート変数

    def get_user(self, user_id: int) -> Optional[User]:
        """ユーザーを取得

        Args:
            user_id: ユーザーID

        Returns:
            User object or None if not found

        Raises:
            DatabaseError: データベースエラーが発生した場合
        """
        try:
            result = self._database.query(user_id)
            return result
        except DatabaseError as e:
            logger.error(f"Failed to get user {user_id}", exc_info=True)
            raise

# Import の整理
# 1. 標準ライブラリ
import os
import sys
from typing import List, Optional

# 2. サードパーティ
import numpy as np
import pandas as pd

# 3. ローカル
from myapp.models import User
from myapp.services import UserService
```

### 3.4 FastAPI/Djangoチェックリスト

```python
# FastAPI チェックリスト

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Optional

# ✅ チェック項目
fastapi_checks = {
    'モデル定義': [
        'Pydantic モデルを使っているか',
        'バリデーションが適切か',
        'OpenAPI ドキュメントが生成されるか',
    ],
    'エラーハンドリング': [
        'HTTPException を使っているか',
        'カスタムエラーハンドラーがあるか',
        'エラーレスポンスが統一されているか',
    ],
    '依存性注入': [
        'Depends を使っているか',
        'データベースセッションが適切に管理されているか',
        '認証が実装されているか',
    ],
}

# ✅ FastAPI ベストプラクティス
class UserCreate(BaseModel):
    """ユーザー作成リクエスト"""
    email: str = Field(..., description="メールアドレス")
    password: str = Field(..., min_length=8, description="パスワード（8文字以上）")
    name: str = Field(..., max_length=100, description="ユーザー名")

    @validator('email')
    def validate_email(cls, v: str) -> str:
        """メールアドレスのバリデーション"""
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v.lower()

class UserResponse(BaseModel):
    """ユーザーレスポンス"""
    id: int
    email: str
    name: str

    class Config:
        from_attributes = True  # ORMモードを有効化

app = FastAPI(
    title="User API",
    description="ユーザー管理API",
    version="1.0.0"
)

# 依存性注入
def get_db():
    """データベースセッションを取得"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# エラーハンドラー
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """バリデーションエラーハンドラー"""
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)}
    )

# エンドポイント
@app.post("/users/", response_model=UserResponse, status_code=201)
async def create_user(
    user: UserCreate,
    db: Session = Depends(get_db)
) -> UserResponse:
    """ユーザーを作成

    Args:
        user: ユーザー作成リクエスト
        db: データベースセッション

    Returns:
        作成されたユーザー

    Raises:
        HTTPException: ユーザーが既に存在する場合
    """
    # メールアドレスの重複チェック
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # ユーザー作成
    db_user = User(
        email=user.email,
        password=hash_password(user.password),
        name=user.name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user
```

---

## 4. Swiftチェックリスト

### 4.1 Swiftメモリ管理チェックリスト

```swift
// Swift メモリ管理チェックリスト

// ✅ チェック項目
let memoryChecks = [
    "Optional": [
        "強制アンラップを避けているか",
        "Optional Binding を使っているか",
        "Optional Chaining を使っているか",
        "nil coalescing を適切に使っているか",
    ],
    "メモリリーク": [
        "循環参照を避けているか",
        "[weak self] を適切に使っているか",
        "[unowned self] を慎重に使っているか",
        "Delegate は weak か",
    ],
    "Combine": [
        "AnyCancellable を適切に保持しているか",
        "メモリリークがないか",
    ],
]

// ❌ 悪い例
class ViewController: UIViewController {
    var closure: (() -> Void)?
    var delegate: SomeDelegate?  // strong 参照

    func setup() {
        // 循環参照
        closure = {
            self.doSomething()
        }

        // 強制アンラップ
        let name = user!.name
    }
}

// ✅ 良い例
protocol SomeDelegate: AnyObject {  // class-only protocol
    func didUpdate()
}

class ViewController: UIViewController {
    var closure: (() -> Void)?
    weak var delegate: SomeDelegate?  // weak 参照

    func setup() {
        // [weak self] で循環参照を回避
        closure = { [weak self] in
            guard let self = self else { return }
            self.doSomething()
        }

        // Optional Binding
        guard let user = user else {
            return
        }
        let name = user.name

        // または nil coalescing
        let userName = user?.name ?? "Unknown"
    }
}

// Combine のメモリ管理
import Combine

class DataService {
    private var cancellables = Set<AnyCancellable>()

    func fetchData() {
        URLSession.shared.dataTaskPublisher(for: url)
            .map(\.data)
            .decode(type: Response.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { [weak self] completion in
                    // weak self を使用
                    self?.handleCompletion(completion)
                },
                receiveValue: { [weak self] value in
                    self?.handleValue(value)
                }
            )
            .store(in: &cancellables)  // 保持を忘れない
    }
}
```

### 4.2 SwiftUIチェックリスト

```swift
// SwiftUI チェックリスト

import SwiftUI

// ✅ チェック項目
let swiftUIChecks = [
    "State Management": [
        "@State を適切に使っているか",
        "@Binding を適切に使っているか",
        "@ObservedObject/@StateObject を正しく使い分けているか",
        "@EnvironmentObject を適切に使っているか",
    ],
    "Performance": [
        "不要な再描画を避けているか",
        "GeometryReader を適切に使っているか",
        "LazyVStack/LazyHStack を使っているか",
    ],
    "Modifiers": [
        "Modifier の順序が適切か",
        "カスタム Modifier を抽出しているか",
    ],
]

// ❌ 悪い例
struct UserListView: View {
    @State private var users: [User] = []

    var body: some View {
        VStack {
            // パフォーマンス問題: 毎回ソート
            ForEach(users.sorted(by: { $0.name < $1.name })) { user in
                Text(user.name)
            }
        }
        .onAppear {
            loadUsers()
        }
    }
}

// ✅ 良い例
struct UserListView: View {
    @StateObject private var viewModel = UserListViewModel()

    var body: some View {
        List {
            // LazyVStack を使用
            ForEach(viewModel.sortedUsers) { user in
                UserRow(user: user)
            }
        }
        .onAppear {
            viewModel.loadUsers()
        }
    }
}

class UserListViewModel: ObservableObject {
    @Published private(set) var users: [User] = []

    // Computed property でソート（キャッシュされる）
    var sortedUsers: [User] {
        users.sorted(by: { $0.name < $1.name })
    }

    func loadUsers() {
        // ユーザー読み込み
    }
}

// カスタム Modifier
struct CardStyle: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding()
            .background(Color.white)
            .cornerRadius(8)
            .shadow(radius: 2)
    }
}

extension View {
    func cardStyle() -> some View {
        modifier(CardStyle())
    }
}

// 使用例
Text("Hello")
    .cardStyle()
```

### 4.3 Swiftエラーハンドリングチェックリスト

```swift
// Swift エラーハンドリングチェックリスト

import Foundation

// ✅ チェック項目
let errorHandlingChecks = [
    "Error型": [
        "カスタムエラーを定義しているか",
        "LocalizedError を実装しているか",
        "適切なエラー情報を提供しているか",
    ],
    "throws": [
        "throws 関数を適切に扱っているか",
        "do-catch でエラーを処理しているか",
        "try? / try! を適切に使い分けているか",
    ],
    "Result型": [
        "Result<Success, Failure> を使っているか",
        "非同期処理のエラーハンドリングが適切か",
    ],
]

// ❌ 悪い例
func loadData() -> Data? {
    // エラー情報が失われる
    return try? Data(contentsOf: url)
}

// ✅ 良い例

// カスタムエラー定義
enum NetworkError: Error {
    case invalidURL
    case noData
    case decodingFailed(Error)
    case serverError(statusCode: Int)
}

// LocalizedError 実装
extension NetworkError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "無効なURLです"
        case .noData:
            return "データが取得できませんでした"
        case .decodingFailed(let error):
            return "データのデコードに失敗しました: \(error.localizedDescription)"
        case .serverError(let statusCode):
            return "サーバーエラー (ステータスコード: \(statusCode))"
        }
    }
}

// Result型を使用
func fetchData(from urlString: String) -> Result<Data, NetworkError> {
    guard let url = URL(string: urlString) else {
        return .failure(.invalidURL)
    }

    do {
        let data = try Data(contentsOf: url)
        return .success(data)
    } catch {
        return .failure(.noData)
    }
}

// async/await with throws
func fetchDataAsync(from urlString: String) async throws -> Data {
    guard let url = URL(string: urlString) else {
        throw NetworkError.invalidURL
    }

    let (data, response) = try await URLSession.shared.data(from: url)

    guard let httpResponse = response as? HTTPURLResponse else {
        throw NetworkError.noData
    }

    guard (200...299).contains(httpResponse.statusCode) else {
        throw NetworkError.serverError(statusCode: httpResponse.statusCode)
    }

    return data
}

// 使用例
Task {
    do {
        let data = try await fetchDataAsync(from: "https://api.example.com/data")
        // データ処理
    } catch let error as NetworkError {
        // 具体的なエラー処理
        print("Network error: \(error.localizedDescription)")
    } catch {
        // その他のエラー
        print("Unexpected error: \(error)")
    }
}
```

---

## 5. Goチェックリスト

### 5.1 Goエラーハンドリングチェックリスト

```go
// Go エラーハンドリングチェックリスト

package main

import (
    "errors"
    "fmt"
    "log"
)

// ✅ チェック項目
var errorChecks = map[string][]string{
    "エラー処理": {
        "エラーを無視していないか",
        "エラーを適切に返しているか",
        "エラーをラップしているか（Go 1.13+）",
        "カスタムエラーを定義しているか",
    },
    "エラーメッセージ": {
        "エラーメッセージが明確か",
        "コンテキスト情報を含んでいるか",
        "大文字で始まっていないか",
    },
}

// ❌ 悪い例
func badExample() {
    data, err := readFile("file.txt")
    // エラーを無視
    fmt.Println(data)
}

// ✅ 良い例

// カスタムエラー定義
type FileError struct {
    Filename string
    Err      error
}

func (e *FileError) Error() string {
    return fmt.Sprintf("file error: %s: %v", e.Filename, e.Err)
}

func (e *FileError) Unwrap() error {
    return e.Err
}

// エラーハンドリング
func readFile(filename string) ([]byte, error) {
    data, err := os.ReadFile(filename)
    if err != nil {
        // エラーをラップ
        return nil, &FileError{
            Filename: filename,
            Err:      err,
        }
    }
    return data, nil
}

// エラーチェック
func processFile(filename string) error {
    data, err := readFile(filename)
    if err != nil {
        // エラーをラップして返す
        return fmt.Errorf("failed to process file: %w", err)
    }

    // 処理
    if err := validateData(data); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }

    return nil
}

// エラーの型確認
func handleError(err error) {
    var fileErr *FileError
    if errors.As(err, &fileErr) {
        log.Printf("File error occurred: %s", fileErr.Filename)
    } else if errors.Is(err, os.ErrNotExist) {
        log.Println("File does not exist")
    } else {
        log.Printf("Unknown error: %v", err)
    }
}
```

### 5.2 Go並行処理チェックリスト

```go
// Go 並行処理チェックリスト

package main

import (
    "context"
    "sync"
    "time"
)

// ✅ チェック項目
var concurrencyChecks = map[string][]string{
    "Goroutine": {
        "goroutine リークがないか",
        "panic が適切に処理されているか",
        "context を使っているか",
    },
    "Channel": {
        "チャネルが適切にクローズされているか",
        "デッドロックの可能性がないか",
        "バッファサイズが適切か",
    },
    "同期": {
        "sync.Mutex を適切に使っているか",
        "sync.WaitGroup を使っているか",
        "データ競合がないか",
    },
}

// ❌ 悪い例
func badConcurrency() {
    // goroutine リーク
    go func() {
        for {
            // 終了条件なし
            doWork()
        }
    }()
}

// ✅ 良い例

// Context を使った goroutine 管理
func goodConcurrency(ctx context.Context) {
    go func() {
        for {
            select {
            case <-ctx.Done():
                // goroutine を適切に終了
                return
            default:
                doWork()
            }
        }
    }()
}

// Worker Pool パターン
type WorkerPool struct {
    workers int
    jobs    chan Job
    results chan Result
    wg      sync.WaitGroup
}

func NewWorkerPool(workers int) *WorkerPool {
    return &WorkerPool{
        workers: workers,
        jobs:    make(chan Job, 100),    // バッファありチャネル
        results: make(chan Result, 100),
    }
}

func (p *WorkerPool) Start(ctx context.Context) {
    for i := 0; i < p.workers; i++ {
        p.wg.Add(1)
        go p.worker(ctx, i)
    }
}

func (p *WorkerPool) worker(ctx context.Context, id int) {
    defer p.wg.Done()

    for {
        select {
        case <-ctx.Done():
            return
        case job, ok := <-p.jobs:
            if !ok {
                return  // チャネルがクローズされた
            }

            // panic を処理
            func() {
                defer func() {
                    if r := recover(); r != nil {
                        log.Printf("Worker %d panicked: %v", id, r)
                    }
                }()

                result := processJob(job)
                p.results <- result
            }()
        }
    }
}

func (p *WorkerPool) Stop() {
    close(p.jobs)    // ジョブチャネルをクローズ
    p.wg.Wait()      // 全ワーカーの終了を待つ
    close(p.results) // 結果チャネルをクローズ
}

// 使用例
func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    pool := NewWorkerPool(5)
    pool.Start(ctx)

    // ジョブを送信
    go func() {
        for i := 0; i < 100; i++ {
            pool.jobs <- Job{ID: i}
        }
    }()

    // 結果を受信
    go func() {
        for result := range pool.results {
            processResult(result)
        }
    }()

    // タイムアウトまたは完了を待つ
    <-ctx.Done()
    pool.Stop()
}
```

### 5.3 Goテストチェックリスト

```go
// Go テストチェックリスト

package main

import (
    "testing"
    "time"
)

// ✅ チェック項目
var testChecks = map[string][]string{
    "テスト構造": {
        "Table-Driven Tests を使っているか",
        "サブテストを使っているか",
        "テストが独立しているか",
    },
    "モック": {
        "インターフェースを使っているか",
        "テスト用のモックがあるか",
    },
    "ベンチマーク": {
        "パフォーマンステストがあるか",
        "ベンチマークが適切か",
    },
}

// ✅ Table-Driven Tests
func TestAdd(t *testing.T) {
    tests := []struct {
        name     string
        a, b     int
        expected int
    }{
        {"positive numbers", 2, 3, 5},
        {"negative numbers", -2, -3, -5},
        {"zero", 0, 0, 0},
        {"mixed", -5, 10, 5},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := Add(tt.a, tt.b)
            if result != tt.expected {
                t.Errorf("Add(%d, %d) = %d, want %d",
                    tt.a, tt.b, result, tt.expected)
            }
        })
    }
}

// ✅ インターフェースとモック
type UserRepository interface {
    GetUser(id int) (*User, error)
    SaveUser(user *User) error
}

// モック実装
type MockUserRepository struct {
    users map[int]*User
}

func (m *MockUserRepository) GetUser(id int) (*User, error) {
    user, ok := m.users[id]
    if !ok {
        return nil, errors.New("user not found")
    }
    return user, nil
}

func (m *MockUserRepository) SaveUser(user *User) error {
    m.users[user.ID] = user
    return nil
}

// テストでモックを使用
func TestUserService(t *testing.T) {
    // モックを準備
    repo := &MockUserRepository{
        users: map[int]*User{
            1: {ID: 1, Name: "Alice"},
        },
    }

    service := NewUserService(repo)

    // テスト
    user, err := service.GetUser(1)
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }

    if user.Name != "Alice" {
        t.Errorf("got name %s, want Alice", user.Name)
    }
}

// ✅ ベンチマーク
func BenchmarkAdd(b *testing.B) {
    for i := 0; i < b.N; i++ {
        Add(10, 20)
    }
}

func BenchmarkConcurrentAdd(b *testing.B) {
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            Add(10, 20)
        }
    })
}
```

---

## 6. セキュリティチェックリスト

### 6.1 OWASP Top 10チェックリスト

```markdown
## OWASP Top 10 (2021) チェックリスト

### A01:2021 – Broken Access Control
- [ ] 認証が適切に実装されているか
- [ ] 認可チェックがあるか
- [ ] オブジェクトレベルの権限確認があるか
- [ ] CORS 設定が適切か
- [ ] パストラバーサル対策があるか

### A02:2021 – Cryptographic Failures
- [ ] 機密データが暗号化されているか
- [ ] 通信が HTTPS で保護されているか
- [ ] パスワードがハッシュ化されているか（bcrypt, argon2）
- [ ] 古い暗号化アルゴリズムを使っていないか
- [ ] 暗号鍵が安全に管理されているか

### A03:2021 – Injection
- [ ] SQLインジェクション対策（パラメータ化クエリ）
- [ ] コマンドインジェクション対策
- [ ] XPath/LDAP インジェクション対策
- [ ] ORMを適切に使っているか
- [ ] 入力バリデーションが実装されているか

### A04:2021 – Insecure Design
- [ ] セキュアな設計パターンを使っているか
- [ ] 脅威モデリングが実施されているか
- [ ] セキュリティ要件が定義されているか
- [ ] デフォルト設定が安全か

### A05:2021 – Security Misconfiguration
- [ ] デフォルトパスワードを変更しているか
- [ ] 不要な機能が無効化されているか
- [ ] エラーメッセージが適切か（詳細情報を隠す）
- [ ] セキュリティヘッダーが設定されているか
- [ ] ソフトウェアが最新か

### A06:2021 – Vulnerable and Outdated Components
- [ ] 依存関係が最新か
- [ ] 脆弱性スキャンを実施しているか（npm audit, Snyk）
- [ ] 使用していないライブラリを削除しているか
- [ ] ライセンスが確認されているか

### A07:2021 – Identification and Authentication Failures
- [ ] 多要素認証が実装されているか
- [ ] セッション管理が適切か
- [ ] パスワードポリシーが適切か
- [ ] ブルートフォース攻撃対策があるか
- [ ] セッショントークンが予測不可能か

### A08:2021 – Software and Data Integrity Failures
- [ ] コード署名を使っているか
- [ ] CI/CD パイプラインが保護されているか
- [ ] デシリアライゼーションが安全か
- [ ] 依存関係の整合性が検証されているか

### A09:2021 – Security Logging and Monitoring Failures
- [ ] セキュリティイベントがログに記録されるか
- [ ] ログが保護されているか
- [ ] アラート設定があるか
- [ ] ログが定期的にレビューされているか

### A10:2021 – Server-Side Request Forgery (SSRF)
- [ ] URL入力がバリデーションされているか
- [ ] 内部IPアドレスへのアクセスが制限されているか
- [ ] ホワイトリスト方式を使っているか
```

### 6.2 認証・認可チェックリスト

```typescript
// 認証・認可チェックリスト

const authChecks = {
  'パスワード管理': [
    'パスワードがハッシュ化されているか（bcrypt, argon2）',
    'ソルトが使われているか',
    'パスワードポリシーが適切か（最小8文字、複雑性要件）',
    'パスワードリセット機能が安全か',
  ],

  'セッション管理': [
    'セッショントークンが安全に生成されているか',
    'セッションタイムアウトが設定されているか',
    'セッション固定攻撃対策があるか',
    'ログアウト時にセッションが破棄されるか',
  ],

  'JWT': [
    'JWT署名が検証されているか',
    '適切なアルゴリズムを使っているか（HS256/RS256）',
    '有効期限が設定されているか',
    'リフレッシュトークンが実装されているか',
  ],
};

// ❌ 悪い例
import bcrypt from 'bcrypt';

// パスワードを平文で保存
await db.users.create({
  email,
  password: password,  // ❌ 平文
});

// JWT署名を検証しない
const decoded = jwt.decode(token);  // ❌ 検証なし
const user = await getUser(decoded.userId);

// ✅ 良い例

// パスワードをハッシュ化
const saltRounds = 10;
const hashedPassword = await bcrypt.hash(password, saltRounds);

await db.users.create({
  email,
  password: hashedPassword,
});

// パスワード検証
const user = await db.users.findOne({ email });
const isValid = await bcrypt.compare(password, user.password);

if (!isValid) {
  throw new AuthError('Invalid credentials');
}

// JWT 検証
import jwt from 'jsonwebtoken';

function verifyToken(token: string): TokenPayload {
  try {
    const payload = jwt.verify(token, process.env.JWT_SECRET!) as TokenPayload;

    // 有効期限チェック（jwtが自動で行うが明示的に）
    if (payload.exp && payload.exp < Date.now() / 1000) {
      throw new AuthError('Token expired');
    }

    return payload;
  } catch (error) {
    throw new AuthError('Invalid token');
  }
}

// ミドルウェアで認証
async function authenticate(req: Request, res: Response, next: NextFunction) {
  try {
    const token = req.headers.authorization?.replace('Bearer ', '');

    if (!token) {
      return res.status(401).json({ error: 'No token provided' });
    }

    const payload = verifyToken(token);
    req.user = await getUser(payload.userId);

    next();
  } catch (error) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
}

// 認可チェック
function authorize(roles: string[]) {
  return (req: Request, res: Response, next: NextFunction) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    if (!roles.includes(req.user.role)) {
      return res.status(403).json({ error: 'Forbidden' });
    }

    next();
  };
}

// 使用例
app.delete('/users/:id', authenticate, authorize(['admin']), async (req, res) => {
  // 管理者のみアクセス可能
  await deleteUser(req.params.id);
  res.status(204).send();
});
```

---

## 7. パフォーマンスチェックリスト

### 7.1 データベースパフォーマンスチェックリスト

```sql
-- データベースパフォーマンスチェックリスト

-- ✅ チェック項目
/*
インデックス:
- [ ] 検索対象カラムにインデックスがあるか
- [ ] WHERE句のカラムにインデックスがあるか
- [ ] JOIN条件のカラムにインデックスがあるか
- [ ] 複合インデックスの順序が適切か

クエリ最適化:
- [ ] N+1クエリを避けているか
- [ ] SELECT * を避けているか
- [ ] 必要なカラムのみ取得しているか
- [ ] LIMITを使っているか

トランザクション:
- [ ] トランザクションが短いか
- [ ] デッドロックの可能性がないか
- [ ] 適切な分離レベルか
*/

-- ❌ 悪い例

-- N+1クエリ
SELECT * FROM users;  -- 1クエリ
-- 各ユーザーに対して
SELECT * FROM posts WHERE user_id = ?;  -- Nクエリ

-- SELECT *
SELECT * FROM users WHERE id = 1;  -- 不要なカラムも取得

-- インデックスなし
SELECT * FROM users WHERE email = 'test@example.com';  -- フルスキャン

-- ✅ 良い例

-- Eager Loading（JOIN）
SELECT
    u.id,
    u.name,
    u.email,
    p.id AS post_id,
    p.title
FROM users u
LEFT JOIN posts p ON p.user_id = u.id
WHERE u.id = 1;

-- 必要なカラムのみ
SELECT id, name, email
FROM users
WHERE id = 1;

-- インデックス作成
CREATE INDEX idx_users_email ON users(email);

-- 複合インデックス
CREATE INDEX idx_posts_user_created ON posts(user_id, created_at);

-- LIMITを使用
SELECT id, name, email
FROM users
ORDER BY created_at DESC
LIMIT 100;

-- ページネーション
SELECT id, name, email
FROM users
WHERE id > ?  -- 前回の最後のID
ORDER BY id
LIMIT 100;
```

### 7.2 フロントエンドパフォーマンスチェックリスト

```typescript
// フロントエンドパフォーマンスチェックリスト

const frontendPerformanceChecks = {
  'レンダリング': [
    '不要な再レンダリングを避けているか',
    'useMemo/useCallback を使っているか',
    'React.memo を使っているか',
    '仮想化を実装しているか（大量データ）',
  ],

  'バンドルサイズ': [
    'コード分割を実装しているか',
    '遅延ロードを使っているか',
    'Tree shaking が有効か',
    '不要な依存関係を削除しているか',
  ],

  '画像最適化': [
    '画像が最適化されているか',
    '適切なフォーマットを使っているか（WebP）',
    'lazy loading を実装しているか',
    'レスポンシブ画像を使っているか',
  ],

  'キャッシュ': [
    'HTTP キャッシュヘッダーが設定されているか',
    'Service Worker を使っているか',
    'ブラウザキャッシュを活用しているか',
  ],
};

// ❌ 悪い例
function UserList({ users }: { users: User[] }) {
  // 毎回ソート（パフォーマンス問題）
  const sortedUsers = users.sort((a, b) => a.name.localeCompare(b.name));

  return (
    <div>
      {sortedUsers.map(user => (
        <UserCard key={user.id} user={user} />
      ))}
    </div>
  );
}

// ✅ 良い例

// useMemo でメモ化
function UserList({ users }: { users: User[] }) {
  const sortedUsers = useMemo(
    () => users.sort((a, b) => a.name.localeCompare(b.name)),
    [users]
  );

  return (
    <div>
      {sortedUsers.map(user => (
        <MemoizedUserCard key={user.id} user={user} />
      ))}
    </div>
  );
}

// React.memo でメモ化
const MemoizedUserCard = React.memo(UserCard, (prevProps, nextProps) => {
  // 比較関数（オプション）
  return prevProps.user.id === nextProps.user.id &&
         prevProps.user.name === nextProps.user.name;
});

// 仮想化（大量データ）
import { FixedSizeList } from 'react-window';

function VirtualizedUserList({ users }: { users: User[] }) {
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => (
    <div style={style}>
      <UserCard user={users[index]} />
    </div>
  );

  return (
    <FixedSizeList
      height={600}
      itemCount={users.length}
      itemSize={100}
      width="100%"
    >
      {Row}
    </FixedSizeList>
  );
}

// コード分割
import { lazy, Suspense } from 'react';

const HeavyComponent = lazy(() => import('./HeavyComponent'));

function App() {
  return (
    <Suspense fallback={<Loading />}>
      <HeavyComponent />
    </Suspense>
  );
}

// 画像最適化
function OptimizedImage({ src, alt }: { src: string; alt: string }) {
  return (
    <picture>
      <source srcSet={`${src}.webp`} type="image/webp" />
      <source srcSet={`${src}.jpg`} type="image/jpeg" />
      <img
        src={`${src}.jpg`}
        alt={alt}
        loading="lazy"
        decoding="async"
      />
    </picture>
  );
}
```

---

## 8. テストチェックリスト

### 8.1 ユニットテストチェックリスト

```typescript
// ユニットテストチェックリスト

const unitTestChecks = {
  'テスト構造': [
    'AAA（Arrange-Act-Assert）パターンに従っているか',
    'テストが独立しているか',
    'テスト名が明確か',
    '1つのテストで1つの事柄をテストしているか',
  ],

  'カバレッジ': [
    '正常系をテストしているか',
    '異常系をテストしているか',
    'エッジケースをテストしているか',
    '境界値をテストしているか',
  ],

  'モック': [
    '外部依存をモック化しているか',
    'モックが適切に設定されているか',
    'モックの検証が行われているか',
  ],

  'アサーション': [
    'アサーションが明確か',
    'エラーメッセージが分かりやすいか',
    '必要なアサーションが全てあるか',
  ],
};

// ❌ 悪い例
test('test1', () => {
  const result = fn(5);
  expect(result).toBe(10);  // 何をテストしているか不明
});

test('user service', () => {
  // 複数のことをテスト（テストが長い）
  const user = createUser();
  expect(user).toBeDefined();

  const updated = updateUser(user);
  expect(updated).toBeDefined();

  deleteUser(user.id);
  const deleted = getUser(user.id);
  expect(deleted).toBeNull();
});

// ✅ 良い例

describe('UserService', () => {
  // テストデータの準備
  let userService: UserService;
  let mockRepository: jest.Mocked<UserRepository>;

  beforeEach(() => {
    mockRepository = {
      findById: jest.fn(),
      save: jest.fn(),
      delete: jest.fn(),
    } as any;

    userService = new UserService(mockRepository);
  });

  describe('getUser', () => {
    it('should return user when user exists', async () => {
      // Arrange
      const mockUser = { id: 1, name: 'Alice', email: 'alice@example.com' };
      mockRepository.findById.mockResolvedValue(mockUser);

      // Act
      const result = await userService.getUser(1);

      // Assert
      expect(result).toEqual(mockUser);
      expect(mockRepository.findById).toHaveBeenCalledWith(1);
      expect(mockRepository.findById).toHaveBeenCalledTimes(1);
    });

    it('should throw error when user not found', async () => {
      // Arrange
      mockRepository.findById.mockResolvedValue(null);

      // Act & Assert
      await expect(userService.getUser(999))
        .rejects
        .toThrow('User not found');
    });

    it('should handle repository errors', async () => {
      // Arrange
      mockRepository.findById.mockRejectedValue(new Error('DB Error'));

      // Act & Assert
      await expect(userService.getUser(1))
        .rejects
        .toThrow('Failed to get user');
    });
  });

  describe('createUser', () => {
    it('should create user with valid data', async () => {
      // Arrange
      const userData = {
        name: 'Bob',
        email: 'bob@example.com',
        password: 'password123',
      };
      const savedUser = { id: 1, ...userData };
      mockRepository.save.mockResolvedValue(savedUser);

      // Act
      const result = await userService.createUser(userData);

      // Assert
      expect(result).toEqual(savedUser);
      expect(mockRepository.save).toHaveBeenCalledWith(
        expect.objectContaining({
          name: userData.name,
          email: userData.email,
        })
      );
    });

    it('should hash password before saving', async () => {
      // Arrange
      const userData = {
        name: 'Bob',
        email: 'bob@example.com',
        password: 'password123',
      };
      mockRepository.save.mockResolvedValue({ id: 1, ...userData });

      // Act
      await userService.createUser(userData);

      // Assert
      expect(mockRepository.save).toHaveBeenCalledWith(
        expect.objectContaining({
          password: expect.not.stringContaining('password123'),
        })
      );
    });

    it('should validate email format', async () => {
      // Arrange
      const userData = {
        name: 'Bob',
        email: 'invalid-email',  // 不正なメール
        password: 'password123',
      };

      // Act & Assert
      await expect(userService.createUser(userData))
        .rejects
        .toThrow('Invalid email format');
    });

    it('should check for duplicate email', async () => {
      // Arrange
      const userData = {
        name: 'Bob',
        email: 'existing@example.com',
        password: 'password123',
      };
      mockRepository.save.mockRejectedValue(
        new Error('Duplicate email')
      );

      // Act & Assert
      await expect(userService.createUser(userData))
        .rejects
        .toThrow('Email already exists');
    });
  });
});
```

### 8.2 統合テストチェックリスト

```typescript
// 統合テストチェックリスト

const integrationTestChecks = {
  'テスト環境': [
    'テスト用DBを使っているか',
    '環境が独立しているか',
    'テストデータが適切にセットアップされるか',
    'テスト後にクリーンアップされるか',
  ],

  'テストシナリオ': [
    'エンドツーエンドのフローをテストしているか',
    '複数コンポーネントの統合をテストしているか',
    'APIエンドポイントをテストしているか',
  ],
};

// ✅ 統合テストの例
import request from 'supertest';
import { app } from '../src/app';
import { setupTestDB, clearTestDB } from './helpers/db';

describe('User API Integration Tests', () => {
  beforeAll(async () => {
    await setupTestDB();
  });

  afterAll(async () => {
    await clearTestDB();
  });

  beforeEach(async () => {
    // 各テスト前にDBをリセット
    await clearTestDB();
  });

  describe('POST /api/users', () => {
    it('should create a new user', async () => {
      const userData = {
        name: 'Alice',
        email: 'alice@example.com',
        password: 'password123',
      };

      const response = await request(app)
        .post('/api/users')
        .send(userData)
        .expect(201);

      expect(response.body).toMatchObject({
        id: expect.any(Number),
        name: userData.name,
        email: userData.email,
      });
      expect(response.body.password).toBeUndefined();  // パスワードは返さない
    });

    it('should return 400 for invalid email', async () => {
      const userData = {
        name: 'Alice',
        email: 'invalid-email',
        password: 'password123',
      };

      const response = await request(app)
        .post('/api/users')
        .send(userData)
        .expect(400);

      expect(response.body.error).toContain('email');
    });
  });

  describe('GET /api/users/:id', () => {
    it('should get user by id', async () => {
      // まずユーザーを作成
      const createResponse = await request(app)
        .post('/api/users')
        .send({
          name: 'Bob',
          email: 'bob@example.com',
          password: 'password123',
        });

      const userId = createResponse.body.id;

      // ユーザーを取得
      const response = await request(app)
        .get(`/api/users/${userId}`)
        .expect(200);

      expect(response.body).toMatchObject({
        id: userId,
        name: 'Bob',
        email: 'bob@example.com',
      });
    });

    it('should return 404 for non-existent user', async () => {
      const response = await request(app)
        .get('/api/users/9999')
        .expect(404);

      expect(response.body.error).toContain('not found');
    });
  });

  describe('PUT /api/users/:id', () => {
    it('should update user', async () => {
      // ユーザー作成
      const createResponse = await request(app)
        .post('/api/users')
        .send({
          name: 'Charlie',
          email: 'charlie@example.com',
          password: 'password123',
        });

      const userId = createResponse.body.id;

      // ユーザー更新
      const updateData = {
        name: 'Charlie Updated',
      };

      const response = await request(app)
        .put(`/api/users/${userId}`)
        .send(updateData)
        .expect(200);

      expect(response.body.name).toBe(updateData.name);
    });
  });

  describe('DELETE /api/users/:id', () => {
    it('should delete user', async () => {
      // ユーザー作成
      const createResponse = await request(app)
        .post('/api/users')
        .send({
          name: 'David',
          email: 'david@example.com',
          password: 'password123',
        });

      const userId = createResponse.body.id;

      // ユーザー削除
      await request(app)
        .delete(`/api/users/${userId}`)
        .expect(204);

      // 削除確認
      await request(app)
        .get(`/api/users/${userId}`)
        .expect(404);
    });
  });
});
```

---

## 9. アーキテクチャチェックリスト

### 9.1 SOLID原則チェックリスト

```typescript
// SOLID原則チェックリスト

const solidChecks = {
  'S - Single Responsibility': [
    'クラス/関数が単一の責務を持っているか',
    '変更理由が1つだけか',
    '凝集度が高いか',
  ],

  'O - Open/Closed': [
    '拡張に対して開いているか',
    '修正に対して閉じているか',
    'ストラテジーパターンを使っているか',
  ],

  'L - Liskov Substitution': [
    '派生クラスが基底クラスと置き換え可能か',
    '契約を守っているか',
  ],

  'I - Interface Segregation': [
    'インターフェースが小さいか',
    '使わないメソッドを強制していないか',
  ],

  'D - Dependency Inversion': [
    '抽象に依存しているか',
    '具象に依存していないか',
    'DIコンテナを使っているか',
  ],
};

// ❌ 悪い例

// 単一責任原則違反
class User {
  name: string;
  email: string;

  save() {
    // DB保存（データアクセス責務）
  }

  sendEmail(subject: string, body: string) {
    // メール送信（通知責務）
  }

  generateReport() {
    // レポート生成（レポート責務）
  }
}

// 開放/閉鎖原則違反
function calculateDiscount(user: User): number {
  if (user.type === 'premium') {
    return 0.2;
  } else if (user.type === 'gold') {
    return 0.15;
  } else if (user.type === 'silver') {
    return 0.1;
  }
  return 0;
}

// ✅ 良い例

// 単一責任原則
class User {
  constructor(
    public readonly id: number,
    public name: string,
    public email: string
  ) {}
}

class UserRepository {
  async save(user: User): Promise<void> {
    // DB保存のみ
  }

  async findById(id: number): Promise<User | null> {
    // 検索のみ
  }
}

class EmailService {
  async send(to: string, subject: string, body: string): Promise<void> {
    // メール送信のみ
  }
}

class ReportGenerator {
  generate(user: User): Report {
    // レポート生成のみ
  }
}

// 開放/閉鎖原則（ストラテジーパターン）
interface DiscountStrategy {
  calculate(amount: number): number;
}

class PremiumDiscount implements DiscountStrategy {
  calculate(amount: number): number {
    return amount * 0.2;
  }
}

class GoldDiscount implements DiscountStrategy {
  calculate(amount: number): number {
    return amount * 0.15;
  }
}

class SilverDiscount implements DiscountStrategy {
  calculate(amount: number): number {
    return amount * 0.1;
  }
}

class PriceCalculator {
  constructor(private discountStrategy: DiscountStrategy) {}

  calculate(basePrice: number): number {
    const discount = this.discountStrategy.calculate(basePrice);
    return basePrice - discount;
  }
}

// 使用例
const calculator = new PriceCalculator(new PremiumDiscount());
const finalPrice = calculator.calculate(1000);

// 依存性逆転原則
interface Database {
  query(sql: string, params: any[]): Promise<any>;
}

class UserService {
  constructor(private db: Database) {}  // 抽象に依存

  async getUser(id: number): Promise<User | null> {
    const result = await this.db.query(
      'SELECT * FROM users WHERE id = ?',
      [id]
    );
    return result[0] || null;
  }
}

// 具体的な実装
class PostgresDatabase implements Database {
  async query(sql: string, params: any[]): Promise<any> {
    // PostgreSQL実装
  }
}

class MySQLDatabase implements Database {
  async query(sql: string, params: any[]): Promise<any> {
    // MySQL実装
  }
}

// DIコンテナで注入
const db = new PostgresDatabase();
const userService = new UserService(db);
```

---

## 10. ドキュメントチェックリスト

### 10.1 READMEチェックリスト

```markdown
## READMEチェックリスト

### 必須項目
- [ ] プロジェクト名
- [ ] 概要（1-2段落）
- [ ] インストール手順
- [ ] 使用方法
- [ ] ライセンス

### 推奨項目
- [ ] バッジ（ビルド状況、カバレッジ等）
- [ ] デモ/スクリーンショット
- [ ] 主要機能リスト
- [ ] 要件（Node.js バージョン等）
- [ ] 設定方法
- [ ] API ドキュメントへのリンク
- [ ] コントリビューションガイド
- [ ] トラブルシューティング
- [ ] 関連リソース

### オプション項目
- [ ] アーキテクチャ図
- [ ] ロードマップ
- [ ] FAQ
- [ ] 謝辞
```

### 10.2 APIドキュメントチェックリスト

```markdown
## APIドキュメントチェックリスト

### エンドポイント情報
- [ ] HTTPメソッド
- [ ] URLパス
- [ ] 説明
- [ ] パラメータ（パス、クエリ、ボディ）
- [ ] レスポンス形式
- [ ] ステータスコード
- [ ] エラーレスポンス

### 認証
- [ ] 認証方法（Bearer Token等）
- [ ] 認証例
- [ ] エラー時の対応

### 例
- [ ] リクエスト例（curl, JavaScript等）
- [ ] レスポンス例
- [ ] エラー例

### その他
- [ ] レート制限
- [ ] ペジネーション
- [ ] フィルタリング/ソート
- [ ] バージョニング
```

---

## まとめ

この完全なチェックリストを活用することで:

1. **見逃しを防ぐ**: 体系的にレビュー
2. **品質向上**: 一貫した基準でレビュー
3. **効率化**: チェックリストで高速化
4. **教育**: 新メンバーの学習ツール
5. **自動化**: CI/CDに組み込み可能

各プロジェクトに合わせてカスタマイズし、継続的に改善していきましょう。

---

**更新日**: 2025年1月2日
**次回更新予定**: 四半期ごと
**フィードバック**: skill-feedback@example.com
