# 命名規則 ── 変数・関数・クラスの命名術

> コードは書く時間の10倍読まれる。良い名前は最高のドキュメントであり、悪い名前は最悪の技術的負債である。命名はプログラミングで最も重要かつ最も困難なスキルの一つ。

---

## この章で学ぶこと

1. **命名の基本原則** ── 意図を明確に伝える名前の付け方を理解する
2. **要素別の命名規則** ── 変数・関数・クラス・定数それぞれの命名パターンを身につける
3. **命名のアンチパターン** ── 避けるべき命名習慣と改善方法を把握する

---

## 1. 命名の基本原則

```
+-----------------------------------------------------------+
|  良い名前の3条件                                          |
|  ─────────────────────────────────────                    |
|  1. 意図が明確 (Intention-Revealing)                      |
|     → 何のために存在するか分かる                          |
|  2. 発音可能 (Pronounceable)                              |
|     → チームで口頭議論できる                              |
|  3. 検索可能 (Searchable)                                 |
|     → IDE/grepで見つけられる                              |
+-----------------------------------------------------------+
```

```
  名前の良さと文脈理解にかかる時間

  理解時間
    ^
    |  ***
    |     ***
    |        ***
    |           ****
    |               *****
    |                    ********
    +------------------------------> 名前の質
     d  data  val  userData  activeUserList
     ↑                              ↑
   即座に理解不可能            即座に理解可能
```

**コード例1: 意図を明確にする命名**

```python
# 悪い命名: 何のデータか分からない
d = 86400
l = get_list()
for i in l:
    if i.s == 1:
        process(i)

# 良い命名: コードが自己説明的
SECONDS_PER_DAY = 86400
active_users = get_active_users()
for user in active_users:
    if user.status == UserStatus.ACTIVE:
        send_notification(user)
```

**コード例2: スコープに応じた名前の長さ**

```python
# ループ変数（短いスコープ）: 短い名前でOK
for i in range(10):
    matrix[i][i] = 1

# ただし意味がある場合は明示する
for row_index in range(height):
    for col_index in range(width):
        grid[row_index][col_index] = calculate_cell(row_index, col_index)

# モジュールレベル定数（長いスコープ）: 長くて具体的に
MAX_LOGIN_ATTEMPTS_BEFORE_LOCKOUT = 5
DEFAULT_SESSION_TIMEOUT_MINUTES = 30
```

---

## 2. 要素別の命名規則

### 2.1 変数名

```
  ┌────────────────────────────────────────────────┐
  │ 変数命名のガイドライン                          │
  ├──────────┬─────────────────────────────────────┤
  │ bool     │ is/has/can/should + 形容詞/過去分詞 │
  │          │ isActive, hasPermission, canEdit     │
  ├──────────┼─────────────────────────────────────┤
  │ 数値     │ 単位を含める                         │
  │          │ timeoutMs, fileSizeBytes, ageYears   │
  ├──────────┼─────────────────────────────────────┤
  │ コレクション│ 複数形 or xxxList/xxxMap           │
  │          │ users, orderItems, nameToEmail       │
  ├──────────┼─────────────────────────────────────┤
  │ 一時変数 │ 用途を示す                           │
  │          │ tempFile, swapValue, accumulator     │
  └──────────┴─────────────────────────────────────┘
```

**コード例3: ブール変数の命名**

```typescript
// 悪い: trueの意味が不明
let flag = true;
let check = false;
let status = true;

// 良い: true/falseの意味が明確
let isVisible = true;
let hasAdminPermission = false;
let shouldAutoSave = true;
let canDeletePost = user.role === 'admin';
let wasProcessed = order.processedAt !== null;
```

### 2.2 関数名

| パターン | 用途 | 例 |
|---------|------|-----|
| `get/fetch/find` | データ取得 | `getUserById`, `fetchOrders` |
| `create/build/make` | 生成 | `createUser`, `buildQuery` |
| `update/modify/set` | 更新 | `updateProfile`, `setName` |
| `delete/remove/clear` | 削除 | `deleteUser`, `removeItem` |
| `is/has/can/should` | 判定 | `isValid`, `hasAccess` |
| `validate/check/verify` | 検証 | `validateEmail`, `checkAuth` |
| `convert/transform/to` | 変換 | `toJSON`, `convertToCSV` |
| `calculate/compute` | 計算 | `calculateTotal`, `computeHash` |

**コード例4: 関数名の改善**

```python
# 悪い: 何をする関数か分からない
def handle(data):
    pass

def do_it(x, y):
    pass

def process(items):
    pass

# 良い: 動詞+名詞で動作と対象を明示
def validate_email_format(email: str) -> bool:
    pass

def calculate_monthly_revenue(transactions: list[Transaction]) -> Decimal:
    pass

def send_password_reset_email(user: User) -> None:
    pass
```

### 2.3 クラス名

```
  ┌────────────────────────────────────────────┐
  │ クラス命名のガイドライン                    │
  ├────────────┬───────────────────────────────┤
  │ エンティティ │ 名詞: User, Order, Product   │
  ├────────────┼───────────────────────────────┤
  │ サービス    │ 名詞+Service: PaymentService  │
  │            │ 動詞er: OrderProcessor         │
  ├────────────┼───────────────────────────────┤
  │ リポジトリ  │ 名詞+Repository               │
  │            │ UserRepository                 │
  ├────────────┼───────────────────────────────┤
  │ ファクトリ  │ 名詞+Factory                  │
  │            │ ConnectionFactory              │
  ├────────────┼───────────────────────────────┤
  │ 例外       │ 名詞+Error/Exception          │
  │            │ InvalidInputError              │
  └────────────┴───────────────────────────────┘
```

**コード例5: 名前空間を活用した命名**

```java
// 悪い: プレフィックスで名前空間を代用
class AppUserAccountValidationService { }
class AppUserAccountRepository { }
class AppUserAccountDTO { }

// 良い: パッケージ/モジュールで名前空間を構成
package com.example.user.account;

class ValidationService { }
class Repository { }
class AccountDTO { }

// 使用時: 文脈から意味が明確
import com.example.user.account.ValidationService;
```

---

## 3. 言語別の命名慣習

| 要素 | Python | JavaScript/TS | Java | Go |
|------|--------|--------------|------|-----|
| 変数 | snake_case | camelCase | camelCase | camelCase |
| 関数 | snake_case | camelCase | camelCase | CamelCase(公開)/camelCase(非公開) |
| クラス | PascalCase | PascalCase | PascalCase | PascalCase |
| 定数 | UPPER_SNAKE | UPPER_SNAKE | UPPER_SNAKE | CamelCase |
| ファイル | snake_case | camelCase/kebab | PascalCase | snake_case |
| パッケージ | snake_case | kebab-case | lowercase | lowercase |

| 原則 | 説明 |
|------|------|
| プロジェクト内で統一 | 言語慣習よりもプロジェクト内の一貫性が重要 |
| Linter で自動強制 | ESLint, pylint, checkstyle で命名規則を強制 |
| レビューで確認 | 自動化できない意味の明確さはレビューで補完 |

---

## 4. アンチパターン

### アンチパターン1: ハンガリアン記法の誤用

```typescript
// アンチパターン: 型をプレフィックスに入れる（現代では不要）
let strName: string = "太郎";
let intAge: number = 25;
let arrUsers: User[] = [];
let bIsActive: boolean = true;

// 改善: 型情報は型システムに任せる
let name: string = "太郎";
let age: number = 25;
let users: User[] = [];
let isActive: boolean = true;
```

### アンチパターン2: 略語・暗号的な命名

```python
# アンチパターン: 解読が必要な名前
def calc_ttl_w_dsc(itms, dsc_pct):
    ttl = 0
    for itm in itms:
        ttl += itm.prc * itm.qty
    return ttl * (1 - dsc_pct / 100)

# 改善: 省略せずフルスペルで
def calculate_total_with_discount(items, discount_percent):
    subtotal = sum(item.price * item.quantity for item in items)
    return subtotal * (1 - discount_percent / 100)
```

---

## 5. FAQ

### Q1: 長い名前は悪いのか？

長い名前自体は悪くない。**意味が曖昧な短い名前のほうが遥かに有害**。ただし「関数名が長すぎて1行に収まらない」場合は、その関数が複数の責任を持っている兆候かもしれない。名前の長さではなく、責任の分離を見直す。

### Q2: 命名に迷って時間がかかりすぎる場合はどうすべきか？

仮の名前（`temp_xxx`）を付けて先に実装し、全体の文脈が見えた段階でリネームする。IDE のリファクタリング機能を使えばリネームは安全に行える。**命名は反復的なプロセス**として捉える。

### Q3: 日本語の変数名は使ってよいか？

技術的には多くの言語で使用可能だが、以下の理由から英語が推奨される:
- 国際的なチームでの可読性
- ライブラリ/フレームワークとの一貫性
- 技術用語は英語のほうが正確
- ただし、ドメイン固有の日本語概念（「確定申告」等）はコメントで補足する

---

## まとめ

| 要素 | 命名の鍵 | 例 |
|------|---------|-----|
| 変数 | 何を格納しているか | `activeUserCount` |
| ブール | true/falseの意味 | `isAuthenticated` |
| 関数 | 何をするか（動詞+名詞） | `calculateShippingCost` |
| クラス | 何を表現するか（名詞） | `PaymentProcessor` |
| 定数 | 何の値か（UPPER_SNAKE） | `MAX_RETRY_COUNT` |
| 列挙 | 選択肢の集合 | `OrderStatus.SHIPPED` |

---

## 次に読むべきガイド

- [関数設計](./01-functions.md) ── 命名と密接に関わる関数の設計原則
- [コメント](./03-comments.md) ── 名前で表現しきれない情報の補足方法
- [コードレビューチェックリスト](../03-practices-advanced/04-code-review-checklist.md) ── 命名のレビュー観点

---

## 参考文献

1. **Robert C. Martin** 『Clean Code: A Handbook of Agile Software Craftsmanship』 Prentice Hall, 2008 (Chapter 2: Meaningful Names)
2. **Dustin Boswell, Trevor Foucher** 『The Art of Readable Code』 O'Reilly Media, 2011
3. **Steve McConnell** 『Code Complete: A Practical Handbook of Software Construction』 Microsoft Press, 2004 (Chapter 11: The Power of Variable Names)
