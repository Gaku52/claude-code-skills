# クリーンコード概要 ── なぜコード品質が重要か

> ソフトウェアの総コストの80%以上は保守に費やされる。読みやすく変更しやすいコードは、チーム全体の生産性を劇的に向上させる。

---

## この章で学ぶこと

1. **クリーンコードの定義** ── 著名なエンジニアたちの視点から「良いコード」を理解する
2. **品質がビジネスに与える影響** ── 技術的負債と開発速度の関係を定量的に把握する
3. **クリーンコードの実践原則** ── 日常のコーディングで適用できる基本ルールを身につける
4. **コード品質の定量的測定** ── 複雑度やカバレッジなどの客観的指標を使って品質を評価する
5. **クリーンコード文化の構築** ── チーム全体で品質を維持するための仕組みと文化を作る

---

## 前提知識

このガイドを最大限に活用するために、以下の知識があると望ましい。

| 前提知識 | 説明 | 参照リンク |
|---------|------|-----------|
| プログラミング基礎 | 変数、関数、クラスの基本概念 | `../../02-programming/` |
| オブジェクト指向の基本 | クラス、継承、ポリモーフィズム | `../../02-programming/` |
| Git基礎 | バージョン管理の基本操作 | `../../05-infrastructure/` |

※ 上記は必須ではないが、コード例を理解する上で役立つ。

---

## 1. クリーンコードとは何か

### 1.1 著名エンジニアによる定義

```
+-----------------------------------------------------------+
|  Robert C. Martin (Uncle Bob)                             |
|  「クリーンコードは読みやすく、理解しやすく、              |
|    変更しやすいコードである」                              |
+-----------------------------------------------------------+
|  Bjarne Stroustrup (C++の父)                              |
|  「エレガントで効率的なコードこそクリーンコード。          |
|    論理が明快でバグが隠れにくい」                          |
+-----------------------------------------------------------+
|  Grady Booch (UMLの父)                                    |
|  「クリーンコードは、よく書かれた散文のように読める」      |
+-----------------------------------------------------------+
|  Ward Cunningham (Wikiの発明者)                           |
|  「読んでみて"まさにこうあるべきだ"と感じるコード」        |
+-----------------------------------------------------------+
|  Michael Feathers                                         |
|  「クリーンコードとは、誰か他の人が                        |
|    メンテナンスすることを意識して書かれたコードだ」        |
+-----------------------------------------------------------+
```

これらの定義に共通するのは、**コードの読み手への配慮**である。プログラムはコンピュータに対する命令であると同時に、チームメンバーへのコミュニケーション手段でもある。

### 1.2 なぜ「読みやすさ」が最重要なのか ── WHY の深掘り

ソフトウェア開発において、コードを書く時間と読む時間の比率は概ね 1:10 と言われている（Robert C. Martin の調査）。つまり、コードは書く時間の10倍読まれる。この事実が、可読性を最優先すべき根本的な理由である。

```
  開発者の時間配分

  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  コードを読む時間                                    │
  │  ████████████████████████████████████████  (70%)    │
  │                                                     │
  │  既存コードの修正                                    │
  │  ████████████████  (20%)                            │
  │                                                     │
  │  新規コードの記述                                    │
  │  ██████  (10%)                                      │
  │                                                     │
  └─────────────────────────────────────────────────────┘
  ※ Robert C. Martin および複数の実証研究に基づく概算
```

この比率から導かれる結論は明快だ。**1分余計に費やして読みやすいコードを書けば、将来の10分が節約される**。逆に、書くスピードを優先して読みにくいコードを残すと、将来のコストは10倍に膨れ上がる。

### 1.3 品質の多面的評価

```
          ┌─────────────────────────────────────────┐
          │         コード品質の4象限                 │
          ├──────────────┬──────────────────────────┤
          │  可読性       │  保守性                   │
          │  ・命名が明確 │  ・変更が局所的           │
          │  ・構造が一貫 │  ・テストが容易           │
          │  ・意図が明白 │  ・影響範囲が予測可能     │
          ├──────────────┼──────────────────────────┤
          │  信頼性       │  効率性                   │
          │  ・エラー処理 │  ・適切なアルゴリズム     │
          │  ・エッジケース│ ・不要な計算がない        │
          │  ・型安全性   │  ・メモリ効率が良い       │
          └──────────────┴──────────────────────────┘
```

### 1.4 クリーンコードの内部メカニズム ── 脳科学的視点

なぜクリーンコードが重要なのかを脳科学の視点からも理解しておこう。

人間のワーキングメモリ（短期記憶）は、一度に保持できる情報チャンクが 7 +/- 2 個（Miller の法則）とされている。コードを読む際、以下の要素がそれぞれ1チャンクを消費する。

```
  ワーキングメモリの消費

  ┌────────────────────────────────────────────┐
  │  利用可能チャンク: 約7個                    │
  │                                            │
  │  [変数名の意味] [関数の目的] [制御フロー]   │
  │  [型情報] [エラーケース] [ビジネスルール]    │
  │  [呼び出し元の文脈]                         │
  │                                            │
  │  → 7個でほぼ限界。これ以上の情報を要求する  │
  │    コードは理解不能になる                    │
  └────────────────────────────────────────────┘
```

クリーンコードは、各チャンクの認知負荷を最小化する。良い命名は変数の意味を即座に伝え、小さな関数は目的を一目で把握させ、一貫した構造はパターン認識を助ける。つまり**クリーンコードとは、人間の認知限界を尊重したコード**なのである。

---

## 2. なぜ品質が重要か ── ビジネスインパクト

### 2.1 開発速度の推移

```
開発速度
  ^
  |  ****
  |      ****
  |          ****                    ← クリーンコード
  |              ****  ****  ****
  |
  |  ****
  |      **
  |        *                         ← 汚いコード
  |         *  *  .  .  .
  +------------------------------------> 時間
   1月  3月  6月  1年  2年
```

### 2.2 技術的負債の定量分析

技術的負債（Technical Debt）とは、短期的な速度を優先して品質を犠牲にした結果、将来支払うことになるコストの比喩である。Ward Cunningham が1992年に提唱した概念で、金融の負債と同様に「利子」が発生する。

```
  技術的負債の累積モデル

  コスト
    ^
    |                          ########
    |                     #####
    |                 ####              ← 利子（負債による追加コスト）
    |             ####
    |         ####
    |     ####
    |  ###
    | #
    +-----------------------------------------> 時間
    │  元本（最初に借りた負債）
    │  利子（負債によるスローダウン、バグ増加）
```

**技術的負債の種類と影響:**

| 種類 | 具体例 | 利子（追加コスト） |
|------|--------|-------------------|
| 意図的・慎重な負債 | 「リリース優先で後でリファクタリング」 | 計画的に返済可能 |
| 意図的・無謀な負債 | 「設計する時間がない」 | 急速に利子が増大 |
| 無意識・慎重な負債 | 後から「こうすべきだった」と気づく | 学習コストとして許容可能 |
| 無意識・無謀な負債 | クリーンコードを知らない | 気づかないまま利子が蓄積 |

Martin Fowler の「技術的負債の四象限」に基づく分類。最も危険なのは「無意識・無謀な負債」で、チームがそもそもコード品質の重要性を認識していないケースである。

### 2.3 定量データ

IBMの調査研究と複数の産業レポートに基づく、品質投資の効果を以下に示す。

```
  品質投資のROI（Return on Investment）

  ┌──────────────────────────────────────────────────────┐
  │ 投資: コードレビュー 1時間                            │
  │ 効果: 後工程のバグ修正 3〜20時間を節約               │
  │ ROI:  3x 〜 20x                                     │
  ├──────────────────────────────────────────────────────┤
  │ 投資: テスト自動化（初期コスト 2週間）               │
  │ 効果: 手動テスト削減、リグレッション防止              │
  │ ROI:  3ヶ月で損益分岐、その後は継続的にプラス        │
  ├──────────────────────────────────────────────────────┤
  │ 投資: リファクタリング（技術的負債の返済）            │
  │ 効果: 変更速度の回復、バグ率の低下                    │
  │ ROI:  負債の大きさに依存。大きいほど効果大           │
  └──────────────────────────────────────────────────────┘
```

**バグ修正コストの増大（フェーズ別）:**

| 発見フェーズ | 相対コスト | 例 |
|------------|-----------|-----|
| 要件定義 | 1x | 「この仕様おかしくない？」 |
| 設計 | 3-6x | 設計レビューで発見 |
| コーディング | 10x | コードレビューで発見 |
| テスト | 15-40x | テスト工程で発見 |
| リリース後 | 30-100x | 本番障害として発覚 |

Steve McConnell『Code Complete』やBarry Boehm の研究に基づく。**早期に品質を作り込むことが、最もコスト効率の高い投資**であることが明確にわかる。

---

## 3. クリーンコードの実践 ── コード例

### コード例1: 可読性の比較 ── 意図が読めないコード vs クリーンコード

```python
# ダーティコード: 何をしているか読み取れない
def calc(l, t):
    r = []
    for i in l:
        if i['a'] > t and i['s'] == 1:
            r.append(i['n'])
    return r

# クリーンコード: 意図が明確
ACTIVE = 1

def find_active_users_above_threshold(
    users: list[dict],
    age_threshold: int
) -> list[str]:
    """指定年齢以上のアクティブユーザー名を返す。

    Args:
        users: ユーザー情報の辞書リスト。各辞書は 'age', 'status', 'name' キーを持つ。
        age_threshold: 年齢の下限値（この値より大きいユーザーが対象）。

    Returns:
        条件に合致するユーザー名のリスト。

    Examples:
        >>> users = [
        ...     {'name': '田中', 'age': 25, 'status': 1},
        ...     {'name': '鈴木', 'age': 17, 'status': 1},
        ...     {'name': '佐藤', 'age': 30, 'status': 0},
        ... ]
        >>> find_active_users_above_threshold(users, 20)
        ['田中']
    """
    active_senior_users = []
    for user in users:
        if user['age'] > age_threshold and user['status'] == ACTIVE:
            active_senior_users.append(user['name'])
    return active_senior_users
```

改善のポイント:
- 関数名が処理内容を正確に表現
- 引数名が意味を持つ
- 型ヒントで期待される型が明確
- docstringで使用例まで記載
- マジックナンバー（1）を定数化

### コード例2: 構造化されたエラーハンドリング

```python
# ダーティコード: エラーが飲み込まれる
def get_user(id):
    try:
        return db.query(id)
    except:
        return None  # どんなエラーも握りつぶす → デバッグ不可能

# クリーンコード: エラーの意味が明確
class UserNotFoundError(Exception):
    """ユーザーが見つからない場合のエラー"""
    def __init__(self, user_id: int):
        self.user_id = user_id
        super().__init__(f"ユーザーID {user_id} は存在しません")

class DatabaseConnectionError(Exception):
    """データベース接続に失敗した場合のエラー"""
    pass

def get_user_by_id(user_id: int) -> "User":
    """ユーザーIDからユーザーを取得する。

    Args:
        user_id: 取得対象のユーザーID。

    Returns:
        該当するUserオブジェクト。

    Raises:
        UserNotFoundError: ユーザーが見つからない場合。
        DatabaseConnectionError: DB接続に失敗した場合。
    """
    try:
        user = user_repository.find_by_id(user_id)
    except ConnectionError as e:
        raise DatabaseConnectionError(f"DB接続失敗: {e}") from e

    if user is None:
        raise UserNotFoundError(user_id)

    return user
```

改善のポイント:
- `except:` の代わりに具体的な例外型をキャッチ
- カスタム例外クラスでドメイン固有のエラーを表現
- `from e` で元の例外を保持（例外チェーン）
- docstring に Raises セクションを記載

### コード例3: 単一責任の関数

```javascript
// ダーティコード: 1つの関数で複数の責任
function processOrder(order) {
  // バリデーション (責任1)
  if (!order.items || order.items.length === 0) return false;
  if (!order.customer) return false;

  // 合計計算 (責任2)
  let total = 0;
  for (const item of order.items) {
    total += item.price * item.quantity;
    if (item.discount) total -= item.discount;
  }

  // DB保存 (責任3)
  db.save({ ...order, total, status: 'confirmed' });

  // メール送信 (責任4)
  sendEmail(order.customer.email, `注文確定: ${total}円`);

  return true;
}

// クリーンコード: 各責任を分離
function processOrder(order) {
  validateOrder(order);
  const total = calculateOrderTotal(order.items);
  const confirmedOrder = confirmOrder(order, total);
  notifyCustomer(confirmedOrder);
  return confirmedOrder;
}

function validateOrder(order) {
  if (!order.items || order.items.length === 0) {
    throw new InvalidOrderError('注文には1つ以上の商品が必要です');
  }
  if (!order.customer) {
    throw new InvalidOrderError('顧客情報が必要です');
  }
}

function calculateOrderTotal(items) {
  return items.reduce((total, item) => {
    const itemTotal = item.price * item.quantity;
    const discount = item.discount || 0;
    return total + itemTotal - discount;
  }, 0);
}

function confirmOrder(order, total) {
  const confirmedOrder = { ...order, total, status: 'confirmed' };
  orderRepository.save(confirmedOrder);
  return confirmedOrder;
}

function notifyCustomer(order) {
  emailService.send(
    order.customer.email,
    `注文確定: ${order.total}円`
  );
}
```

### コード例4: マジックナンバーの排除

```java
// ダーティコード: マジックナンバーだらけ
if (user.getAge() >= 18 && user.getScore() > 70 && user.getType() == 3) {
    applyDiscount(0.15);
}

// クリーンコード: 定数で意味を付与
private static final int LEGAL_AGE = 18;
private static final int PREMIUM_SCORE_THRESHOLD = 70;
private static final int GOLD_MEMBER_TYPE = 3;
private static final double GOLD_MEMBER_DISCOUNT_RATE = 0.15;

if (user.isAdult(LEGAL_AGE)
    && user.hasPremiumScore(PREMIUM_SCORE_THRESHOLD)
    && user.isGoldMember()) {
    applyDiscount(GOLD_MEMBER_DISCOUNT_RATE);
}

// さらに改善: ビジネスロジックをメソッドに抽出
if (user.isEligibleForGoldDiscount()) {
    applyDiscount(GOLD_MEMBER_DISCOUNT_RATE);
}
```

### コード例5: ガード節による早期リターン

```typescript
// ダーティコード: ネストが深い
function getPayAmount(employee: Employee): number {
  let result: number;
  if (employee.isSeparated) {
    result = separatedAmount(employee);
  } else {
    if (employee.isRetired) {
      result = retiredAmount(employee);
    } else {
      if (employee.isOnLeave) {
        result = leaveAmount(employee);
      } else {
        result = normalPayAmount(employee);
      }
    }
  }
  return result;
}

// クリーンコード: ガード節で平坦化
function getPayAmount(employee: Employee): number {
  if (employee.isSeparated) return separatedAmount(employee);
  if (employee.isRetired) return retiredAmount(employee);
  if (employee.isOnLeave) return leaveAmount(employee);
  return normalPayAmount(employee);
}
```

### コード例6: コメントではなくコードで意図を伝える

```python
# ダーティコード: コメントに頼る
# 30日以上ログインしていないユーザーを取得
# ステータスがアクティブで、トライアル期間が終了しているもの
users = []
for u in all_users:
    d = (datetime.now() - u.last_login).days
    if d >= 30 and u.status == 1 and u.trial_end < datetime.now():
        users.append(u)

# クリーンコード: コード自体が意図を語る
INACTIVE_THRESHOLD_DAYS = 30

def find_inactive_but_subscribed_users(
    all_users: list[User],
    threshold_days: int = INACTIVE_THRESHOLD_DAYS
) -> list[User]:
    """トライアル終了済みのアクティブユーザーで、一定期間ログインしていない人を返す。"""
    return [
        user for user in all_users
        if user.is_inactive_for(threshold_days)
        and user.is_active
        and user.has_trial_ended()
    ]
```

### コード例7: 条件式の抽出

```java
// ダーティコード: 複雑な条件式
if (date.getMonth() >= 6 && date.getMonth() <= 8
    && temperature > 30
    && !isHoliday(date)
    && employee.getVacationDaysLeft() > 0
    && !employee.isOnCriticalProject()) {
    applySummerBonus(employee);
}

// クリーンコード: 条件をメソッドに抽出
if (isSummerBonusEligible(date, temperature, employee)) {
    applySummerBonus(employee);
}

private boolean isSummerBonusEligible(
    LocalDate date, int temperature, Employee employee
) {
    return isSummerSeason(date)
        && isHotDay(temperature)
        && isWorkingDay(date)
        && employee.hasAvailableVacationDays()
        && !employee.isOnCriticalProject();
}

private boolean isSummerSeason(LocalDate date) {
    int month = date.getMonthValue();
    return month >= 6 && month <= 8;
}

private boolean isHotDay(int temperature) {
    return temperature > SUMMER_BONUS_TEMPERATURE_THRESHOLD;
}
```

---

## 4. クリーンコードの基本原則

| 原則 | 説明 | 効果 | 参照 |
|------|------|------|------|
| 可読性第一 | コードは書く回数より読む回数が圧倒的に多い | 理解時間の短縮 | 本章 |
| DRY | 同じロジックを繰り返さない | 変更箇所の一元化 | [DRY/KISS/YAGNI](./02-dry-kiss-yagni.md) |
| KISS | 複雑さを避け、シンプルに保つ | バグの予防 | [DRY/KISS/YAGNI](./02-dry-kiss-yagni.md) |
| YAGNI | 今必要でない機能は作らない | 無駄な開発の排除 | [DRY/KISS/YAGNI](./02-dry-kiss-yagni.md) |
| SRP | 1つの関数/クラスに1つの責任 | 変更影響の局所化 | [SOLID原則](./01-solid.md) |
| 意図の明示 | 名前・構造でコードの目的を伝える | コメント依存の低減 | [命名規則](../01-practices/00-naming.md) |
| 低結合・高凝集 | モジュール間の依存を最小化し、内部の関連性を最大化 | テスト容易性の向上 | [結合度と凝集度](./03-coupling-cohesion.md) |

### 原則間の関係図

```
  ┌────────────────────────────────────────────────────────┐
  │                 クリーンコードの原則体系                  │
  │                                                        │
  │              ┌──────────────┐                           │
  │              │  可読性第一   │  ← 最上位の価値           │
  │              └──────┬───────┘                           │
  │           ┌────────┼────────┐                           │
  │           v        v        v                           │
  │    ┌──────────┐ ┌──────┐ ┌──────┐                      │
  │    │   KISS   │ │ DRY  │ │ YAGNI│  ← 基本3原則         │
  │    └────┬─────┘ └──┬───┘ └──┬───┘                      │
  │         │          │        │                           │
  │         v          v        v                           │
  │    ┌──────────────────────────────┐                     │
  │    │       SOLID原則               │  ← 設計原則         │
  │    │  (SRP, OCP, LSP, ISP, DIP)   │                     │
  │    └────────────┬─────────────────┘                     │
  │                 │                                       │
  │         ┌───────┼───────┐                               │
  │         v               v                               │
  │  ┌────────────┐  ┌────────────────┐                     │
  │  │ 低結合高凝集 │  │ デメテルの法則  │  ← モジュール原則   │
  │  └────────────┘  └────────────────┘                     │
  └────────────────────────────────────────────────────────┘
```

---

## 5. コード品質の測定

### 5.1 定量的指標

| 指標 | 説明 | 目安 | 測定ツール |
|------|------|------|-----------|
| サイクロマティック複雑度 | 分岐数に基づく複雑さ | 関数あたり10以下 | radon (Python), ESLint (JS) |
| 認知的複雑度 | 人間が感じる理解しにくさ | 関数あたり15以下 | SonarQube |
| コードカバレッジ | テストで実行されるコードの割合 | 80%以上 | pytest-cov, Istanbul |
| 重複率 | コピペされたコードの割合 | 5%以下 | PMD CPD, jscpd |
| 関数の行数 | 1関数の物理行数 | 20行以下推奨 | 各種Linter |
| 依存関係の深さ | モジュール間の依存段数 | 3段以下 | deptry, madge |
| 技術的負債比率 | 修正コスト / 再開発コスト | 5%以下 | SonarQube |

### 5.2 サイクロマティック複雑度の計算方法

サイクロマティック複雑度（Cyclomatic Complexity）は Thomas McCabe が1976年に提案した指標で、プログラム内の独立した実行パスの数を表す。

```python
# 複雑度1: 分岐なし
def greet(name: str) -> str:
    return f"Hello, {name}"

# 複雑度2: if文が1つ
def check_age(age: int) -> str:
    if age >= 18:
        return "成人"
    return "未成年"

# 複雑度4: if + elif + for
def classify_scores(scores: list[int]) -> dict:
    result = {"high": 0, "mid": 0, "low": 0}
    for score in scores:           # +1
        if score >= 80:            # +1
            result["high"] += 1
        elif score >= 50:          # +1
            result["mid"] += 1
        else:
            result["low"] += 1
    return result

# 計算方法: M = E - N + 2P
# E = エッジ数, N = ノード数, P = 連結成分数
# 簡易計算: M = 分岐キーワード数(if, elif, for, while, and, or, except) + 1
```

### 5.3 認知的複雑度（Cognitive Complexity）

SonarSource が提案した、サイクロマティック複雑度の改良版。人間にとっての「理解しにくさ」を、以下のルールで数値化する。

```
  認知的複雑度のカウントルール

  1. ネストが深くなるほどペナルティが増加
     if (a) {           // +1
       if (b) {         // +2 (ネスト1段)
         if (c) {       // +3 (ネスト2段)
         }
       }
     }

  2. break in linear flow
     if, else if, else, switch, for, while,
     catch, &&, ||, ?:  → 各+1

  3. ネストを増加させない構造
     else, elif        → ネストペナルティなし

  サイクロマティック複雑度との違い:
  ・サイクロマティック: switch 10分岐 → 複雑度10（高い）
  ・認知的: switch 10分岐 → 複雑度1（人間にとっては読みやすい）
```

### 5.4 品質ダッシュボードの構成例

```
  ┌────────────────────────────────────────────────────┐
  │  品質ダッシュボード                                  │
  ├────────────────────────────────────────────────────┤
  │                                                    │
  │  [カバレッジ]    [複雑度]      [重複率]              │
  │   ████ 85%      平均 6.2      ██ 3.1%             │
  │   目標: 80%     目標: <10     目標: <5%            │
  │                                                    │
  │  [技術的負債]    [セキュリティ]  [新規Issue]         │
  │   12日分        脆弱性 0件     今週 +3             │
  │   先週比: -2日  ブロッカー 0   先週比: -5          │
  │                                                    │
  │  [トレンド]                                         │
  │   カバレッジ ↑  複雑度 →  負債 ↓  (改善傾向)      │
  └────────────────────────────────────────────────────┘
```

---

## 6. クリーンコード実現のためのツールチェーン

### 6.1 自動フォーマッタとLinter

| カテゴリ | Python | JavaScript/TS | Java | Go |
|---------|--------|--------------|------|-----|
| フォーマッタ | Black, Ruff | Prettier | google-java-format | gofmt |
| Linter | Ruff, pylint | ESLint | Checkstyle, SpotBugs | golangci-lint |
| 型チェック | mypy | TypeScript | javac | コンパイラ |
| 複雑度測定 | radon | eslint-plugin-complexity | PMD | gocyclo |
| テストカバレッジ | pytest-cov | Istanbul/c8 | JaCoCo | go test -cover |

### 6.2 CI/CDパイプラインでの品質ゲート

```
  ┌──────────────────────────────────────────────┐
  │  CI/CD 品質ゲートの設定例                      │
  │                                              │
  │  Stage 1: Lint & Format                      │
  │  ├── Formatter check (--check)               │
  │  ├── Linter (error のみブロック)              │
  │  └── 型チェック                               │
  │                                              │
  │  Stage 2: Test                               │
  │  ├── ユニットテスト                           │
  │  ├── インテグレーションテスト                  │
  │  └── カバレッジ計測 (80%以上)                 │
  │                                              │
  │  Stage 3: Quality Analysis                   │
  │  ├── SonarQube / SonarCloud                  │
  │  ├── 技術的負債チェック                        │
  │  └── セキュリティスキャン                      │
  │                                              │
  │  Gate: すべてパスしたらマージ許可              │
  └──────────────────────────────────────────────┘
```

---

## 7. クリーンコード文化の構築

### 7.1 ボーイスカウトルール

> 「コードベースを触ったら、見つけた時よりも少しだけ綺麗にして離れよ」── Robert C. Martin

```
  ボーイスカウトルールの実践

  タスク: ユーザー検索機能にフィルタ追加

  ① 機能実装（本来の仕事）
     user_search.py に filter 機能を追加

  ② 小さな改善（ボーイスカウトルール）
     ・変数名 lst → users に変更
     ・未使用 import を削除
     ・docstring 追加

  ③ やりすぎない（スコープ制限）
     ・大規模リファクタリングは別タスクに
     ・テストがないファイルの全面改修は避ける
     ・関係ないファイルは触らない
```

### 7.2 コードレビューでの品質チェック

```
  コードレビューチェックリスト（品質観点）

  □ 命名は意図を正確に伝えているか
  □ 関数は1つのことだけをしているか
  □ マジックナンバーは定数化されているか
  □ エラーハンドリングは適切か
  □ テストは境界値とエッジケースをカバーしているか
  □ 重複コードはないか
  □ 不要なコメントはないか（コードで表現すべき）
  □ 依存関係の方向は正しいか（DIP）
  □ 変更の影響範囲は局所的か
  □ パフォーマンスの懸念はないか
```

### 7.3 段階的導入戦略

チームにクリーンコード文化を導入する際は、段階的に進めることが成功の鍵。

| Phase | 期間 | 施策 | 成功指標 |
|-------|------|------|---------|
| 1. 認識 | 1-2週 | 勉強会、書籍共有 | チームメンバーが原則を説明できる |
| 2. 自動化 | 2-4週 | Linter/Formatter導入、CI設定 | 全PRが品質ゲートを通過 |
| 3. 実践 | 1-3月 | コードレビュー強化、ペアプロ | レビュー指摘の減少 |
| 4. 文化 | 3-6月 | 品質ダッシュボード、振り返り | メトリクスの継続的改善 |
| 5. 定着 | 6月〜 | 新人オンボーディングに組み込み | 新メンバーも自然に実践 |

---

## 8. トレードオフとエッジケース

### 8.1 クリーンコード vs パフォーマンス

クリーンコードとパフォーマンスが対立する場面は存在する。その際の判断基準を以下に示す。

```
  判断フロー

  パフォーマンス問題が存在するか？
  ├── No → クリーンコードを優先
  └── Yes → 測定したか？
       ├── No → まず測定する（推測で最適化しない）
       └── Yes → ボトルネック箇所を特定
            ├── ボトルネック → 最適化する（コメントで理由を残す）
            └── 非ボトルネック → クリーンコードを維持
```

**Donald Knuth の格言:**

> 「早すぎる最適化は諸悪の根源である」（"Premature optimization is the root of all evil"）

ただし、この言葉の全文を知っておくことも重要:

> 「プログラマは、プログラムの重要でない部分の速度について考えたり、心配したりすることに膨大な時間を費やしている。そしてこれらの効率化の試みは、デバッグと保守を考えると実際には大きな悪影響を持つ。我々は97%の時間、小さな効率性を忘れるべきである: **早すぎる最適化は諸悪の根源である**。しかし、残りの3%の重要な機会を逃してはならない。」

### 8.2 クリーンコード vs 締め切り

```
  技術的負債の「意図的借入」判断マトリクス

  ┌───────────────────┬────────────────────┐
  │  借入すべき場面     │  借入を避けるべき場面 │
  ├───────────────────┼────────────────────┤
  │ ・事業の存続に関わる │ ・恒常的な締め切り   │
  │   リリース          │   プレッシャー       │
  │ ・実験的な機能      │ ・コア機能の品質     │
  │   （検証後に廃棄    │   低下              │
  │    の可能性あり）    │ ・チームが返済を      │
  │ ・返済計画が明確    │   認識していない     │
  │                    │ ・返済の見通しなし    │
  └───────────────────┴────────────────────┘
```

### 8.3 代替アプローチ: プラグマティック品質

「完璧なクリーンコード」を目指すのではなく、**プラグマティック（実用主義的）な品質**を目指すアプローチもある。

| 完璧主義 | プラグマティック |
|---------|----------------|
| 全コードを理想的な品質に | 変更頻度の高い箇所を重点改善 |
| 全テストを網羅的に | リスクの高い箇所を重点テスト |
| 全設計をSOLID準拠に | 拡張予定のある箇所を重点設計 |
| 一度に全面リファクタリング | 触るたびに少しずつ改善 |

---

## 9. アンチパターン

### アンチパターン1: 過度な最適化（Premature Optimization）

```python
# NG: 読みにくい最適化を早期に行う
def f(d):
    return {k: v for k, v in sorted(
        ((k, sum(x['v'] for x in g))
         for k, g in __import__('itertools').groupby(
             sorted(d, key=lambda x: x['k']),
             key=lambda x: x['k'])),
        key=lambda x: -x[1])}

# OK: まず可読性を優先し、必要時に最適化
from collections import defaultdict

def aggregate_and_sort_by_value(data: list[dict]) -> dict:
    """キーごとに値を集計し、降順で返す。

    Args:
        data: 'key' と 'value' を持つ辞書のリスト。

    Returns:
        キーごとの合計値を降順に並べた辞書。

    Examples:
        >>> data = [
        ...     {'key': 'a', 'value': 10},
        ...     {'key': 'b', 'value': 5},
        ...     {'key': 'a', 'value': 20},
        ... ]
        >>> aggregate_and_sort_by_value(data)
        {'a': 30, 'b': 5}
    """
    aggregated = defaultdict(int)
    for item in data:
        aggregated[item['key']] += item['value']

    return dict(sorted(
        aggregated.items(),
        key=lambda pair: pair[1],
        reverse=True
    ))
```

### アンチパターン2: コメントで汚いコードを正当化する

```java
// NG: コメントで複雑さを説明
// i は顧客のインデックス、j は注文のインデックス、
// k は商品のインデックス、t は合計金額
for (int i = 0; i < c.length; i++) {
    for (int j = 0; j < c[i].o.length; j++) {
        for (int k = 0; k < c[i].o[j].p.length; k++) {
            t += c[i].o[j].p[k].pr * c[i].o[j].p[k].q;
        }
    }
}

// OK: コード自体が意味を語る
for (Customer customer : customers) {
    for (Order order : customer.getOrders()) {
        for (Product product : order.getProducts()) {
            totalRevenue += product.getPrice() * product.getQuantity();
        }
    }
}
```

### アンチパターン3: 過度な抽象化（Astronaut Architecture）

```python
# NG: 単純な処理を過度に抽象化
class AbstractDataProcessorFactory:
    def create_processor(self): ...

class ConcreteDataProcessorFactory(AbstractDataProcessorFactory):
    def create_processor(self):
        return DataProcessor(
            reader=FileReaderAdapter(CsvReader()),
            transformer=DataTransformationPipeline([
                WhitespaceNormalizer(),
                EncodingConverter(),
            ]),
            writer=OutputWriterAdapter(ConsoleWriter()),
        )

# OK: 直接的でシンプル
import csv

def read_and_display_csv(filepath: str) -> None:
    """CSVファイルを読み込んでコンソールに表示する。"""
    with open(filepath) as f:
        reader = csv.reader(f)
        for row in reader:
            print(", ".join(row))
```

---

## 10. 実践演習

### 演習1（基礎）: コードの可読性改善

以下のコードをクリーンコードの原則に従って改善せよ。

```python
# 改善前
def p(d):
    r = 0
    for x in d:
        if x['t'] == 'i':
            r += x['a']
        elif x['t'] == 'e':
            r -= x['a']
    if r < 0:
        r = 0
    return r
```

**期待される改善ポイント:**
- 意味のある変数名・関数名に変更
- 定数化すべき文字列リテラルの抽出
- 型ヒントとdocstringの追加

**期待される出力例:**

```python
INCOME = 'income'
EXPENSE = 'expense'

def calculate_balance(transactions: list[dict]) -> float:
    """取引リストから残高を計算する。残高は0未満にならない。

    Args:
        transactions: 'type' ('income' or 'expense') と 'amount' (float) を持つ辞書リスト。

    Returns:
        計算された残高（最小値は0）。
    """
    balance = 0.0
    for transaction in transactions:
        if transaction['type'] == INCOME:
            balance += transaction['amount']
        elif transaction['type'] == EXPENSE:
            balance -= transaction['amount']
    return max(balance, 0.0)
```

### 演習2（応用）: 技術的負債の分析

以下のコードの技術的負債を特定し、改善計画を立案せよ。

```python
import json, os, smtplib, sqlite3

class App:
    def run(self, action, data):
        conn = sqlite3.connect('app.db')
        if action == 'register':
            if data.get('email') and '@' in data['email'] and len(data.get('password', '')) >= 8:
                conn.execute("INSERT INTO users VALUES (?, ?, ?)",
                    (data['email'], data['password'], json.dumps({'created': str(datetime.now())})))
                conn.commit()
                try:
                    s = smtplib.SMTP('localhost')
                    s.sendmail('noreply@app.com', data['email'], 'Welcome!')
                    s.quit()
                except:
                    pass
                return {'status': 'ok'}
            return {'status': 'error', 'msg': 'invalid'}
        elif action == 'login':
            r = conn.execute("SELECT * FROM users WHERE email=? AND password=?",
                (data.get('email', ''), data.get('password', ''))).fetchone()
            if r:
                return {'status': 'ok', 'token': os.urandom(16).hex()}
            return {'status': 'error'}
        elif action == 'delete':
            conn.execute("DELETE FROM users WHERE email=?", (data.get('email', ''),))
            conn.commit()
            return {'status': 'ok'}
        conn.close()
```

**期待される分析:**

| 負債 | 種類 | 優先度 | 改善案 |
|------|------|--------|--------|
| God Class | SRP違反 | 高 | 責任分離（Service, Repository, Validator） |
| 平文パスワード保存 | セキュリティ | 最高 | bcrypt等によるハッシュ化 |
| SQL インジェクション耐性 | セキュリティ | 高 | パラメータ化クエリ（実装済みだが検証不足） |
| except: pass | エラー握りつぶし | 中 | ログ記録 + 適切なエラー処理 |
| マジックナンバー(8) | 可読性 | 低 | 定数化 |
| DB接続の管理 | リソースリーク | 高 | コンテキストマネージャ（with文） |
| テスト不在 | 保守性 | 高 | ユニットテスト追加 |

### 演習3（発展）: クリーンコードへのリファクタリング

演習2のコードを、以下の品質基準を満たすようにリファクタリングせよ。

**品質基準:**
- SOLID原則に準拠
- 各クラス/関数の責任が明確
- エラーハンドリングが適切
- テスト可能な設計
- セキュリティ上の問題が解消

**期待される出力例（一部）:**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import bcrypt

# ドメインモデル
@dataclass
class User:
    email: str
    password_hash: str

# リポジトリ層
class UserRepository(ABC):
    @abstractmethod
    def save(self, user: User) -> None: ...

    @abstractmethod
    def find_by_email(self, email: str) -> Optional[User]: ...

    @abstractmethod
    def delete_by_email(self, email: str) -> None: ...

# バリデーション
class UserValidator:
    MIN_PASSWORD_LENGTH = 8

    def validate_registration(self, email: str, password: str) -> list[str]:
        errors = []
        if not email or '@' not in email:
            errors.append("有効なメールアドレスを入力してください")
        if len(password) < self.MIN_PASSWORD_LENGTH:
            errors.append(f"パスワードは{self.MIN_PASSWORD_LENGTH}文字以上必要です")
        return errors

# 認証サービス
class AuthService:
    def __init__(self, repository: UserRepository, validator: UserValidator):
        self.repository = repository
        self.validator = validator

    def register(self, email: str, password: str) -> "RegistrationResult":
        errors = self.validator.validate_registration(email, password)
        if errors:
            return RegistrationResult.failure(errors)

        password_hash = bcrypt.hashpw(
            password.encode(), bcrypt.gensalt()
        ).decode()

        user = User(email=email, password_hash=password_hash)
        self.repository.save(user)
        return RegistrationResult.success(user)
```

---

## 11. FAQ

### Q1: クリーンコードを書くと開発が遅くならないか？

短期的には命名や構造の検討に時間がかかる。しかし中長期的には、読む時間・デバッグ時間・変更時間が大幅に削減され、**トータルの生産性は向上する**。IBMの調査では、コードレビューにより後工程の欠陥修正コストが10〜100倍削減されることが示されている。

具体的な数字で考えてみよう。1日8時間の開発時間のうち、コードを読む時間が5.6時間（70%）、修正が1.6時間（20%）、新規コード記述が0.8時間（10%）だとする。クリーンコードによって読む時間が30%削減されれば、1日あたり1.68時間の節約になる。これは月に33.6時間、年に403時間に相当する。

### Q2: レガシーコードはどこからクリーンにすべきか？

**ボーイスカウトルール**（来た時よりも綺麗にして帰る）に従い、触ったファイルを少しずつ改善する。全面書き換えではなく、テストを追加しながら段階的にリファクタリングする。優先度は「頻繁に変更されるファイル」から。

具体的なステップ:
1. **Hotspot分析**: Gitのコミット履歴から変更頻度の高いファイルを特定（`git log --format=format: --name-only | sort | uniq -c | sort -rn | head -20`）
2. **テスト追加**: 改善対象ファイルにまずテストを書く（振る舞いを固定）
3. **段階的改善**: ボーイスカウトルールに従い、触るたびに少しずつ改善
4. **計測**: 改善前後の品質メトリクスを比較

### Q3: チーム全体でクリーンコードを徹底するには？

1. **コーディング規約の策定と自動化**（Linter、Formatter）
2. **コードレビュー文化の醸成**（レビューチェックリストの活用）
3. **ペアプログラミング/モブプログラミング**の導入
4. **技術的負債の可視化**（品質ダッシュボード）
5. **チーム読書会**（Clean Code, Refactoring 等を輪読）
6. **リファクタリングスプリント**の定期実施

### Q4: コードが「クリーン」かどうかの判断基準は？

以下のチェックリストを使って判断できる:

- **5秒ルール**: 関数を見て5秒以内に目的が把握できるか？
- **名前テスト**: 関数名だけで何をするか説明できるか？
- **驚き最小の原則**: コードの動作に驚く部分がないか？
- **修正テスト**: この箇所を変更する場合、影響範囲は予測できるか？
- **テストテスト**: この関数のユニットテストは容易に書けるか？

### Q5: クリーンコードと設計パターンの関係は？

設計パターンはクリーンコードを実現するための**手段の一つ**であり、目的ではない。パターンを知ることは重要だが、「パターンを適用するためにパターンを使う」のは本末転倒。

正しい関係:
```
問題の認識 → 原則（SOLID, DRY等）で判断 → 必要ならパターンを適用
```

誤った関係:
```
パターンを知っている → パターンを適用できる場面を探す → 無理やり適用
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| クリーンコードの本質 | 読みやすく、理解しやすく、変更しやすいコード |
| ビジネス効果 | 保守コスト削減、開発速度の維持、バグの予防 |
| 基本原則 | 可読性第一、DRY、KISS、YAGNI、SRP、低結合・高凝集 |
| 測定方法 | 複雑度、カバレッジ、重複率、関数サイズ、技術的負債比率 |
| 実践の鍵 | 自動化ツール + コードレビュー + 継続的改善 |
| 脳科学的根拠 | ワーキングメモリの制約に配慮したコードが理解しやすい |
| 導入戦略 | 段階的（認識→自動化→実践→文化→定着） |
| トレードオフ | パフォーマンス・締め切りとのバランスを状況に応じて判断 |

### クリーンコードの原則と実践の対応表

| 原則 | 対応する実践 | 測定指標 | ツール |
|------|------------|---------|--------|
| 可読性第一 | 命名規則、コメント最小化 | レビュー時間 | Code review metrics |
| DRY | 共通モジュール抽出 | 重複率 | jscpd, PMD CPD |
| KISS | 最小限の抽象化 | 認知的複雑度 | SonarQube |
| YAGNI | 要件駆動の実装 | 未使用コード率 | dead code analysis |
| SRP | 小さなクラス・関数 | クラスサイズ | Linter |
| 低結合 | DI、インターフェース | 依存関係の数 | deptry, madge |
| 高凝集 | ドメイン基準の構成 | LCOM | SonarQube |

---

## 次に読むべきガイド

- [SOLID原則](./01-solid.md) ── オブジェクト指向設計の5大原則
- [DRY/KISS/YAGNI](./02-dry-kiss-yagni.md) ── 重複排除と単純化の原則
- [結合度と凝集度](./03-coupling-cohesion.md) ── モジュール設計の基盤
- [デメテルの法則](./04-law-of-demeter.md) ── 最小知識の原則
- [命名規則](../01-practices/00-naming.md) ── 意図を伝える命名術
- [関数設計](../01-practices/01-functions.md) ── 単一責任・引数・副作用
- [エラーハンドリング](../01-practices/02-error-handling.md) ── 堅牢なエラー処理
- [コードスメル](../02-refactoring/00-code-smells.md) ── 問題のあるコードの兆候
- [デザインパターン](../../design-patterns-guide/00-creational/) ── 設計パターンの活用

---

## 参考文献

1. **Robert C. Martin** 『Clean Code: A Handbook of Agile Software Craftsmanship』 Prentice Hall, 2008
2. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 (2nd Edition)
3. **Steve McConnell** 『Code Complete: A Practical Handbook of Software Construction』 Microsoft Press, 2004 (2nd Edition)
4. **Dustin Boswell, Trevor Foucher** 『The Art of Readable Code』 O'Reilly Media, 2011
5. **John Ousterhout** 『A Philosophy of Software Design』 Yaknyam Press, 2018
6. **Thomas McCabe** "A Complexity Measure" IEEE Transactions on Software Engineering, 1976
7. **G. Ann Campbell** "Cognitive Complexity: A New Way of Measuring Understandability" SonarSource, 2018
8. **Ward Cunningham** "The WyCash Portfolio Management System" OOPSLA Experience Report, 1992 ── 技術的負債の原典
