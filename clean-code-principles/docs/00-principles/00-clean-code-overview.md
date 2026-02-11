# クリーンコード概要 ── なぜコード品質が重要か

> ソフトウェアの総コストの80%以上は保守に費やされる。読みやすく変更しやすいコードは、チーム全体の生産性を劇的に向上させる。

---

## この章で学ぶこと

1. **クリーンコードの定義** ── 著名なエンジニアたちの視点から「良いコード」を理解する
2. **品質がビジネスに与える影響** ── 技術的負債と開発速度の関係を定量的に把握する
3. **クリーンコードの実践原則** ── 日常のコーディングで適用できる基本ルールを身につける

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
```

### 1.2 品質の多面的評価

```
          ┌─────────────────────────────────────────┐
          │         コード品質の4象限                 │
          ├──────────────┬──────────────────────────┤
          │  可読性       │  保守性                   │
          │  ・命名が明確 │  ・変更が局所的           │
          │  ・構造が一貫 │  ・テストが容易           │
          ├──────────────┼──────────────────────────┤
          │  信頼性       │  効率性                   │
          │  ・エラー処理 │  ・適切なアルゴリズム     │
          │  ・エッジケース│ ・不要な計算がない        │
          └──────────────┴──────────────────────────┘
```

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

**コード例1: 可読性の比較 ── 意図が読めないコード vs クリーンコード**

```python
# ダーティコード: 何をしているか読み取れない
def calc(l, t):
    r = []
    for i in l:
        if i['a'] > t and i['s'] == 1:
            r.append(i['n'])
    return r

# クリーンコード: 意図が明確
def find_active_users_above_threshold(users, age_threshold):
    """指定年齢以上のアクティブユーザー名を返す"""
    active_senior_users = []
    for user in users:
        if user['age'] > age_threshold and user['status'] == ACTIVE:
            active_senior_users.append(user['name'])
    return active_senior_users
```

**コード例2: 構造化されたエラーハンドリング**

```python
# ダーティコード: エラーが飲み込まれる
def get_user(id):
    try:
        return db.query(id)
    except:
        return None

# クリーンコード: エラーの意味が明確
def get_user_by_id(user_id: int) -> User:
    """ユーザーIDからユーザーを取得する。

    Raises:
        UserNotFoundError: ユーザーが見つからない場合
        DatabaseConnectionError: DB接続に失敗した場合
    """
    try:
        user = user_repository.find_by_id(user_id)
    except ConnectionError as e:
        raise DatabaseConnectionError(f"DB接続失敗: {e}") from e

    if user is None:
        raise UserNotFoundError(f"ユーザーID {user_id} は存在しません")

    return user
```

**コード例3: 単一責任の関数**

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
```

**コード例4: マジックナンバーの排除**

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
```

**コード例5: ガード節による早期リターン**

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
      result = normalPayAmount(employee);
    }
  }
  return result;
}

// クリーンコード: ガード節で平坦化
function getPayAmount(employee: Employee): number {
  if (employee.isSeparated) return separatedAmount(employee);
  if (employee.isRetired) return retiredAmount(employee);
  return normalPayAmount(employee);
}
```

---

## 3. クリーンコードの基本原則

| 原則 | 説明 | 効果 |
|------|------|------|
| 可読性第一 | コードは書く回数より読む回数が圧倒的に多い | 理解時間の短縮 |
| DRY | 同じロジックを繰り返さない | 変更箇所の一元化 |
| KISS | 複雑さを避け、シンプルに保つ | バグの予防 |
| YAGNI | 今必要でない機能は作らない | 無駄な開発の排除 |
| SRP | 1つの関数/クラスに1つの責任 | 変更影響の局所化 |
| 意図の明示 | 名前・構造でコードの目的を伝える | コメント依存の低減 |

---

## 4. コード品質の測定

| 指標 | 説明 | 目安 |
|------|------|------|
| サイクロマティック複雑度 | 分岐数に基づく複雑さ | 関数あたり10以下 |
| 認知的複雑度 | 人間が感じる理解しにくさ | 関数あたり15以下 |
| コードカバレッジ | テストで実行されるコードの割合 | 80%以上 |
| 重複率 | コピペされたコードの割合 | 5%以下 |
| 関数の行数 | 1関数の物理行数 | 20行以下推奨 |
| 依存関係の深さ | モジュール間の依存段数 | 3段以下 |

---

## 5. アンチパターン

### アンチパターン1: 過度な最適化（Premature Optimization）

```python
# アンチパターン: 読みにくい最適化を早期に行う
def f(d):
    return {k: v for k, v in sorted(
        ((k, sum(x['v'] for x in g))
         for k, g in __import__('itertools').groupby(
             sorted(d, key=lambda x: x['k']),
             key=lambda x: x['k'])),
        key=lambda x: -x[1])}

# 改善: まず可読性を優先し、必要時に最適化
def aggregate_and_sort_by_value(data: list[dict]) -> dict:
    """キーごとに値を集計し、降順で返す"""
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
// アンチパターン: コメントで複雑さを説明
// i は顧客のインデックス、j は注文のインデックス、
// k は商品のインデックス、t は合計金額
for (int i = 0; i < c.length; i++) {
    for (int j = 0; j < c[i].o.length; j++) {
        for (int k = 0; k < c[i].o[j].p.length; k++) {
            t += c[i].o[j].p[k].pr * c[i].o[j].p[k].q;
        }
    }
}

// 改善: コード自体が意味を語る
for (Customer customer : customers) {
    for (Order order : customer.getOrders()) {
        for (Product product : order.getProducts()) {
            totalRevenue += product.getPrice() * product.getQuantity();
        }
    }
}
```

---

## 6. FAQ

### Q1: クリーンコードを書くと開発が遅くならないか？

短期的には命名や構造の検討に時間がかかる。しかし中長期的には、読む時間・デバッグ時間・変更時間が大幅に削減され、**トータルの生産性は向上する**。IBMの調査では、コードレビューにより後工程の欠陥修正コストが10〜100倍削減されることが示されている。

### Q2: レガシーコードはどこからクリーンにすべきか？

**ボーイスカウトルール**（来た時よりも綺麗にして帰る）に従い、触ったファイルを少しずつ改善する。全面書き換えではなく、テストを追加しながら段階的にリファクタリングする。優先度は「頻繁に変更されるファイル」から。

### Q3: チーム全体でクリーンコードを徹底するには？

1. **コーディング規約の策定と自動化**（Linter、Formatter）
2. **コードレビュー文化の醸成**（レビューチェックリストの活用）
3. **ペアプログラミング/モブプログラミング**の導入
4. **技術的負債の可視化**（品質ダッシュボード）

---

## まとめ

| 項目 | 内容 |
|------|------|
| クリーンコードの本質 | 読みやすく、理解しやすく、変更しやすいコード |
| ビジネス効果 | 保守コスト削減、開発速度の維持、バグの予防 |
| 基本原則 | 可読性第一、DRY、KISS、YAGNI、SRP |
| 測定方法 | 複雑度、カバレッジ、重複率、関数サイズ |
| 実践の鍵 | 自動化ツール + コードレビュー + 継続的改善 |

---

## 次に読むべきガイド

- [SOLID原則](./01-solid.md) ── オブジェクト指向設計の5大原則
- [DRY/KISS/YAGNI](./02-dry-kiss-yagni.md) ── 重複排除と単純化の原則
- [命名規則](../01-practices/00-naming.md) ── 意図を伝える命名術

---

## 参考文献

1. **Robert C. Martin** 『Clean Code: A Handbook of Agile Software Craftsmanship』 Prentice Hall, 2008
2. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 (2nd Edition)
3. **Steve McConnell** 『Code Complete: A Practical Handbook of Software Construction』 Microsoft Press, 2004 (2nd Edition)
4. **Dustin Boswell, Trevor Foucher** 『The Art of Readable Code』 O'Reilly Media, 2011
