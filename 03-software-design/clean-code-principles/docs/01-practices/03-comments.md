# コメント ── 良いコメント・悪いコメント・自己文書化コード

> 「コメントは、コードで表現できなかった己の失敗を補うもの」── Robert C. Martin。最良のコメントは書かなくて済むコメントだが、適切なコメントはコードの理解を大幅に助ける。コメントの良し悪しを見極め、自己文書化コードを書く力を身につける。

---

## この章で学ぶこと

1. **良いコメントの種類** ── 書くべきコメントのパターンを理解する
2. **悪いコメントの種類** ── 避けるべきコメントのアンチパターンを把握する
3. **自己文書化コードの書き方** ── コメントに頼らずコード自体が意図を伝える技法を身につける

---

## 1. コメントの基本方針

```
+-----------------------------------------------------------+
|  コメントの優先順位                                        |
|  ─────────────────────────────────────                    |
|  1st: コードで意図を表現する（命名、構造）                |
|  2nd: コードで表現できないことをコメントで補足             |
|  3rd: 外部ドキュメントに詳細を記載                        |
|                                                           |
|  「何をしているか (What)」はコードが語る                  |
|  「なぜそうしているか (Why)」をコメントが補足する         |
+-----------------------------------------------------------+
```

```
  コメントの必要度マトリクス

           コードが明確    コードが不明確
         ┌──────────────┬──────────────────┐
  意図が  │ コメント不要  │ コメント必要      │
  自明    │ x = x + 1    │ (まずコードを改善)│
         ├──────────────┼──────────────────┤
  意図が  │ Why コメント  │ コメント必須      │
  非自明  │ が有効        │ + コード改善も    │
         └──────────────┴──────────────────┘
```

---

## 2. 良いコメントの種類

**コード例1: なぜ（Why）を説明するコメント**

```python
# 良いコメント: 「なぜ」この実装なのかを説明
class RateLimiter:
    def should_allow(self, client_id: str) -> bool:
        # Sliding Window アルゴリズムを使用。
        # Fixed Window だとウィンドウ境界でバーストが発生するため。
        # 参考: https://blog.cloudflare.com/counting-things-a-lot-of-different-things/
        window_start = time.time() - self.window_size
        request_count = self.store.count_since(client_id, window_start)
        return request_count < self.max_requests

    def _cleanup_old_entries(self):
        # Redis のメモリ使用量を抑えるため、古いエントリを定期削除。
        # 本来TTLで自動削除されるが、大量のキーがある場合
        # Redis の lazy deletion が追いつかないことがある。
        cutoff = time.time() - (self.window_size * 2)
        self.store.delete_before(cutoff)
```

**コード例2: 法的コメント・ライセンス表記**

```java
/*
 * Copyright (c) 2024 Example Corp.
 * Licensed under the MIT License.
 * See LICENSE file in the project root for full license information.
 */
```

**コード例3: 警告コメント**

```python
# WARNING: この関数はスレッドセーフではない。
# マルチスレッド環境で使用する場合は外部で排他制御が必要。
def update_global_cache(key: str, value: any) -> None:
    global_cache[key] = value

# TODO: v2.0でOAuth2に移行予定。Basic認証は非推奨。
# Issue: https://github.com/example/app/issues/1234
def authenticate_basic(username: str, password: str) -> bool:
    pass

# HACK: MySQL 5.7のバグ(#12345)を回避するためのワークアラウンド。
# MySQL 8.0にアップグレード後に除去すること。
def query_with_workaround(sql: str) -> list:
    sql = sql.replace("GROUP BY", "GROUP BY 1, ")  # バグ回避
    return db.execute(sql)
```

**コード例4: 正規表現・複雑なアルゴリズムの説明**

```python
# RFC 5322準拠のメールアドレスバリデーション
# ローカル部: 英数字、ドット、ハイフン、アンダースコア、プラス
# ドメイン部: ラベル（英数字+ハイフン）をドットで接続
# 参考: https://datatracker.ietf.org/doc/html/rfc5322#section-3.4.1
EMAIL_PATTERN = re.compile(
    r'^[a-zA-Z0-9._%+-]+'    # ローカル部
    r'@'                       # @ 記号
    r'[a-zA-Z0-9.-]+'         # ドメイン名
    r'\.[a-zA-Z]{2,}$'        # トップレベルドメイン
)

# ダイクストラ法: 始点から全頂点への最短距離を計算
# 計算量: O((V + E) log V) where V=頂点数, E=辺数
# 負の重みがある場合はベルマンフォード法を使用すること
def dijkstra(graph: Graph, source: int) -> dict[int, float]:
    distances = {v: float('inf') for v in graph.vertices}
    distances[source] = 0
    priority_queue = [(0, source)]
    # ...
```

---

## 3. 悪いコメントの種類

**コード例5: 悪いコメント集**

```python
# ----- 悪いコメント1: コードをそのまま繰り返す -----
# ユーザー名を取得する
username = user.get_name()  # コードを見れば分かる

# カウンタをインクリメントする
counter += 1  # これは不要

# ----- 悪いコメント2: 嘘のコメント（コードと不一致）-----
# 偶数かどうかをチェック
if number % 2 != 0:  # 実際は奇数チェック！
    process_odd(number)

# ----- 悪いコメント3: コメントアウトされたコード -----
def calculate_total(items):
    total = sum(item.price for item in items)
    # tax = total * 0.08  # 旧税率
    # discount = total * 0.05 if is_member else 0
    # total = total + tax - discount
    # if apply_coupon:
    #     total = total * 0.9
    tax = total * 0.10
    return total + tax

# ----- 悪いコメント4: 属人的なコメント -----
# 田中さんに確認済み (2023/01/15)
# TODO: 佐藤くんが後でリファクタリングする

# ----- 悪いコメント5: セクション区切りとしてのコメント -----
##################################
# ユーザー関連の処理
##################################
# → クラスやモジュールで分離すべき
```

---

## 4. 自己文書化コードへの変換

| Before（コメント依存） | After（自己文書化） |
|----------------------|-------------------|
| `# 18歳以上かチェック` `if age >= 18:` | `if user.is_adult():` |
| `# 税込み価格を計算` `p * 1.10` | `calculate_price_with_tax(price)` |
| `# アクティブユーザーのみ` `if s == 1:` | `if user.status == Status.ACTIVE:` |
| `# 5回以上失敗でロック` `if c >= 5:` | `if login_attempts >= MAX_ATTEMPTS:` |

| テクニック | 説明 |
|-----------|------|
| Extract Method | コメントで説明していたブロックを関数に抽出 |
| Rename | より意図が伝わる名前に変更 |
| Introduce Constant | マジックナンバーに名前を付ける |
| Replace Conditional with Polymorphism | 条件分岐をポリモーフィズムに |

---

## 5. ドキュメンテーションコメント

```
  ドキュメンテーションコメントの構成

  ┌─────────────────────────────────────────┐
  │ 1行目: 何をするかの要約                  │
  │                                         │
  │ 詳細説明（必要な場合）                   │
  │                                         │
  │ Args/Parameters:                        │
  │   パラメータの説明                       │
  │                                         │
  │ Returns:                                │
  │   戻り値の説明                           │
  │                                         │
  │ Raises/Throws:                          │
  │   発生する例外の説明                     │
  │                                         │
  │ Examples: (任意)                         │
  │   使用例                                 │
  └─────────────────────────────────────────┘
```

```python
def transfer_funds(
    source: Account,
    destination: Account,
    amount: Decimal,
    currency: Currency = Currency.JPY
) -> TransferReceipt:
    """指定された金額を送金元から送金先に振り替える。

    同一通貨間の即時振替を実行する。異なる通貨間の
    振替は未対応（CurrencyMismatchError を送出）。

    Args:
        source: 送金元口座。残高が amount 以上必要。
        destination: 送金先口座。凍結されていないこと。
        amount: 振替金額。正の数であること。
        currency: 通貨。デフォルトは日本円。

    Returns:
        TransferReceipt: 取引IDとタイムスタンプを含む受領証。

    Raises:
        InsufficientBalanceError: 送金元の残高不足。
        AccountFrozenError: いずれかの口座が凍結されている。
        CurrencyMismatchError: 口座の通貨と指定通貨が不一致。

    Example:
        >>> receipt = transfer_funds(account_a, account_b, Decimal('10000'))
        >>> print(receipt.transaction_id)
        'TXN-20240101-001'
    """
```

---

## 6. アンチパターン

### アンチパターン1: コメントで設計の欠陥を隠す

```java
// アンチパターン: 複雑なロジックをコメントで説明
// ステータスが1（アクティブ）で、タイプが3（プレミアム）または
// タイプが4（エンタープライズ）で、最終ログインが30日以内の場合
if (user.status == 1 && (user.type == 3 || user.type == 4)
    && daysSince(user.lastLogin) <= 30) {

// 改善: コード自体が意図を語る
if (user.isActive() && user.isPremiumOrAbove() && user.isRecentlyActive()) {
```

### アンチパターン2: 変更履歴をコメントで管理

```python
# アンチパターン: ファイル内の変更履歴
# 変更履歴:
# 2024-01-01 田中: 初版作成
# 2024-02-15 佐藤: バリデーション追加
# 2024-03-20 鈴木: パフォーマンス改善
# 2024-04-10 田中: バグ修正 (#1234)
# → バージョン管理システム（Git）が担うべき役割
```

---

## 7. FAQ

### Q1: コメントは英語で書くべきか日本語で書くべきか？

チームの共通言語に合わせる。日本語チームなら日本語コメントで問題ない。ただし、OSSやグローバルチームでは英語が必須。**重要なのは一貫性**。同一プロジェクト内で言語を混在させない。

### Q2: TODOコメントはどう管理すべきか？

TODOコメントは**Issueトラッカーと紐づけて**管理する。`TODO(#1234): 〜` の形式でIssue番号を含め、定期的にTODOを棚卸しする。放置されたTODOは技術的負債になるため、CI/CDでTODOの数を監視するのも効果的。

### Q3: APIのドキュメンテーションコメントはどこまで書くべきか？

**公開API（public）には必須**。以下を含める:
- 何をするか（1行要約）
- パラメータの意味と制約
- 戻り値の型と意味
- 発生する例外/エラー
- 使用例（複雑な場合）

privateメソッドは、名前から意図が明確なら省略可。

---

## まとめ

| 種類 | 書くべきか | 例 |
|------|-----------|-----|
| Why（なぜ） | 書くべき | ビジネスルールの背景、技術的理由 |
| What（何を） | 原則不要 | コード自体が語るべき |
| How（どうやって） | アルゴリズムのみ | 複雑な正規表現、数学的処理 |
| 警告・注意 | 書くべき | スレッド安全性、パフォーマンス |
| TODO/FIXME | Issueと紐づけて | 期限と担当を明記 |
| ライセンス | 必須 | 法的要件 |

---

## 次に読むべきガイド

- [命名規則](./00-naming.md) ── コメント不要のコードを書く第一歩
- [関数設計](./01-functions.md) ── 自己文書化コードの構造
- [コードレビューチェックリスト](../03-practices-advanced/04-code-review-checklist.md) ── コメントのレビュー観点

---

## 参考文献

1. **Robert C. Martin** 『Clean Code: A Handbook of Agile Software Craftsmanship』 Prentice Hall, 2008 (Chapter 4: Comments)
2. **Dustin Boswell, Trevor Foucher** 『The Art of Readable Code』 O'Reilly Media, 2011 (Part II: Simplifying Loops and Logic)
3. **Kevlin Henney** "Comment Only What the Code Cannot Say" 『97 Things Every Programmer Should Know』 O'Reilly Media, 2010
