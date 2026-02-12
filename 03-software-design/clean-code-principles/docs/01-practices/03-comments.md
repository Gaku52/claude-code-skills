# コメント ── 良いコメント・悪いコメント・自己文書化コード

> 「コメントは、コードで表現できなかった己の失敗を補うもの」── Robert C. Martin。最良のコメントは書かなくて済むコメントだが、適切なコメントはコードの理解を大幅に助ける。コメントの良し悪しを見極め、自己文書化コードを書く力を身につける。

---

## この章で学ぶこと

1. **良いコメントの種類と書き方** ── Why コメント、警告コメント、ドキュメンテーションコメントなど、書くべきコメントのパターンと具体的な記述技法を理解する
2. **悪いコメントの種類と排除方法** ── 冗長コメント、嘘コメント、コメントアウトコード等のアンチパターンを認識し、コードベースから排除する方法を身につける
3. **自己文書化コードの設計技法** ── 命名、構造化、型システムを駆使してコメントに頼らずコード自体が意図を伝える技法を習得する
4. **ドキュメンテーションコメントの設計** ── 公開APIに対する効果的なドキュメンテーションコメントの書き方と各言語での標準形式を身につける
5. **チームにおけるコメント戦略** ── コメントポリシーの策定、レビュー基準、自動チェックの導入方法を学ぶ

---

## 前提知識

この章を理解するために、以下の知識があると望ましい。

| 前提知識 | 参照先 |
|---------|--------|
| 命名規則の基本 | [命名規則](./00-naming.md) |
| 関数設計の原則 | [関数設計](./01-functions.md) |
| クラス設計の基本 | [クラス設計](./02-classes.md) |
| Git の基本操作 | バージョン管理の基礎知識 |

---

## 1. コメントの基本方針 ── なぜコメントの質が重要なのか

### 1.1 コメントのコスト

コメントは「無料」ではない。コメントには以下の隠れたコストがある。

```
コメントの隠れたコスト
────────────────────────────────────
1. 保守コスト: コードを変更するたびにコメントも更新が必要
2. 認知コスト: 読む人はコードとコメントの両方を処理する必要がある
3. 信頼コスト: 嘘のコメントはコメント全体への信頼を毀損する
4. 誘導コスト: 悪いコメントは読み手を間違った方向に誘導する

コメントが多い ≠ 良いコード
コメントが少ない ≠ 悪いコード
────────────────────────────────────
適切なコメントが、適切な場所にあることが重要
```

Robert C. Martin が Clean Code で繰り返し強調するのは、「コメントを書く前にコードを改善できないか考えよ」という原則である。コメントを書くこと自体がコードの設計不足を示唆している可能性がある。しかし同時に、コードだけでは伝えられない情報（ビジネス上の理由、技術的制約、歴史的経緯）は確実に存在し、そこにこそコメントの真の価値がある。

コメントの本質を理解するには、「コードは How（どうやっているか）を伝えるが、Why（なぜそうしているか）を伝えるのは苦手」という点を押さえる必要がある。Why を伝えることがコメントの最大の役割である。

### 1.2 コメントの優先順位

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

この優先順位がなぜ重要かを掘り下げる。コードで意図を表現する（1st）が最優先な理由は、コードはコンパイラ/インタプリタによって正確性が検証されるが、コメントは誰にも検証されないためである。コードが変更されてもコメントが自動で追従することはない。したがって、情報の正確性を保証できるコードに、可能な限り情報を載せるべきなのである。

### 1.3 コメントの必要度マトリクス

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

  判断フロー:
  1. コードだけで意図が伝わるか？ → Yes → コメント不要
  2. コードを改善できるか？      → Yes → まず改善、それでも不足ならコメント
  3. WHY を説明する必要があるか？ → Yes → Why コメントを追加
  4. 警告・制約があるか？        → Yes → 警告コメントを追加
```

### 1.4 コメントの分類全体像

```
  コメントの全体分類図

  コメント
  ├── 良いコメント（書くべき）
  │   ├── Why コメント ── ビジネスルール、技術的理由の説明
  │   ├── 警告コメント ── スレッド安全性、パフォーマンス上の注意
  │   ├── TODO/FIXME ── 将来の改善予定（Issue紐づけ必須）
  │   ├── ライセンスコメント ── 法的要件
  │   ├── ドキュメンテーションコメント ── 公開 API の説明
  │   └── 複雑なアルゴリズムの説明 ── 正規表現、数学的処理
  │
  ├── 悪いコメント（避けるべき）
  │   ├── What コメント ── コードの繰り返し
  │   ├── 嘘のコメント ── コードとの不一致
  │   ├── コメントアウトコード ── Git が担うべき役割
  │   ├── 属人的コメント ── 特定個人への依存
  │   ├── 変更履歴コメント ── VCS が担うべき役割
  │   ├── 区切りコメント ── 構造で分離すべき
  │   └── ジャーナルコメント ── 日記的なメモ
  │
  └── 自己文書化コード（コメントを不要にする技法）
      ├── 意味のある命名
      ├── Extract Method
      ├── 定数の導入
      ├── 型システムの活用
      └── ポリモーフィズムの活用
```

---

## 2. 良いコメントの種類

### 2.1 Why（なぜ）を説明するコメント

最も価値の高いコメントは「なぜこの実装なのか」を説明するものである。コードからは「何をしているか」は読み取れるが、「なぜそうしているか」は読み取れない。Why コメントが特に有効なケースは以下の通りである。

- **ビジネスルールの背景**: なぜこの値、この条件、この処理順序なのか
- **技術的制約の理由**: なぜこのライブラリ、このアルゴリズム、この回避策なのか
- **歴史的経緯**: なぜ直感に反する実装になっているのか
- **トレードオフの記録**: なぜ別の選択肢ではなくこの方式を選んだのか

**コード例1: ビジネスルールの背景を説明する Why コメント**

```python
class RateLimiter:
    def should_allow(self, client_id: str) -> bool:
        # Sliding Window アルゴリズムを使用。
        # Fixed Window だとウィンドウ境界でバーストが発生するため。
        # 例: Fixed Window で100リクエスト/分の場合、
        # 59秒目に100リクエスト + 次の分の0秒目に100リクエスト
        # = 実質1秒間に200リクエスト通過してしまう。
        # 参考: https://blog.cloudflare.com/counting-things-a-lot-of-different-things/
        window_start = time.time() - self.window_size
        request_count = self.store.count_since(client_id, window_start)
        return request_count < self.max_requests

    def _cleanup_old_entries(self):
        # Redis のメモリ使用量を抑えるため、古いエントリを定期削除。
        # 本来TTLで自動削除されるが、大量のキーがある場合
        # Redis の lazy deletion が追いつかないことがある。
        # (Redis 6.0 以降は active-expire-effort で調整可能だが、
        #  パフォーマンスへの影響が読みにくいため明示的に削除する方針)
        cutoff = time.time() - (self.window_size * 2)
        self.store.delete_before(cutoff)
```

**コード例2: 技術的制約の Why コメント**

```python
class PaymentProcessor:
    def process(self, payment: Payment) -> PaymentResult:
        # Stripe API はべき等キーによる重複防止をサポート。
        # ネットワーク障害時のリトライで二重課金を防止するため、
        # 注文IDをべき等キーとして使用する。
        # 参考: https://stripe.com/docs/api/idempotent_requests
        idempotency_key = f"payment-{payment.order_id}"

        try:
            charge = stripe.Charge.create(
                amount=payment.amount,
                currency="jpy",
                idempotency_key=idempotency_key,
            )
            return PaymentResult.success(charge.id)
        except stripe.error.CardError as e:
            return PaymentResult.failure(str(e))
```

**コード例3: パフォーマンス上の理由を説明する Why コメント**

```java
public class UserSearchService {
    public List<User> searchByName(String query) {
        // 全文検索インデックスではなく LIKE 検索を使用。
        // 理由: ユーザー数が1万件未満のため、Elasticsearch の
        // 運用コスト (インフラ費用、運用工数) に見合わない。
        // ユーザー数が10万件を超えた場合は全文検索への移行を検討すること。
        // Issue: https://github.com/example/app/issues/567
        return userRepository.findByNameLike("%" + query + "%");
    }

    public List<User> findActive() {
        // ソート済みリストを返す。
        // UI側でページネーション + 無限スクロールを使用しており、
        // 一貫した順序が保証されないとスクロール時にアイテムが
        // 重複・欠落するため。
        return userRepository.findByStatus(Status.ACTIVE, Sort.by("id"));
    }
}
```

**コード例4: トレードオフの記録**

```python
class SessionStore:
    def __init__(self):
        # Redis ではなくインメモリ辞書を使用。
        # トレードオフ:
        #   利点: 外部依存なし、レイテンシ最小、開発環境構築が容易
        #   欠点: プロセス再起動でセッション消失、水平スケール不可
        # 現状はシングルインスタンスでの運用のため許容。
        # マルチインスタンス構成への移行時に Redis に切り替える。
        # 判断日: 2024-01-15, 判断者: アーキテクチャレビュー会議
        self._sessions: dict[str, Session] = {}
```

### 2.2 法的コメント・ライセンス表記

法的に必要なコメントは省略してはならない。OSS ライセンスの遵守は法的義務であり、コメントの省略は著作権侵害に繋がる可能性がある。

```java
/*
 * Copyright (c) 2024 Example Corp.
 * Licensed under the MIT License.
 * See LICENSE file in the project root for full license information.
 */
```

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Example Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
```

SPDX (Software Package Data Exchange) 形式のライセンス表記は、機械可読性が高く推奨される。SPDX 識別子の一覧は https://spdx.org/licenses/ を参照。

### 2.3 警告コメント

後続の開発者への重要な注意事項は必ずコメントで残す。警告コメントが特に必要なケースは以下の通りである。

- **スレッド安全性**: 並行実行時の制約
- **パフォーマンス**: 処理時間、メモリ使用量の注意
- **副作用**: 予期しない影響を及ぼす可能性
- **処理順序**: 順序を変更してはいけない理由

**コード例5: 警告コメントのパターン集**

```python
# WARNING: この関数はスレッドセーフではない。
# マルチスレッド環境で使用する場合は外部で排他制御が必要。
# threading.Lock() または concurrent.futures の使用を推奨。
def update_global_cache(key: str, value: any) -> None:
    global_cache[key] = value

# CAUTION: この処理は平均2秒かかる（最大10秒）。
# ユーザー向けのリクエストパスでは使用せず、
# バックグラウンドジョブ (Celery等) 経由で呼び出すこと。
# 直接呼び出すとリクエストタイムアウトが発生する。
def rebuild_search_index() -> None:
    pass

# NOTE: この定数は外部 API の仕様に基づく。
# 変更する場合は API ドキュメントを確認すること。
# https://api.example.com/docs#rate-limits
MAX_REQUESTS_PER_MINUTE = 60

# IMPORTANT: 以下の処理順序を変更しないこと。
# 在庫確認 → 決済 → 在庫引き当ての順序でないと
# 在庫の二重引き当てが発生する。
# 過去にこの順序を変えて本番障害が発生（2023-06 P1 インシデント）。
def process_order(order: Order) -> None:
    check_inventory(order)      # 1. 在庫確認
    process_payment(order)      # 2. 決済
    reserve_inventory(order)    # 3. 在庫引き当て
```

```
警告コメントのプレフィックス規約
────────────────────────────────────
  WARNING   : 重大な問題を引き起こす可能性がある注意事項
  CAUTION   : パフォーマンス・リソースに関する注意
  NOTE      : 補足情報。知っておくと有用な情報
  IMPORTANT : 変更してはいけない制約
  SECURITY  : セキュリティに関する注意事項
────────────────────────────────────
```

### 2.4 TODO / FIXME / HACK コメント

将来の改善予定を記録するコメント。必ずIssueトラッカーと紐づける。

**コード例6: TODO/FIXME/HACK の適切な使い方**

```python
# TODO(#1234): v2.0でOAuth2に移行予定。Basic認証は非推奨。
# 期限: 2025-Q2
# 担当: auth-team
def authenticate_basic(username: str, password: str) -> bool:
    pass

# FIXME(#2345): 大量データ（100万件以上）で OOM が発生する。
# 原因: 全件メモリ展開しているため。
# 対策: ストリーミング処理に変更する。
def export_all_users() -> list:
    return db.query("SELECT * FROM users")

# HACK(#3456): MySQL 5.7のバグ(#12345)を回避するためのワークアラウンド。
# MySQL 8.0にアップグレード後に除去すること。
# 参考: https://bugs.mysql.com/bug.php?id=12345
def query_with_workaround(sql: str) -> list:
    sql = sql.replace("GROUP BY", "GROUP BY 1, ")
    return db.execute(sql)

# OPTIMIZE(#4567): N+1 クエリが発生している。
# 現状はデータ量が少ないため許容しているが、
# テナント数が100を超えたら JOIN クエリに変更すること。
def get_tenant_users(tenant_ids: list[str]) -> list[User]:
    result = []
    for tid in tenant_ids:
        result.extend(db.query("SELECT * FROM users WHERE tenant_id = %s", tid))
    return result
```

```
TODO コメントの書式ルール
────────────────────────────────────
形式: # TODO(#<Issue番号>): <説明>
必須: Issue番号、簡潔な説明
推奨: 期限、担当チーム、参考リンク

  プレフィックス一覧:
  TODO     : 将来実装すべき機能・改善
  FIXME    : 既知のバグ・不具合
  HACK     : 一時的な回避策（除去予定）
  OPTIMIZE : パフォーマンス改善の余地
  REVIEW   : レビューで再検討が必要な箇所

  悪い例: # TODO: あとで直す
  良い例: # TODO(#1234): v2.0 で OAuth2 に移行。Basic認証は 2025-Q2 で廃止予定。
────────────────────────────────────
```

### 2.5 正規表現・複雑なアルゴリズムの説明

正規表現やアルゴリズムは、コード自体が「What」を伝えにくい代表例である。これらのコメントは What であっても正当化される。

**コード例7: 正規表現の詳細コメント**

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

# 日本の電話番号バリデーション（固定電話 + 携帯電話）
# 形式1: 03-1234-5678（市外局番-市内局番-加入者番号）
# 形式2: 090-1234-5678（携帯電話）
# 形式3: 0120-123-456（フリーダイヤル）
# ハイフンあり/なし両方に対応
PHONE_PATTERN = re.compile(
    r'^0'                      # 先頭は 0
    r'[0-9]{1,4}'             # 市外局番（1-4桁）
    r'-?'                      # ハイフン（任意）
    r'[0-9]{1,4}'             # 市内局番（1-4桁）
    r'-?'                      # ハイフン（任意）
    r'[0-9]{3,4}$'            # 加入者番号（3-4桁）
)

# クレジットカード番号のマスキング
# 先頭6桁（BIN）と末尾4桁を残し、中間をマスク
# 例: 4111-1111-1111-1111 → 411111******1111
# PCI DSS 準拠: 表示時は先頭6桁+末尾4桁まで
MASK_PATTERN = re.compile(r'^(\d{6})\d+(\d{4})$')
```

**コード例8: 複雑なアルゴリズムの説明コメント**

```python
# ダイクストラ法: 始点から全頂点への最短距離を計算
# 計算量: O((V + E) log V) where V=頂点数, E=辺数
# 負の重みがある場合はベルマンフォード法を使用すること
#
# アルゴリズムの概要:
# 1. 始点の距離を0、他の全頂点の距離を inf に初期化
# 2. 未確定頂点のうち距離最小のものを選択（優先度キュー使用）
# 3. 選択した頂点の隣接頂点の距離を更新（緩和操作）
# 4. 全頂点が確定するまで 2-3 を繰り返す
def dijkstra(graph: Graph, source: int) -> dict[int, float]:
    distances = {v: float('inf') for v in graph.vertices}
    distances[source] = 0
    priority_queue = [(0, source)]
    visited = set()

    while priority_queue:
        current_dist, u = heapq.heappop(priority_queue)

        if u in visited:
            continue
        visited.add(u)

        for v, weight in graph.neighbors(u):
            # 緩和操作: 既知の距離より短いパスが見つかったら更新
            new_dist = current_dist + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(priority_queue, (new_dist, v))

    return distances
```

### 2.6 公開APIの意図説明コメント

```python
class EventBus:
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """イベントハンドラを登録する。

        同一イベントに複数のハンドラを登録可能。
        ハンドラは登録順に呼び出される（FIFO）。

        # なぜ順序保証するのか:
        # 監査ログハンドラがビジネスロジックハンドラより先に
        # 呼ばれる必要があるユースケースがあるため。
        # 順序に依存しない設計が望ましいが、現状の要件では必要。

        Args:
            event_type: イベントの種類を示す文字列。
            handler: イベント発生時に呼び出されるコールバック関数。

        Raises:
            ValueError: event_type が空文字の場合。
            TypeError: handler が callable でない場合。
        """
        self._handlers.setdefault(event_type, []).append(handler)
```

---

## 3. 悪いコメントの種類

### 3.1 What コメント（コードの繰り返し）

コードを日本語に翻訳しただけのコメントは、情報量がゼロであり、保守コストだけが発生する。このタイプのコメントはコードベース内で最も多く見られる悪いコメントである。

**コード例9: What コメントの例と改善**

```python
# -----  NG: コードをそのまま繰り返す -----

# ユーザー名を取得する
username = user.get_name()  # コードを見れば分かる

# カウンタをインクリメントする
counter += 1  # これは不要

# リストに追加する
items.append(new_item)  # 自明すぎる

# null チェック
if user is not None:  # コードそのまま
    process(user)


# ----- OK: コメントが不要なコード -----

username = user.get_name()
counter += 1
items.append(new_item)
if user is not None:
    process(user)

# もし「なぜ」がある場合のみコメントする:
# ユーザーが削除済みの場合でもグレースピリオド中は
# プロフィール表示が必要なため、None チェックが必要。
if user is not None:
    process(user)
```

### 3.2 嘘のコメント（コードとの不一致）

コメントとコードが矛盾している場合、最も危険な状態になる。読み手はコメントを信じてしまうが、実際の動作はコードが決定する。嘘のコメントは意図的に書かれることは少なく、コードの変更時にコメントの更新が漏れることで生まれる。

**コード例10: 嘘のコメントの検出と修正**

```python
# NG: 嘘のコメント（条件の説明が逆）
# 偶数かどうかをチェック
if number % 2 != 0:  # 実際は奇数チェック！
    process_odd(number)

# NG: 古くなったコメント（値の変更がコメントに反映されていない）
# 最大リトライ回数は3回
MAX_RETRIES = 5  # いつの間にか5に変更されている

# NG: 引数の説明が実態と不一致
def send_email(
    to: str,       # 送信先メールアドレス
    subject: str,  # 件名
    body: str,     # 本文（プレーンテキスト） ← 実際は HTML も送れる
) -> bool:
    pass

# NG: 戻り値の説明が間違っている
def find_user(user_id: str):
    """ユーザーを取得する。見つからない場合は None を返す。"""
    # 実際は見つからない場合 UserNotFoundError を raise する
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    if not user:
        raise UserNotFoundError(user_id)
    return user


# OK: コメントを修正するか、コードを改善する
def is_odd(number: int) -> bool:
    """数値が奇数かどうかを判定する。"""
    return number % 2 != 0

if is_odd(number):
    process_odd(number)
```

### 3.3 コメントアウトされたコード

コメントアウトされたコードは、読み手に「削除して大丈夫なのか？」という不安を与え、コードベースにノイズを蓄積させる。Git に履歴が残っているため、コメントアウトではなく削除すべきである。

```python
# NG: コメントアウトされたコードの放置
def calculate_total(items):
    total = sum(item.price for item in items)
    # tax = total * 0.08  # 旧税率
    # discount = total * 0.05 if is_member else 0
    # total = total + tax - discount
    # if apply_coupon:
    #     total = total * 0.9
    tax = total * 0.10
    return total + tax

# なぜダメか:
# 1. 「削除して大丈夫なのか？」と読み手が悩む
# 2. 何の目的で残されているのか不明
# 3. コードの流れが読みにくくなる
# 4. Git に履歴があるので復元可能 → 削除すべき
```

```
コメントアウトコードの正しい対処法
────────────────────────────────────
1. 必要なら Git の履歴から復元できる → 削除
2. 将来使う予定がある → Issue を起票して削除
3. デバッグ用 → デバッグ完了後に削除
4. 代替実装の参考 → 外部ドキュメントに移動して削除
5. A/B テスト中 → Feature Flag で分岐（コメントアウト不要）
────────────────────────────────────
原則: コメントアウトコードは見つけ次第削除する
```

### 3.4 属人的なコメント

```python
# NG: 属人的なコメント ── 個人に依存する情報
# 田中さんに確認済み (2023/01/15)
# TODO: 佐藤くんが後でリファクタリングする
# この部分は山田さんしか分からない
# 鈴木さんの要望で追加した機能

# OK: 客観的な情報として記載
# 2023-01-15 承認済み: 金融庁ガイドライン準拠の確認完了 (Issue #789)
# TODO(#1234): パフォーマンス改善のためクエリをリファクタリング
# NOTE: このロジックの仕様は docs/pricing-algorithm.md を参照
# 要件: チケット #567 で追加された割引機能
```

### 3.5 変更履歴コメント

```python
# NG: ファイル内の変更履歴（Git が担うべき役割）
# 変更履歴:
# 2024-01-01 田中: 初版作成
# 2024-02-15 佐藤: バリデーション追加
# 2024-03-20 鈴木: パフォーマンス改善
# 2024-04-10 田中: バグ修正 (#1234)

# 代替手段: Git コマンドで確認
# → git log --oneline -- path/to/file.py
# → git blame path/to/file.py
# → git log --author="田中" -- path/to/file.py
```

### 3.6 セクション区切りコメント

```python
# NG: コメントで構造を区切る（God Class の兆候）
class UserService:
    ##################################
    # ユーザー関連の処理
    ##################################
    def create_user(self): ...
    def update_user(self): ...
    def delete_user(self): ...

    ##################################
    # 認証関連の処理
    ##################################
    def login(self): ...
    def logout(self): ...

    ##################################
    # 通知関連の処理
    ##################################
    def send_email(self): ...
    def send_sms(self): ...

# OK: クラスやモジュールで責務を分離する
class UserService:
    def create_user(self): ...
    def update_user(self): ...

class AuthService:
    def login(self): ...
    def logout(self): ...

class NotificationService:
    def send_email(self): ...
    def send_sms(self): ...
```

セクション区切りコメントが必要になること自体が、そのクラス/関数が大きすぎるというコードスメルの兆候である。詳しくは [コードスメル](../02-refactoring/00-code-smells.md) を参照。

### 3.7 ノイズコメント

コードが自明であるにもかかわらず、何かを書かなければならないという義務感から書かれるコメント。ドキュメンテーションツールの要件で「全 public メソッドに docstring 必須」とした結果、無意味な docstring が量産されることがある。

```python
# NG: ノイズコメント
class User:
    def __init__(self):
        """デフォルトコンストラクタ"""  # 何も情報を追加していない
        pass

    def get_name(self) -> str:
        """名前を取得する"""  # メソッド名と同じ
        return self.name

    def set_name(self, name: str) -> None:
        """名前を設定する"""  # メソッド名と同じ
        self.name = name

    def is_active(self) -> bool:
        """アクティブかどうかを返す"""  # メソッド名の翻訳
        return self.status == Status.ACTIVE
```

---

## 4. 自己文書化コードへの変換

自己文書化コード (Self-Documenting Code) とは、コメントがなくても意図が明確に伝わるコードのことである。Robert C. Martin は「コメントを書く前に、コードを改善してコメントを不要にできないか考えよ」と述べている。この節では、コメント依存のコードを自己文書化コードに変換する具体的なテクニックを解説する。

### 4.1 変換テクニック一覧

| テクニック | 説明 | 効果 |
|-----------|------|------|
| Extract Method | コメントで説明していたブロックを関数に抽出 | コメントが関数名になる |
| Rename | より意図が伝わる名前に変更 | コメントが不要になる |
| Introduce Constant | マジックナンバーに名前を付ける | 値の意味が自明になる |
| Introduce Explaining Variable | 複雑な式に名前を付ける | 中間結果の意味が明確になる |
| Replace Conditional with Guard Clause | 早期リターンでネストを減らす | 条件の意図が明確になる |
| Replace Conditional with Polymorphism | 条件分岐をポリモーフィズムに | 型による振り分けが自明になる |
| Replace Type Code with Class | 文字列/数値コードをクラスに | ドメイン概念が型として表現される |
| Introduce Parameter Object | 関連パラメータをオブジェクトに | パラメータの関係が明確になる |

### 4.2 Before / After 比較表

| Before（コメント依存） | After（自己文書化） |
|----------------------|-------------------|
| `# 18歳以上かチェック` `if age >= 18:` | `if user.is_adult():` |
| `# 税込み価格を計算` `p * 1.10` | `calculate_price_with_tax(price)` |
| `# アクティブユーザーのみ` `if s == 1:` | `if user.status == Status.ACTIVE:` |
| `# 5回以上失敗でロック` `if c >= 5:` | `if login_attempts >= MAX_ATTEMPTS:` |
| `# 30日以内にログイン` `if d <= 30:` | `if user.is_recently_active():` |
| `# 合計が1万円以上` `if t >= 10000:` | `if order.qualifies_for_free_shipping():` |
| `# メールフォーマットチェック` `if re.match(...)` | `if Email.is_valid(address):` |
| `# 営業日のみ` `if w not in [5, 6]:` | `if date.is_business_day():` |

### 4.3 Extract Method による自己文書化

**コード例11: コメントブロックを関数に変換**

```python
# BEFORE: コメントで各セクションを説明
def process_order(order: Order) -> OrderResult:
    # バリデーション
    if not order.items:
        raise ValidationError("商品が選択されていません")
    if not order.customer_id:
        raise ValidationError("顧客情報がありません")
    for item in order.items:
        if item.quantity <= 0:
            raise ValidationError(f"数量が不正: {item.name}")

    # 合計計算
    subtotal = sum(item.price * item.quantity for item in order.items)
    tax = subtotal * Decimal("0.10")
    shipping = Decimal("500") if subtotal < Decimal("5000") else Decimal("0")
    total = subtotal + tax + shipping

    # 保存
    order.total = total
    db.save(order)

    # 通知
    email_service.send_confirmation(order)

    return OrderResult.success(order)


# AFTER: 関数名がコメントの代わりになる
def process_order(order: Order) -> OrderResult:
    validate_order(order)
    pricing = calculate_order_pricing(order)
    saved_order = save_order(order, pricing)
    send_order_confirmation(saved_order)
    return OrderResult.success(saved_order)

def validate_order(order: Order) -> None:
    """注文の入力値を検証する。"""
    if not order.items:
        raise ValidationError("商品が選択されていません")
    if not order.customer_id:
        raise ValidationError("顧客情報がありません")
    for item in order.items:
        if item.quantity <= 0:
            raise ValidationError(f"数量が不正: {item.name}")

def calculate_order_pricing(order: Order) -> OrderPricing:
    """注文の価格情報を算出する。"""
    subtotal = sum(item.price * item.quantity for item in order.items)
    tax = subtotal * TAX_RATE
    shipping = calculate_shipping_fee(subtotal)
    return OrderPricing(subtotal=subtotal, tax=tax, shipping=shipping)

FREE_SHIPPING_THRESHOLD = Decimal("5000")
STANDARD_SHIPPING_FEE = Decimal("500")

def calculate_shipping_fee(subtotal: Decimal) -> Decimal:
    """小計に基づいて送料を算出する。"""
    if subtotal >= FREE_SHIPPING_THRESHOLD:
        return Decimal("0")
    return STANDARD_SHIPPING_FEE
```

### 4.4 定数の導入によるマジックナンバー排除

**コード例12: Introduce Constant**

```python
# BEFORE: マジックナンバーにコメント
def check_password(password: str) -> bool:
    if len(password) < 8:      # 最小文字数
        return False
    if len(password) > 128:    # 最大文字数
        return False
    if not re.search(r'[A-Z]', password):  # 大文字必須
        return False
    if not re.search(r'[0-9]', password):  # 数字必須
        return False
    return True


# AFTER: 定数名が説明する
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
UPPERCASE_PATTERN = re.compile(r'[A-Z]')
DIGIT_PATTERN = re.compile(r'[0-9]')

def check_password(password: str) -> bool:
    if len(password) < MIN_PASSWORD_LENGTH:
        return False
    if len(password) > MAX_PASSWORD_LENGTH:
        return False
    if not UPPERCASE_PATTERN.search(password):
        return False
    if not DIGIT_PATTERN.search(password):
        return False
    return True
```

### 4.5 説明用変数の導入

**コード例13: Introduce Explaining Variable**

```python
# BEFORE: 複雑な条件にコメント
def should_send_reminder(user: User, order: Order) -> bool:
    # プレミアム以上のアクティブユーザーで、
    # 最終注文から30日以上経過し、
    # 通知設定がオンの場合にリマインダーを送る
    return (user.tier in ('premium', 'enterprise')
            and user.status == 'active'
            and (datetime.now() - order.last_order_date).days > 30
            and user.notification_enabled)


# AFTER: 説明用変数で条件の意図を表現
INACTIVITY_THRESHOLD_DAYS = 30

def should_send_reminder(user: User, order: Order) -> bool:
    is_high_tier_user = user.tier in ('premium', 'enterprise')
    is_active = user.status == 'active'
    days_since_last_order = (datetime.now() - order.last_order_date).days
    is_inactive_buyer = days_since_last_order > INACTIVITY_THRESHOLD_DAYS
    accepts_notifications = user.notification_enabled

    return (is_high_tier_user
            and is_active
            and is_inactive_buyer
            and accepts_notifications)
```

### 4.6 型システムによる自己文書化

型は「コンパイラが検証するコメント」である。型で表現した制約は、コメントと違って陳腐化せず、IDE の補完やエラー検出にも活用される。

**コード例14: 型で意図を表現する**

```python
# BEFORE: コメントで型の意味を補足
def create_user(
    name: str,       # ユーザー名（2-50文字）
    email: str,      # メールアドレス（RFC 5322準拠）
    age: int,        # 年齢（0-150）
    role: str,       # ロール（"admin", "user", "viewer"）
) -> dict:           # ユーザー情報辞書
    pass


# AFTER: 型がドキュメントになる
class UserName:
    """2-50文字のユーザー名を表す値オブジェクト。"""
    def __init__(self, value: str):
        if not (2 <= len(value) <= 50):
            raise ValueError(f"ユーザー名は2-50文字: {len(value)}文字")
        self.value = value

class Email:
    """RFC 5322準拠のメールアドレスを表す値オブジェクト。"""
    def __init__(self, value: str):
        if not EMAIL_PATTERN.match(value):
            raise ValueError(f"不正なメールアドレス: {value}")
        self.value = value

class Age:
    """0-150の年齢を表す値オブジェクト。"""
    def __init__(self, value: int):
        if not (0 <= value <= 150):
            raise ValueError(f"不正な年齢: {value}")
        self.value = value

class Role(Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

@dataclass
class User:
    name: UserName
    email: Email
    age: Age
    role: Role

def create_user(name: UserName, email: Email, age: Age, role: Role) -> User:
    # コメント不要 ── 型が全てを語る
    return User(name=name, email=email, age=age, role=role)
```

### 4.7 ガード節（Early Return）による自己文書化

**コード例15: ネストした条件分岐をガード節で平坦化**

```python
# BEFORE: 深いネスト + コメントで条件を説明
def calculate_discount(order: Order) -> Decimal:
    # 注文が空の場合は割引なし
    if order.items:
        # プレミアム会員の場合
        if order.customer.is_premium:
            # 合計が1万円以上の場合は15%割引
            if order.total >= 10000:
                return order.total * Decimal("0.15")
            # それ以外は10%割引
            else:
                return order.total * Decimal("0.10")
        # 一般会員の場合
        else:
            # 合計が1万円以上の場合は5%割引
            if order.total >= 10000:
                return order.total * Decimal("0.05")
    return Decimal("0")


# AFTER: ガード節 + 意味のある定数で自明にする
PREMIUM_HIGH_DISCOUNT_RATE = Decimal("0.15")
PREMIUM_BASE_DISCOUNT_RATE = Decimal("0.10")
STANDARD_DISCOUNT_RATE = Decimal("0.05")
DISCOUNT_THRESHOLD = Decimal("10000")

def calculate_discount(order: Order) -> Decimal:
    if not order.items:
        return Decimal("0")

    if not order.customer.is_premium:
        if order.total >= DISCOUNT_THRESHOLD:
            return order.total * STANDARD_DISCOUNT_RATE
        return Decimal("0")

    if order.total >= DISCOUNT_THRESHOLD:
        return order.total * PREMIUM_HIGH_DISCOUNT_RATE
    return order.total * PREMIUM_BASE_DISCOUNT_RATE
```

---

## 5. ドキュメンテーションコメント

ドキュメンテーションコメントは、公開 API の「契約書」である。利用者が実装を読まなくても正しく使えるようにするための情報を提供する。内部実装のコメントとは異なり、ドキュメンテーションコメントには What（何をするか）の説明が必須である。

### 5.1 構造

```
  ドキュメンテーションコメントの構成

  ┌─────────────────────────────────────────┐
  │ 1行目: 何をするかの要約（必須）          │
  │                                         │
  │ 詳細説明（必要な場合）                   │
  │                                         │
  │ Args/Parameters:（引数がある場合）       │
  │   パラメータの説明                       │
  │                                         │
  │ Returns:（戻り値がある場合）             │
  │   戻り値の説明                           │
  │                                         │
  │ Raises/Throws:（例外がある場合）         │
  │   発生する例外の説明                     │
  │                                         │
  │ Examples: (任意だが推奨)                 │
  │   使用例                                 │
  │                                         │
  │ Notes: (任意)                            │
  │   注意事項、制約                         │
  │                                         │
  │ See Also: (任意)                         │
  │   関連する関数・クラスへの参照           │
  └─────────────────────────────────────────┘
```

### 5.2 Python docstring（Google スタイル）

**コード例16: 完全なドキュメンテーションコメント**

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

    トランザクション分離レベルは SERIALIZABLE で実行される。
    ネットワーク障害時のリトライは呼び出し側で行うこと。

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
        ValueError: amount が 0 以下の場合。

    Example:
        >>> receipt = transfer_funds(account_a, account_b, Decimal('10000'))
        >>> print(receipt.transaction_id)
        'TXN-20240101-001'
        >>> print(receipt.timestamp)
        datetime(2024, 1, 1, 12, 0, 0)

    Note:
        同一口座間の振替は InsufficientBalanceError を送出する。
        これはビジネスルールによる制約である。

    See Also:
        - `Account.withdraw`: 口座からの引き出し
        - `Account.deposit`: 口座への入金
        - `TransferReceipt`: 受領証の詳細
    """
```

### 5.3 Java Javadoc

```java
/**
 * 商品の在庫数を更新する。
 *
 * <p>在庫数の更新は楽観的ロックを使用して排他制御を行う。
 * 同時更新が発生した場合は {@link OptimisticLockException} がスローされる。</p>
 *
 * <p>在庫数が0以下になる更新はビジネスルールにより拒否される。</p>
 *
 * @param productId 商品ID（null不可）
 * @param delta     在庫変動数（正: 入庫、負: 出庫）
 * @return 更新後の在庫数
 * @throws ProductNotFoundException 商品IDが存在しない場合
 * @throws InsufficientStockException 在庫が不足して出庫できない場合
 * @throws OptimisticLockException 同時更新が検出された場合
 * @since 2.0
 * @see Product#getStockCount()
 */
public int updateStock(String productId, int delta) {
    // ...
}
```

### 5.4 TypeScript TSDoc

```typescript
/**
 * ユーザーの認証トークンを検証する。
 *
 * @remarks
 * JWTの署名検証、有効期限チェック、リボケーションチェックを行う。
 * トークンが有効な場合は復号化されたペイロードを返す。
 *
 * @param token - JWT形式の認証トークン
 * @param options - 検証オプション
 * @returns 復号化されたトークンペイロード
 * @throws {@link TokenExpiredError} トークンの有効期限が切れている場合
 * @throws {@link InvalidSignatureError} 署名が不正な場合
 * @throws {@link RevokedTokenError} トークンが無効化されている場合
 *
 * @example
 * ```typescript
 * const payload = await verifyToken("eyJhbGci...", {
 *   audience: "my-app",
 *   issuer: "auth-server",
 * });
 * console.log(payload.userId); // "user-123"
 * ```
 */
async function verifyToken(
  token: string,
  options?: VerifyOptions
): Promise<TokenPayload> {
  // ...
}
```

### 5.5 各言語のドキュメンテーションコメント比較

| 言語 | 形式 | ツール | 特徴 |
|------|------|--------|------|
| Python | docstring (Google/NumPy/Sphinx) | Sphinx, pydoc | 3つの主要スタイルから選択 |
| Java | Javadoc (`/** ... */`) | javadoc | HTML タグを使用可能 |
| TypeScript | TSDoc / JSDoc | TypeDoc | `@remarks` で詳細説明 |
| Rust | `///` (doc comments) | rustdoc | Markdown + テスト埋め込み |
| Go | `//` (先頭行がパッケージ名) | godoc | シンプルなテキスト |
| C# | XML コメント (`///`) | Sandcastle, DocFX | 構造化された XML |
| Kotlin | KDoc (`/** ... */`) | Dokka | Javadoc の Kotlin 版 |
| Swift | `///` (Markup) | jazzy | Markdown ベース |

### 5.6 docstring のスタイル比較（Python）

```python
# ----- Google スタイル -----
def connect(host: str, port: int, timeout: float = 30.0) -> Connection:
    """サーバーに接続する。

    Args:
        host: サーバーのホスト名またはIPアドレス。
        port: ポート番号（1-65535）。
        timeout: 接続タイムアウト（秒）。

    Returns:
        確立された接続オブジェクト。

    Raises:
        ConnectionError: 接続に失敗した場合。
    """

# ----- NumPy スタイル -----
def connect(host: str, port: int, timeout: float = 30.0) -> Connection:
    """サーバーに接続する。

    Parameters
    ----------
    host : str
        サーバーのホスト名またはIPアドレス。
    port : int
        ポート番号（1-65535）。
    timeout : float, optional
        接続タイムアウト（秒）。デフォルトは30.0。

    Returns
    -------
    Connection
        確立された接続オブジェクト。

    Raises
    ------
    ConnectionError
        接続に失敗した場合。
    """

# ----- reStructuredText (Sphinx) スタイル -----
def connect(host: str, port: int, timeout: float = 30.0) -> Connection:
    """サーバーに接続する。

    :param host: サーバーのホスト名またはIPアドレス。
    :param port: ポート番号（1-65535）。
    :param timeout: 接続タイムアウト（秒）。
    :returns: 確立された接続オブジェクト。
    :raises ConnectionError: 接続に失敗した場合。
    """
```

| スタイル | 可読性 | ツール対応 | 推奨用途 |
|---------|--------|-----------|---------|
| Google | 高い | Sphinx (napoleon) | 一般的なプロジェクト |
| NumPy | 中程度 | Sphinx (napoleon) | 科学計算、データ分析 |
| Sphinx reST | 低め | Sphinx (ネイティブ) | Sphinx を深く使うプロジェクト |

---

## 6. コメントの言語選択とチームポリシー

### 6.1 コメント言語の選択基準

| 状況 | 推奨言語 | 理由 |
|------|---------|------|
| 日本語チーム・国内プロジェクト | 日本語 | 読み書きの効率が高い |
| グローバルチーム | 英語 | 全員が読める共通言語 |
| OSS プロジェクト | 英語 | 国際的な貢献者を想定 |
| 法的コメント | プロジェクト言語 | 法的要件に準拠 |
| 日本語チームだが将来 OSS 化の可能性 | 英語 | 切り替えコストを回避 |

**重要な原則: 同一プロジェクト内で言語を混在させない。** 混在が発生しやすいケースは、外部ライブラリのコードを社内にコピーした場合や、チームメンバーが入れ替わった場合である。

### 6.2 コメントポリシーのテンプレート

```markdown
# コメントポリシー (テンプレート)

## 基本方針
- コメントの言語: 日本語
- コードで意図を表現することを最優先する
- Why コメントを積極的に書く
- What コメントは原則不要

## 必須コメント
- 公開APIのドキュメンテーションコメント (Google スタイル)
- ライセンスヘッダー (SPDX 形式)
- 複雑なアルゴリズムの説明
- 非自明な設計判断の理由

## 禁止コメント
- コードの直訳コメント
- コメントアウトされたコード（Git を使うこと）
- 属人的な情報（個人名）
- ファイル内の変更履歴

## TODO コメント
- 形式: `# TODO(#<Issue番号>): <説明>`
- Issue トラッカーとの紐づけ必須
- 四半期ごとに棚卸しを実施
- CI で TODO 数を監視（閾値超過で警告）

## ドキュメンテーションコメント
- public メソッド/クラス: 必須
- protected メソッド: 推奨
- private メソッド: 名前で意図が不明な場合のみ
- スタイル: Google スタイル
```

### 6.3 自動チェックの導入

```yaml
# .github/workflows/comment-check.yml
name: Comment Quality Check

on: pull_request

jobs:
  check-comments:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check for commented-out code blocks
        run: |
          # 3行以上連続するコメントアウトコードを検出
          python -c "
          import re, sys, pathlib
          pattern = re.compile(r'(^\s*#\s*(if|for|def|class|return|import|from)\b.*\n){3,}', re.MULTILINE)
          found = False
          for f in pathlib.Path('src').rglob('*.py'):
              text = f.read_text()
              if pattern.search(text):
                  print(f'WARNING: Possible commented-out code in {f}')
                  found = True
          sys.exit(1 if found else 0)
          "

      - name: Check TODO format
        run: |
          # Issue番号なしの TODO を検出
          if grep -rn 'TODO[^(]' --include="*.py" --include="*.ts" src/; then
            echo "ERROR: TODO without issue number found"
            echo "Required format: TODO(#<issue_number>): <description>"
            exit 1
          fi

      - name: Count TODOs
        run: |
          count=$(grep -rn 'TODO' --include="*.py" src/ | wc -l)
          echo "Current TODO count: $count"
          if [ "$count" -gt 50 ]; then
            echo "WARNING: TODO count exceeds threshold (50)"
          fi
```

---

## 7. 高度なテクニック ── コメントのリファクタリング

### 7.1 コメントから関数への変換パターン

コメントでセクションを区切っているコードは、Extract Method の絶好の候補である。コメントの内容がそのまま関数名になる。

```python
# パターン1: セクションコメント → 関数抽出
# BEFORE
def process_data(raw_data):
    # データのクレンジング
    data = raw_data.strip()
    data = data.replace('\n', ' ')
    data = re.sub(r'\s+', ' ', data)

    # データの変換
    parts = data.split(',')
    result = [int(p) for p in parts if p.isdigit()]

    # データの検証
    if not result:
        raise ValueError("有効なデータがありません")
    if max(result) > 1000000:
        raise ValueError("値が範囲外です")

    return result

# AFTER: コメントが関数名になった
def process_data(raw_data: str) -> list[int]:
    cleaned = cleanse_whitespace(raw_data)
    integers = extract_integers(cleaned)
    validate_integer_range(integers)
    return integers

def cleanse_whitespace(raw_data: str) -> str:
    """空白文字を正規化する。"""
    data = raw_data.strip()
    data = data.replace('\n', ' ')
    return re.sub(r'\s+', ' ', data)

def extract_integers(data: str) -> list[int]:
    """カンマ区切り文字列から整数を抽出する。"""
    parts = data.split(',')
    return [int(p) for p in parts if p.isdigit()]

MAX_VALUE = 1_000_000

def validate_integer_range(data: list[int]) -> None:
    """整数リストが空でなく、値が範囲内であることを検証する。"""
    if not data:
        raise ValueError("有効なデータがありません")
    if max(data) > MAX_VALUE:
        raise ValueError(f"値が範囲外です (最大: {MAX_VALUE})")
```

### 7.2 コメント密度の計測と管理

```
コメント密度（Comment Density）の目安
────────────────────────────────────
  コメント密度 = コメント行数 / 総行数 * 100

  0-5%:   コメントが少なすぎる可能性
          → 公開APIにドキュメンテーションがあるか確認

  5-15%:  適切な範囲
          → Why コメントが中心なら良好

  15-25%: やや多い
          → What コメントが多くないか確認

  25%+:   過剰なコメント
          → コード自体の可読性改善が必要
────────────────────────────────────
注意: コメント密度は絶対指標ではない。
コードの性質（アルゴリズム、ビジネスルール）によって適切な値は変わる。
正規表現が多いファイルは密度が高くなるのが自然。
```

### 7.3 コメントの「腐敗」を防ぐ仕組み

コメントは時間の経過とともに「腐敗」する。コードは変更されるが、コメントは更新されず、やがてコードとの不一致（嘘のコメント）が発生する。これを防ぐ仕組みを以下に示す。

```
コメント腐敗の防止策
────────────────────────────────────

1. コードレビューでの確認
   - コード変更時に関連コメントも更新されているか確認
   - レビューチェックリストに「コメントの整合性」を含める

2. テスト可能なドキュメンテーション
   - Python: doctest で Example を実行テストに組み込む
   - Rust: doc test でコンパイル・実行テスト
   - TypeScript: tsdoc の @example を実際のテストコードとリンク

3. 自動チェック
   - TODO の Issue 番号が closed でないか CI で検証
   - ドキュメンテーションの引数名とコードの引数名が一致するか検証
   - 定期的な棚卸し（四半期ごとに TODO/FIXME を全件レビュー）

4. コメントを減らす設計
   - コメントが必要になる根本原因（複雑なコード）を解消
   - 型システム・命名・構造化で自己文書化を進める
────────────────────────────────────
```

---

## 8. アンチパターン

### アンチパターン1: コメントで設計の欠陥を隠す

```java
// NG: 複雑なロジックをコメントで説明
// ステータスが1（アクティブ）で、タイプが3（プレミアム）または
// タイプが4（エンタープライズ）で、最終ログインが30日以内の場合
if (user.status == 1 && (user.type == 3 || user.type == 4)
    && daysSince(user.lastLogin) <= 30) {
    // ...
}

// OK: コード自体が意図を語る
if (user.isActive() && user.isPremiumOrAbove() && user.isRecentlyActive()) {
    // ...
}
```

**なぜダメか:** コメントは「このコードは読みにくい」という告白。まずコード自体を改善すべき。コメントで説明が必要なコードは設計に問題がある兆候である。

**対策:** Extract Method、Rename、Introduce Constant 等のリファクタリングでコード自体の可読性を向上させる。それでも伝えきれない「なぜ」だけをコメントで補足する。

### アンチパターン2: 変更履歴をコメントで管理

```python
# NG: ファイル内の変更履歴
# 変更履歴:
# 2024-01-01 田中: 初版作成
# 2024-02-15 佐藤: バリデーション追加
# → バージョン管理システム（Git）が担うべき役割
```

**なぜダメか:** Git が担うべき役割をコメントで代替している。情報が二重管理になり、コメントの方が古くなるのは時間の問題。Git は誰が・いつ・何を・なぜ変更したかを正確に記録しており、それ以上の情報量をコメントで実現することは不可能である。

### アンチパターン3: API ドキュメントの省略

```python
# NG: 公開 API にドキュメントがない
def search(q, opts=None):
    pass

# OK: 公開 API にはドキュメンテーションコメント必須
def search(
    query: str,
    options: SearchOptions | None = None,
) -> SearchResult:
    """全文検索を実行する。

    Elasticsearch のインデックスに対してクエリを実行し、
    関連度順にソートされた結果を返す。

    Args:
        query: 検索クエリ文字列。Lucene クエリ構文をサポート。
        options: 検索オプション（ページネーション、フィルタ等）。

    Returns:
        SearchResult: ヒット件数と検索結果のリスト。

    Raises:
        InvalidQueryError: クエリ構文が不正な場合。
        SearchTimeoutError: 検索がタイムアウトした場合（デフォルト30秒）。
    """
```

**なぜダメか:** 公開 API は「契約書」である。利用者が実装を読まなくても正しく使えるべきであり、ドキュメンテーションコメントがないと、利用者は実装を読むかトライアンドエラーを強いられる。

### アンチパターン4: コメントの放置（腐敗）

```python
# NG: 古いコメントが放置されている
# 最大3回リトライする
MAX_RETRIES = 5  # 値は5に変更されたがコメントは未更新

# 月額料金を計算（消費税8%）
def calculate_monthly_fee(base_price: int) -> int:
    return int(base_price * 1.10)  # 税率は10%に変更済み

# OK: コメントが不要になるようコードを改善
MAX_RETRIES = 5  # コメント不要（定数名が自明）

TAX_RATE = Decimal("1.10")

def calculate_monthly_fee_with_tax(base_price: int) -> int:
    return int(base_price * TAX_RATE)
```

**なぜダメか:** 嘘のコメントは「コメントがない状態」より悪い。読み手を間違った方向に誘導し、バグの原因になる。

---

## 9. 実践演習

### 演習1（基礎）: 悪いコメントの除去

以下のコードから悪いコメントを除去し、必要なコメントだけを残してください。

```python
# ユーザーサービスクラス
class UserService:
    # コンストラクタ
    def __init__(self, repo, mailer):
        # リポジトリを設定
        self.repo = repo
        # メーラーを設定
        self.mailer = mailer

    # ユーザーを作成する
    def create_user(self, name, email):
        # 名前のバリデーション
        if not name:
            # 名前が空の場合はエラー
            raise ValueError("名前は必須です")
        # メールのバリデーション
        if not email:
            # メールが空の場合はエラー
            raise ValueError("メールは必須です")

        # ユーザーオブジェクトを作成
        user = User(name=name, email=email)
        # データベースに保存
        self.repo.save(user)
        # 確認メールを送信
        # ※ メール送信が失敗してもユーザー作成は成功扱い
        # ※ 非同期にすべきだが、現状のインフラでは同期のみ
        try:
            self.mailer.send_welcome(email)
        except MailError:
            pass  # TODO: ログに記録する
        # ユーザーを返す
        return user
```

**期待される出力:**

```python
class UserService:
    def __init__(self, repo: UserRepository, mailer: Mailer):
        self.repo = repo
        self.mailer = mailer

    def create_user(self, name: str, email: str) -> User:
        if not name:
            raise ValueError("名前は必須です")
        if not email:
            raise ValueError("メールは必須です")

        user = User(name=name, email=email)
        self.repo.save(user)

        # メール送信失敗はユーザー作成の成功に影響しない。
        # 理由: ウェルカムメールは再送可能であり、
        # ユーザー作成のトランザクションを巻き戻す必要がないため。
        # TODO(#567): 非同期メール送信への移行（インフラ対応待ち）
        try:
            self.mailer.send_welcome(email)
        except MailError:
            logger.warning(f"ウェルカムメール送信失敗: {email}")

        return user
```

**解説:** 除去したコメントは全て What コメント（コードの繰り返し）である。残したのは Why コメント（なぜメール失敗時に例外を握りつぶすのか）と、Issue紐づけの TODO のみ。

### 演習2（応用）: 自己文書化コードへのリファクタリング

以下のコメント依存のコードを、コメントなしで意図が伝わるコードにリファクタリングしてください。

```python
def calc(data):
    result = []
    for item in data:
        # 有効期限が過ぎていないかチェック
        if item['exp'] >= datetime.now():
            # ステータスがアクティブかチェック
            if item['st'] == 1:
                # 金額に税率を掛ける
                amount = item['amt'] * 1.10
                # 会員の場合は5%割引
                if item['mbr']:
                    amount = amount * 0.95
                # 結果に追加
                result.append({
                    'id': item['id'],
                    'total': amount,
                    'name': item['nm']
                })
    return result
```

**期待される出力:**

```python
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

TAX_RATE = Decimal("1.10")
MEMBER_DISCOUNT_RATE = Decimal("0.95")

@dataclass
class Item:
    id: str
    name: str
    amount: Decimal
    is_active: bool
    expiry_date: datetime
    is_member: bool

    def is_expired(self) -> bool:
        return self.expiry_date < datetime.now()

@dataclass
class PricedItem:
    id: str
    name: str
    total: Decimal

def calculate_priced_items(items: list[Item]) -> list[PricedItem]:
    return [
        price_item(item)
        for item in items
        if is_eligible(item)
    ]

def is_eligible(item: Item) -> bool:
    return not item.is_expired() and item.is_active

def price_item(item: Item) -> PricedItem:
    total = apply_tax(item.amount)
    if item.is_member:
        total = apply_member_discount(total)
    return PricedItem(id=item.id, name=item.name, total=total)

def apply_tax(amount: Decimal) -> Decimal:
    return amount * TAX_RATE

def apply_member_discount(amount: Decimal) -> Decimal:
    return amount * MEMBER_DISCOUNT_RATE
```

**解説:** 以下の変換を適用した。(1) 省略された変数名を意味のある名前に変更（Rename）。(2) マジックナンバーを定数化（Introduce Constant）。(3) 辞書をデータクラスに変換（Replace Type Code with Class）。(4) 条件判定を関数に抽出（Extract Method）。(5) ネストをリスト内包表記で平坦化。

### 演習3（発展）: ドキュメンテーションコメントの設計

以下のクラスに対して、完全なドキュメンテーションコメントを設計してください。Google スタイルの Python docstring を使用すること。

```python
class CacheManager:
    def __init__(self, max_size, ttl_seconds, eviction_policy):
        self._store = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._policy = eviction_policy

    def get(self, key):
        pass

    def set(self, key, value, ttl=None):
        pass

    def invalidate(self, key):
        pass

    def clear(self):
        pass

    def stats(self):
        pass
```

**期待される出力:**

```python
class CacheManager:
    """インメモリキャッシュマネージャ。

    TTL（有効期限）付きのキーバリューキャッシュを提供する。
    最大サイズに達した場合、指定された退避ポリシーに基づいて
    エントリを退避する。

    スレッドセーフではない。マルチスレッド環境では
    外部で排他制御を行うか、ThreadSafeCacheManager を使用すること。

    Attributes:
        max_size: キャッシュの最大エントリ数。
        ttl_seconds: デフォルトの TTL（秒）。0 は無期限。
        eviction_policy: 退避ポリシー（"lru", "lfu", "fifo"）。

    Example:
        >>> cache = CacheManager(max_size=1000, ttl_seconds=300,
        ...                      eviction_policy="lru")
        >>> cache.set("user:123", {"name": "Alice"})
        >>> user = cache.get("user:123")
        >>> print(user)
        {"name": "Alice"}
        >>> print(cache.stats())
        CacheStats(hits=1, misses=0, size=1, hit_rate=1.0)
    """

    def get(self, key: str) -> Any | None:
        """キーに対応する値を取得する。

        TTL が切れているエントリは None を返し、内部から削除する。

        Args:
            key: キャッシュキー。

        Returns:
            キャッシュされた値。キーが存在しないか TTL 切れの場合は None。
        """

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """キーと値のペアをキャッシュに格納する。

        キャッシュが max_size に達している場合、eviction_policy に
        基づいて既存エントリを1つ退避してから格納する。

        Args:
            key: キャッシュキー。
            value: 格納する値。シリアライズ可能であること。
            ttl: このエントリの TTL（秒）。None の場合はデフォルト TTL を使用。

        Raises:
            ValueError: ttl が負の数の場合。
        """

    def invalidate(self, key: str) -> bool:
        """指定されたキーのエントリを無効化（削除）する。

        Args:
            key: 無効化するキャッシュキー。

        Returns:
            キーが存在して削除された場合は True、存在しない場合は False。
        """

    def clear(self) -> None:
        """全てのキャッシュエントリを削除する。

        統計情報もリセットされる。
        """

    def stats(self) -> CacheStats:
        """キャッシュの統計情報を取得する。

        Returns:
            CacheStats: ヒット数、ミス数、現在のサイズ、ヒット率を含む統計。
        """
```

**解説:** ポイントは以下の通り。(1) クラスレベルの docstring で全体像・制約・使用例を説明。(2) 各メソッドの docstring で引数・戻り値・例外を明記。(3) スレッドセーフ性の警告を含める。(4) TTL 切れ時の動作など、戻り値だけでは分からない振る舞いを説明。

---

## 10. FAQ

### Q1: コメントは英語で書くべきか日本語で書くべきか？

チームの共通言語に合わせる。日本語チームなら日本語コメントで問題ない。ただし、OSSやグローバルチームでは英語が必須。**重要なのは一貫性**。同一プロジェクト内で言語を混在させない。

実務的には以下の判断基準を推奨する:
- **国内チームの社内プロジェクト**: 日本語
- **外部に公開する可能性のあるプロジェクト**: 英語
- **ドキュメンテーションコメント（public API）**: プロジェクト言語と同一
- **inline コメント**: チームの共通言語

### Q2: TODOコメントはどう管理すべきか？

TODOコメントは**Issueトラッカーと紐づけて**管理する。`TODO(#1234): 〜` の形式でIssue番号を含め、定期的にTODOを棚卸しする。放置されたTODOは技術的負債になるため、CI/CDでTODOの数を監視するのも効果的。

具体的な管理ルール:
1. **作成時**: Issue を起票し、Issue番号をコメントに含める
2. **週次**: `grep -rn "TODO" src/` で一覧を確認
3. **四半期**: 全 TODO を棚卸し、不要なものは削除
4. **CI**: Issue が closed の TODO を自動検出して警告

### Q3: APIのドキュメンテーションコメントはどこまで書くべきか？

**公開API（public）には必須**。以下を含める:
- 何をするか（1行要約）
- パラメータの意味と制約
- 戻り値の型と意味
- 発生する例外/エラー
- 使用例（複雑な場合）

privateメソッドは、名前から意図が明確なら省略可。ただし、以下の場合は private でもドキュメンテーションが推奨される:
- 複雑なアルゴリズム
- 非自明な副作用がある
- 他の開発者が修正する可能性が高い

### Q4: コメントアウトしたコードを「一時的に」残してよいか？

**原則 No**。Git に履歴があるので復元可能。ただし、以下の限定的なケースでは許容される:
- **デバッグ中のブランチ**: メインブランチへのマージ前に必ず削除
- **A/B テスト中**: テスト期間終了後に削除するチケットを必ず作成

いずれの場合も、マージ前にコメントアウトコードが残っていないことをレビューで確認すべきである。

### Q5: チームでコメントの書き方が統一されていない場合は？

以下の順序でルール整備を進める:
1. **コメントポリシー文書の作成**: 上記テンプレートを参考にチームで議論・合意
2. **リンターの導入**: pylint, ESLint 等でドキュメンテーションの有無を自動チェック
3. **コードレビュー基準に追加**: レビュー時にコメントの品質もチェック
4. **テンプレートの提供**: IDE のスニペットやテンプレートを用意

### Q6: 「良いコメント」と「悪いコメント」の判断に迷った場合は？

以下の3つの質問で判断する:
1. **「このコメントを削除したら情報が失われるか？」** → No なら不要なコメント
2. **「コードを改善すればこのコメントは不要になるか？」** → Yes ならまずコードを改善
3. **「このコメントは Why を説明しているか？」** → Yes なら価値あるコメント

---

## まとめ

### コメントの分類と判断表

| 種類 | 書くべきか | 例 | 判断基準 |
|------|-----------|-----|---------|
| Why（なぜ） | 書くべき | ビジネスルールの背景、技術的理由 | コードから読み取れない情報 |
| What（何を） | 原則不要 | コードの直訳 | コード自体が語るべき |
| How（どうやって） | アルゴリズムのみ | 複雑な正規表現、数学的処理 | 実装の意図が不明な場合 |
| 警告・注意 | 書くべき | スレッド安全性、パフォーマンス | 後続開発者への重要情報 |
| TODO/FIXME | Issueと紐づけて | 期限と担当を明記 | 管理された改善予定 |
| ライセンス | 必須 | 法的要件 | 法的に必要 |
| ドキュメンテーション | 公開APIに必須 | 引数、戻り値、例外 | 利用者への契約 |

### 自己文書化コードの変換テクニック表

| テクニック | コメント依存のコード | 自己文書化コード |
|-----------|-------------------|----------------|
| Extract Method | コメント付きブロック | 意味ある関数名に分割 |
| Rename | `x = x + 1 # 増加` | `retry_count += 1` |
| Introduce Constant | `if d <= 30: # 30日以内` | `if days <= ACTIVE_PERIOD` |
| Explaining Variable | `if a && b && c: # 条件説明` | `if is_eligible:` |
| 型システム活用 | `s: str # メールアドレス` | `email: Email` |
| ガード節 | 深いネスト + コメント | 早期リターンで平坦化 |

### コメント品質のチェックリスト

```
コメントレビュー時のチェックリスト
────────────────────────────────────
□ What コメント（コードの繰り返し）がないか
□ Why コメントが適切に書かれているか
□ コメントとコードが一致しているか（嘘コメントがないか）
□ コメントアウトされたコードがないか
□ TODO に Issue 番号が紐づいているか
□ 公開 API にドキュメンテーションコメントがあるか
□ 警告コメント（スレッド安全性等）が漏れていないか
□ マジックナンバーが定数化されているか
□ コメントの言語がプロジェクト内で統一されているか
□ 属人的な情報（個人名）が含まれていないか
────────────────────────────────────
```

---

## 次に読むべきガイド

- [命名規則](./00-naming.md) ── コメント不要のコードを書く第一歩。良い命名はコメントの必要性を大幅に減らす
- [関数設計](./01-functions.md) ── Extract Method による自己文書化コードの実現手法
- [クラス設計](./02-classes.md) ── クラスレベルでの自己文書化と責務の明確化
- [テスト原則](./04-testing-principles.md) ── テストコードにおけるコメントとドキュメンテーション
- [コードスメル](../02-refactoring/00-code-smells.md) ── セクション区切りコメントは God Class の兆候
- [リファクタリング技法](../02-refactoring/01-refactoring-techniques.md) ── コメント依存コードの改善手法
- [コードレビューチェックリスト](../03-practices-advanced/04-code-review-checklist.md) ── コメントのレビュー観点と品質基準

---

## 参考文献

1. **Robert C. Martin** 『Clean Code: A Handbook of Agile Software Craftsmanship』 Prentice Hall, 2008 (Chapter 4: Comments) ── コメントに関する原則の原典。「コメントは必ず嘘をつく」という挑発的なテーゼの背景を解説
2. **Dustin Boswell, Trevor Foucher** 『The Art of Readable Code』 O'Reilly Media, 2011 (Part II: Simplifying Loops and Logic) ── 可読性向上のための実践的テクニック。What/Why/How の使い分けが明快
3. **Kevlin Henney** "Comment Only What the Code Cannot Say" 『97 Things Every Programmer Should Know』 O'Reilly Media, 2010 ── コメントの本質を一文で表現した名エッセイ
4. **Steve McConnell** 『Code Complete』 Microsoft Press, 2004 (2nd Edition, Chapter 32: Self-Documenting Code) ── 自己文書化コードの体系的解説。コメントの密度と品質の関係を数値で示す
5. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 ── Extract Method 等のリファクタリングでコメントを不要にする手法
