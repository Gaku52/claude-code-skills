# 抽象化

> 抽象化は「複雑さを隠し、本質的な特徴だけを公開する」原則。インターフェース設計、抽象クラスの使い方、そして「リーキー抽象化」の回避がポイント。

## この章で学ぶこと

- [ ] 抽象化のレベルと適用方法を理解する
- [ ] インターフェースと抽象クラスの使い分けを把握する
- [ ] リーキー抽象化の問題とその回避を学ぶ

---

## 1. 抽象化のレベル

```
抽象化 = 「不要な詳細を隠し、重要な情報だけを公開する」

レベル1: データ抽象化
  → 内部表現を隠す（カプセル化と重なる）
  → Date クラス: 内部がタイムスタンプか年月日構造体かを隠す

レベル2: 手続き抽象化
  → 処理の詳細を関数/メソッドに閉じ込める
  → array.sort(): ソートアルゴリズムの詳細を隠す

レベル3: 型抽象化（インターフェース）
  → 「何ができるか」だけを定義し、「どうやるか」は隠す
  → Iterable: 反復可能であることだけを約束

レベル4: モジュール抽象化
  → パッケージ/モジュールの公開APIのみを見せる
  → 内部のクラス群の複雑さを隠す

  ┌──────────── 利用者が見る世界 ────────────┐
  │  database.query("SELECT * FROM users")    │
  └────────────────────────────────────────────┘
                     ↓ 隠蔽
  ┌──────────── 内部の複雑さ ──────────────────┐
  │ コネクションプール管理                      │
  │ SQL パース → クエリプラン最適化             │
  │ インデックス検索 → ページ読み込み           │
  │ ロック管理 → トランザクション制御           │
  │ 結果セットのシリアライズ                    │
  └────────────────────────────────────────────┘
```

---

## 2. インターフェース vs 抽象クラス

```
┌──────────────┬─────────────────┬─────────────────┐
│              │ インターフェース │ 抽象クラス       │
├──────────────┼─────────────────┼─────────────────┤
│ 実装         │ なし（契約のみ）│ 部分的に可能     │
├──────────────┼─────────────────┼─────────────────┤
│ フィールド   │ なし            │ あり             │
├──────────────┼─────────────────┼─────────────────┤
│ 多重         │ 複数実装可能    │ 単一継承のみ     │
├──────────────┼─────────────────┼─────────────────┤
│ 関係         │ can-do          │ is-a             │
├──────────────┼─────────────────┼─────────────────┤
│ 用途         │ 能力の定義      │ 共通実装の提供   │
├──────────────┼─────────────────┼─────────────────┤
│ 例           │ Serializable    │ AbstractList     │
│              │ Comparable      │ HttpServlet      │
└──────────────┴─────────────────┴─────────────────┘

選択基準:
  「何ができるか」を定義 → インターフェース
  「どう動くか」の共通部分を提供 → 抽象クラス
  迷ったら → インターフェース（より柔軟）
```

```typescript
// TypeScript: インターフェースの実践

// 能力を表すインターフェース
interface Printable {
  print(): string;
}

interface Serializable {
  serialize(): string;
  deserialize(data: string): void;
}

interface Loggable {
  toLogString(): string;
}

// 複数のインターフェースを実装
class Invoice implements Printable, Serializable, Loggable {
  constructor(
    private id: string,
    private items: { name: string; price: number }[],
    private date: Date,
  ) {}

  print(): string {
    const total = this.items.reduce((sum, item) => sum + item.price, 0);
    return `請求書 #${this.id}\n合計: ¥${total}`;
  }

  serialize(): string {
    return JSON.stringify({ id: this.id, items: this.items, date: this.date });
  }

  deserialize(data: string): void {
    const parsed = JSON.parse(data);
    Object.assign(this, parsed);
  }

  toLogString(): string {
    return `[Invoice:${this.id}] items=${this.items.length}`;
  }
}
```

```python
# Python: 抽象クラス（ABC）
from abc import ABC, abstractmethod

class DataStore(ABC):
    """データストアの抽象基底クラス"""

    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        self._connected = False

    # 共通実装
    def ensure_connected(self):
        if not self._connected:
            self.connect()
            self._connected = True

    # テンプレートメソッド（共通フロー）
    def save(self, key: str, value: any) -> None:
        self.ensure_connected()
        self._validate(key, value)
        self._do_save(key, value)

    def _validate(self, key: str, value: any) -> None:
        if not key:
            raise ValueError("Key cannot be empty")

    # サブクラスが実装すべき抽象メソッド
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def _do_save(self, key: str, value: any) -> None: ...

    @abstractmethod
    def load(self, key: str) -> any: ...

class RedisStore(DataStore):
    def connect(self) -> None:
        # Redis固有の接続処理
        pass

    def _do_save(self, key: str, value: any) -> None:
        # Redis固有の保存処理
        pass

    def load(self, key: str) -> any:
        # Redis固有の読み込み処理
        pass
```

---

## 3. リーキー抽象化

```
Joel Spolsky の「リーキー抽象化の法則」(2002):
  「すべての重要な抽象化は、ある程度漏れている」

例:
  TCP/IP: 「信頼性のある通信」を抽象化
    → でもネットワーク遅延、パケットロスは隠しきれない
    → タイムアウト設定が必要 = 抽象化が漏れている

  ORM（Object-Relational Mapping）:
    → DBをオブジェクトとして抽象化
    → でも N+1 問題、JOIN の最適化は隠しきれない
    → SQL の知識が結局必要 = 抽象化が漏れている

  ファイルシステム:
    → 「ファイルは連続したバイト列」と抽象化
    → でもシーク時間、フラグメンテーションは存在する

対策:
  1. 抽象化の下のレイヤーも理解しておく
  2. 抽象化が漏れるケースをドキュメント化
  3. エスケープハッチ（生のアクセス手段）を提供
```

```typescript
// ORM のリーキー抽象化の例
class UserRepository {
  // 抽象化: オブジェクトとして操作
  async findUsersWithPosts(): Promise<User[]> {
    // ❌ N+1問題（抽象化が漏れる）
    const users = await User.findAll();
    for (const user of users) {
      user.posts = await Post.findByUserId(user.id); // N回のクエリ
    }
    return users;
  }

  // ✅ SQLの知識を使って最適化（抽象化の漏れに対処）
  async findUsersWithPostsOptimized(): Promise<User[]> {
    return await User.findAll({
      include: [{ model: Post }], // Eager loading（JOINに変換）
    });
  }
}
```

---

## 4. 良い抽象化の設計原則

```
1. 適切な粒度
   → 細かすぎ: 使いにくい（メソッドが多すぎる）
   → 粗すぎ: 柔軟性がない（何もカスタマイズできない）

2. 一貫性
   → 同じレベルの抽象度で統一
   → save() と write_bytes_to_disk() が混在しない

3. 最小驚き原則
   → 名前から想像できる動作をする
   → sort() が元の配列を破壊するのは驚き（Rubyの sort vs sort!）

4. 情報隠蔽
   → 知る必要のないことは隠す
   → ただしエスケープハッチは用意する
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 抽象化 | 複雑さを隠し本質のみ公開 |
| インターフェース | 能力（can-do）を定義。複数実装可 |
| 抽象クラス | 共通実装を提供。is-a 関係 |
| リーキー抽象化 | 全ての抽象化は漏れる。下層の理解も必要 |
| 設計原則 | 適切な粒度、一貫性、最小驚き |

---

## 次に読むべきガイド
→ [[../02-design-principles/00-solid-overview.md]] — SOLID原則

---

## 参考文献
1. Spolsky, J. "The Law of Leaky Abstractions." 2002.
2. Liskov, B. "Data Abstraction and Hierarchy." 1988.
