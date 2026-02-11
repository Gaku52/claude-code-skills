# プログラミングパラダイム概論

> パラダイムとは「問題をどのように分解し、解決策をどのように構造化するか」の思想。

## この章で学ぶこと

- [ ] 主要なプログラミングパラダイムの特徴を理解する
- [ ] 各パラダイムの適切な使い所を判断できる
- [ ] マルチパラダイムの実践的な活用法を理解する

---

## 1. 手続き型プログラミング（Procedural）

```
思想: 「処理を手順として上から下へ順番に記述する」

特徴:
  - 命令の逐次実行
  - 変数への代入（状態の変更）
  - 制御構造（if, for, while）
  - 手続き（関数/サブルーチン）による分割
```

```c
// C: 純粋な手続き型
#include <stdio.h>

int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;  // 状態を変更
    }
    return result;
}

int main() {
    printf("5! = %d\n", factorial(5));
    return 0;
}
```

```
利点:
  ✓ 直感的で理解しやすい（人間の思考に近い）
  ✓ ハードウェアに近い（効率的）
  ✓ デバッグしやすい（ステップ実行）

欠点:
  ✗ 大規模になるとグローバル状態が複雑化
  ✗ コードの再利用が難しい
  ✗ データと処理が分離しがち

適用:
  システムプログラミング、スクリプト、組み込み
```

---

## 2. オブジェクト指向プログラミング（OOP）

```
思想: 「データと振る舞いをオブジェクトとして結合する」

4つの柱:
  1. カプセル化  — 内部状態を隠蔽し、インターフェースを公開
  2. 継承        — 既存クラスの機能を引き継ぐ
  3. 多態性      — 同じインターフェースで異なる動作
  4. 抽象化      — 本質的な特徴を抽出
```

```python
# Python: OOP の例
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius: float):
        self._radius = radius  # カプセル化

    def area(self) -> float:  # 多態性
        return 3.14159 * self._radius ** 2

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height

    def area(self) -> float:  # 多態性
        return self._width * self._height

# 多態性: 同じメソッド呼び出しで異なる計算
shapes: list[Shape] = [Circle(5), Rectangle(3, 4)]
for shape in shapes:
    print(f"Area: {shape.area()}")
```

```
利点:
  ✓ 大規模開発に適する（分業・責任分担）
  ✓ コードの再利用（継承・コンポジション）
  ✓ 現実世界のモデリングに自然

欠点:
  ✗ 過度な設計（クラス爆発、深い継承階層）
  ✗ 状態の管理が複雑になりうる
  ✗ 並行処理との相性が悪い（共有可変状態）

適用:
  GUI、ゲーム、エンタープライズ、フレームワーク設計
```

---

## 3. 関数型プログラミング（Functional）

```
思想: 「計算を数学的関数の適用として記述する」

原則:
  1. 純粋関数     — 同じ入力に対して常に同じ出力（副作用なし）
  2. 不変性       — データを変更しない（新しいデータを生成）
  3. 第一級関数   — 関数を値として渡す・返す
  4. 参照透過性   — 式を値に置き換えても意味が変わらない
```

```haskell
-- Haskell: 純粋関数型
-- 階乗（再帰）
factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- リスト処理（高階関数）
doublePositives :: [Int] -> [Int]
doublePositives = map (*2) . filter (>0)

-- 合成
process :: [Int] -> Int
process = sum . map (*2) . filter (>0)
-- process [1, -2, 3, -4, 5] → 18
```

```javascript
// JavaScript: 関数型スタイル
const numbers = [1, -2, 3, -4, 5];

// 命令型
let result = 0;
for (const n of numbers) {
    if (n > 0) result += n * 2;
}

// 関数型
const result2 = numbers
    .filter(n => n > 0)
    .map(n => n * 2)
    .reduce((sum, n) => sum + n, 0);
```

```
利点:
  ✓ テストが容易（純粋関数は入出力だけで検証可能）
  ✓ 並行処理に強い（不変データ = 競合なし）
  ✓ 合成可能性（小さな関数を組み合わせて大きな機能を構築）
  ✓ 推論しやすい（参照透過性）

欠点:
  ✗ 学習曲線が急（モナド、代数的データ型）
  ✗ パフォーマンスの予測が難しい（遅延評価）
  ✗ 入出力（副作用）の扱いが煩雑

適用:
  データ変換、並行処理、コンパイラ、金融システム
```

---

## 4. その他のパラダイム

### 論理型プログラミング

```prolog
% Prolog: 論理的な関係を宣言
parent(tom, bob).
parent(tom, liz).
parent(bob, ann).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

% クエリ: ?- grandparent(tom, ann).
% → true（tom → bob → ann）
```

### リアクティブプログラミング

```javascript
// RxJS: データストリームを宣言的に処理
import { fromEvent } from 'rxjs';
import { debounceTime, map, filter, switchMap } from 'rxjs/operators';

fromEvent(searchInput, 'input').pipe(
    debounceTime(300),
    map(event => event.target.value),
    filter(query => query.length >= 3),
    switchMap(query => fetch(`/api/search?q=${query}`))
).subscribe(results => renderResults(results));
```

### アクターモデル

```
Erlang / Elixir のアクターモデル:

  アクター = 独立したプロセス（軽量、数百万単位で生成可能）

  通信はメッセージパッシングのみ（共有メモリなし）

  ┌─────────┐  メッセージ  ┌─────────┐
  │ Actor A │────────────→│ Actor B │
  └─────────┘             └─────────┘
       ↑                       │
       └───────────────────────┘

  利点: 障害分離（1つのアクターの失敗が他に影響しない）
  適用: 電話交換機、チャットシステム、IoT
```

---

## 5. マルチパラダイム — 現代の主流

```
現代の主要言語はマルチパラダイム:

  Python:     手続き型 + OOP + 関数型
  JavaScript: 手続き型 + OOP(プロトタイプ) + 関数型
  Rust:       手続き型 + 関数型 + ジェネリクス（OOPなし）
  Kotlin:     OOP + 関数型
  Swift:      OOP + 関数型 + プロトコル指向
  Scala:      OOP + 関数型（最もバランスが良い）
  TypeScript: OOP + 関数型 + ジェネリクス
```

```typescript
// TypeScript: マルチパラダイムの例

// OOP 的なアプローチ
class UserService {
    constructor(private repo: UserRepository) {}

    async getActiveUsers(): Promise<User[]> {
        const users = await this.repo.findAll();
        return users.filter(u => u.isActive);
    }
}

// 関数型的なアプローチ
const getActiveUsers = (repo: UserRepository) =>
    repo.findAll().then(users => users.filter(u => u.isActive));

// 実務では両方のスタイルを適材適所で使う
// - ドメインモデル → OOP（状態と振る舞いの結合）
// - データ変換 → 関数型（パイプライン処理）
// - ユーティリティ → 手続き型（シンプルな処理）
```

---

## 6. パラダイム選択の指針

```
問題の種類                   適するパラダイム
──────────────────────────────────────────────
状態を持つエンティティ       → OOP
データの変換・集計           → 関数型
手順が明確な処理             → 手続き型
非同期イベント処理           → リアクティブ
ルールベースの推論           → 論理型
高並行システム               → アクターモデル

原則:
  「パラダイムは道具であり、信仰ではない」
  1つのプロジェクト内でも場面に応じて使い分ける
```

---

## まとめ

| パラダイム | 中心概念 | 得意領域 | 代表言語 |
|-----------|---------|---------|---------|
| 手続き型 | 命令の順次実行 | スクリプト・システム | C, Go |
| OOP | オブジェクト | 大規模開発 | Java, C# |
| 関数型 | 純粋関数・不変性 | データ変換・並行 | Haskell, Elixir |
| リアクティブ | データストリーム | UI・イベント処理 | RxJS |
| アクター | メッセージパッシング | 高並行システム | Erlang |

---

## 次に読むべきガイド
→ [[03-choosing-a-language.md]] — 言語の選び方

---

## 参考文献
1. Van Roy, P. & Haridi, S. "Concepts, Techniques, and Models of Computer Programming." MIT Press, 2004.
2. Armstrong, J. "Programming Erlang." 2nd Ed, Pragmatic Bookshelf, 2013.
