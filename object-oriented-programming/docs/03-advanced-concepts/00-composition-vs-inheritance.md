# コンポジション vs 継承

> 「継承よりコンポジションを優先せよ」はGoF以来の鉄則。しかし盲目的にコンポジションを使うのではなく、両者のトレードオフを理解して使い分けることが重要。

## この章で学ぶこと

- [ ] コンポジションと継承の本質的な違いを理解する
- [ ] 「継承よりコンポジション」の理由を把握する
- [ ] 実践的な判断基準を学ぶ

---

## 1. 本質的な違い

```
継承（Inheritance）: is-a 関係
  → Dog is-a Animal（犬は動物である）
  → 親のすべてを引き継ぐ（強い結合）
  → コンパイル時に関係が固定

コンポジション（Composition）: has-a 関係
  → Car has-a Engine（車はエンジンを持つ）
  → 必要な部品を組み合わせる（弱い結合）
  → 実行時に部品を差し替え可能

  継承:
    ┌─────────┐
    │ Animal  │
    └────┬────┘
         │ is-a
    ┌────┴────┐
    │   Dog   │
    └─────────┘

  コンポジション:
    ┌─────────┐     ┌────────┐
    │   Car   │────→│ Engine │  has-a
    │         │────→│ Wheels │  has-a
    │         │────→│ GPS    │  has-a
    └─────────┘     └────────┘
```

---

## 2. なぜ「継承よりコンポジション」か

```
継承の問題:
  1. 強い結合: 親の変更が全子クラスに波及
  2. カプセル化の破壊: 子が親の実装詳細に依存
  3. 柔軟性の欠如: 実行時に振る舞いを変更できない
  4. 爆発的な組み合わせ:

  例: ゲームキャラクター
  継承で設計すると:
    Character
    ├── Warrior
    │   ├── FireWarrior
    │   ├── IceWarrior
    │   └── FlyingWarrior
    ├── Mage
    │   ├── FireMage
    │   ├── IceMage
    │   └── FlyingMage
    └── Archer
        ├── FireArcher
        ├── IceArcher
        └── FlyingArcher
    → 3属性 × 3職業 = 9クラス
    → 新属性追加で +3クラス、新職業追加で +3クラス

  コンポジションで設計すると:
    Character
    ├── has-a: AttackStyle（Warrior, Mage, Archer）
    ├── has-a: Element（Fire, Ice, Lightning）
    └── has-a: Movement（Walk, Fly, Teleport）
    → 3 + 3 + 3 = 9コンポーネント
    → 新属性追加で +1コンポーネント
```

### コンポジションによるリファクタリング

```typescript
// ❌ 継承: クラスの爆発
class Animal {
  eat(): void { console.log("食べる"); }
}
class FlyingAnimal extends Animal {
  fly(): void { console.log("飛ぶ"); }
}
class SwimmingAnimal extends Animal {
  swim(): void { console.log("泳ぐ"); }
}
class FlyingSwimmingAnimal extends ??? {
  // 多重継承できない！
}

// ✅ コンポジション: 柔軟な組み合わせ
interface MovementAbility {
  move(): string;
}

class Flying implements MovementAbility {
  move(): string { return "空を飛ぶ"; }
}

class Swimming implements MovementAbility {
  move(): string { return "水中を泳ぐ"; }
}

class Walking implements MovementAbility {
  move(): string { return "地上を歩く"; }
}

class Animal {
  private abilities: MovementAbility[] = [];

  addAbility(ability: MovementAbility): void {
    this.abilities.push(ability);
  }

  moveAll(): string[] {
    return this.abilities.map(a => a.move());
  }
}

// カモ: 飛べる + 泳げる + 歩ける
const duck = new Animal();
duck.addAbility(new Flying());
duck.addAbility(new Swimming());
duck.addAbility(new Walking());
console.log(duck.moveAll()); // ["空を飛ぶ", "水中を泳ぐ", "地上を歩く"]
```

---

## 3. Strategy パターンとの関係

```typescript
// コンポジション + Strategy = 実行時に振る舞いを変更

interface SortStrategy {
  sort<T>(data: T[], compareFn: (a: T, b: T) => number): T[];
}

class QuickSort implements SortStrategy {
  sort<T>(data: T[], compareFn: (a: T, b: T) => number): T[] {
    // クイックソートの実装
    return [...data].sort(compareFn);
  }
}

class MergeSort implements SortStrategy {
  sort<T>(data: T[], compareFn: (a: T, b: T) => number): T[] {
    // マージソートの実装
    return [...data].sort(compareFn);
  }
}

class DataProcessor {
  // コンポジション: 戦略を外部から注入
  constructor(private sortStrategy: SortStrategy) {}

  // 実行時に戦略を変更可能
  setSortStrategy(strategy: SortStrategy): void {
    this.sortStrategy = strategy;
  }

  process(data: number[]): number[] {
    return this.sortStrategy.sort(data, (a, b) => a - b);
  }
}

const processor = new DataProcessor(new QuickSort());
processor.process([3, 1, 4, 1, 5]);
// データ量が増えたら戦略を変更
processor.setSortStrategy(new MergeSort());
```

---

## 4. 判断基準

```
継承を使うべき場面:
  ✓ 明確な is-a 関係（ListはCollectionである）
  ✓ フレームワークの拡張ポイント（AbstractController）
  ✓ テンプレートメソッドパターン
  ✓ 子クラスが親の全メソッドを意味的に満たす

コンポジションを使うべき場面:
  ✓ has-a 関係（CarはEngineを持つ）
  ✓ 振る舞いの組み合わせが必要
  ✓ 実行時に振る舞いを変更したい
  ✓ 複数の「機能」を組み合わせたい
  ✓ テスト時にモックに差し替えたい

迷ったとき:
  → コンポジションを選ぶ（より安全）
  → 「このクラスは本当に親の "一種" か？」を自問
  → 「コード再利用のためだけの継承」は避ける
```

---

## まとめ

| 観点 | 継承 | コンポジション |
|------|------|---------------|
| 関係 | is-a | has-a |
| 結合度 | 強い | 弱い |
| 柔軟性 | 低い | 高い |
| 実行時変更 | 不可 | 可能 |
| 推奨度 | 限定的 | 優先 |

---

## 次に読むべきガイド
→ [[01-interfaces-and-traits.md]] — インターフェースとトレイト

---

## 参考文献
1. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994. (Favor composition over inheritance)
2. Bloch, J. "Effective Java." Item 18, 2018.
