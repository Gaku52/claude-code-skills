# 状態管理概論

> 状態管理はWebアプリの複雑さの根源。ローカル状態、グローバル状態、サーバー状態、URL状態の分類を理解し、各カテゴリに最適なツールを選択することで、シンプルで保守しやすい状態管理を実現する。

## この章で学ぶこと

- [ ] 状態の4つのカテゴリを理解する
- [ ] 各カテゴリに適したツールの選定基準を把握する
- [ ] 状態管理の設計原則を学ぶ

---

## 1. 状態の分類

```
4つの状態カテゴリ:

  ① ローカル状態（UI State）:
     → コンポーネント固有の一時的な状態
     → モーダルの開閉、フォーム入力値、ホバー状態
     → ツール: useState, useReducer

  ② グローバル状態（Client State）:
     → 複数コンポーネントで共有する状態
     → テーマ、言語設定、ユーザー認証状態
     → ツール: Zustand, Jotai, Context

  ③ サーバー状態（Server State）:
     → APIから取得したデータ
     → ユーザー一覧、商品データ、注文履歴
     → ツール: TanStack Query, SWR

  ④ URL状態（URL State）:
     → URLに反映される状態
     → 検索クエリ、フィルタ、ページ番号、ソート
     → ツール: useSearchParams, nuqs

よくある間違い:
  ✗ サーバー状態を useState で管理
    → キャッシュ、リトライ、再検証が全て手動に
    → TanStack Query に任せるべき

  ✗ ローカル状態をグローバルに置く
    → 不要な再レンダリング
    → useState で十分

  ✗ URL状態を useState で管理
    → ブックマーク不可、共有不可
    → useSearchParams に

原則:
  「最も局所的な場所で、最も適切なツールで管理する」
```

---

## 2. ローカル状態

```typescript
// useState: 最もシンプル
function ToggleButton() {
  const [isOpen, setIsOpen] = useState(false);
  return <button onClick={() => setIsOpen(!isOpen)}>{isOpen ? 'Close' : 'Open'}</button>;
}

// useReducer: 複雑な状態遷移
type State = { count: number; step: number };
type Action =
  | { type: 'increment' }
  | { type: 'decrement' }
  | { type: 'setStep'; step: number }
  | { type: 'reset' };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'increment': return { ...state, count: state.count + state.step };
    case 'decrement': return { ...state, count: state.count - state.step };
    case 'setStep': return { ...state, step: action.step };
    case 'reset': return { count: 0, step: 1 };
  }
}

function Counter() {
  const [state, dispatch] = useReducer(reducer, { count: 0, step: 1 });
  return (
    <div>
      <span>{state.count}</span>
      <button onClick={() => dispatch({ type: 'increment' })}>+{state.step}</button>
      <button onClick={() => dispatch({ type: 'decrement' })}>-{state.step}</button>
    </div>
  );
}

// useReducer を使うべき場面:
// → 3つ以上の関連する状態
// → 状態遷移のルールが複雑
// → 次の状態が前の状態に依存
```

---

## 3. グローバル状態の選定

```
ライブラリ比較:

  Zustand:
  → シンプル、ボイラープレート最小
  → ストア = 関数（Reduxより直感的）
  → React外からもアクセス可能
  → 推奨: 中規模以上のアプリ

  Jotai:
  → アトムベース（Recoilの後継的）
  → コンポーネント単位の細かい再レンダリング制御
  → 推奨: 複雑なUIの状態管理

  React Context:
  → React組み込み、追加依存なし
  → 頻繁に変化する値には不向き（再レンダリング問題）
  → 推奨: テーマ、認証情報等の低頻度更新

  Redux Toolkit:
  → 最も成熟したエコシステム
  → DevTools が優秀
  → ボイラープレートが多い
  → 推奨: 大規模エンタープライズ

選定フロー:
  テーマ/認証/言語（低頻度更新）→ Context
  中規模の共有状態 → Zustand
  アトム単位の細かい制御 → Jotai
  大規模 + 厳密なアーキテクチャ → Redux Toolkit
```

---

## 4. 設計原則

```
状態管理のベストプラクティス:

  ① 状態の最小化:
     → 計算できる値は状態にしない
     → fullName = firstName + lastName → useMemo

  ② Derived State:
     // NG: 同期が必要な冗長な状態
     const [items, setItems] = useState([]);
     const [filteredItems, setFilteredItems] = useState([]);

     // OK: 1つの状態から導出
     const [items, setItems] = useState([]);
     const [filter, setFilter] = useState('');
     const filteredItems = useMemo(
       () => items.filter(i => i.name.includes(filter)),
       [items, filter]
     );

  ③ Colocate State（状態を使う場所の近くに配置）:
     → グローバルに上げる前に、本当に必要か考える
     → props drilling は2-3階層まで許容
     → それ以上ならコンポジションで解決を試みる

  ④ Single Source of Truth:
     → 同じデータを複数の場所で管理しない
     → サーバーデータのキャッシュは TanStack Query に一元化

  ⑤ 不変性（Immutability）:
     → 状態を直接変更しない
     → 新しいオブジェクト/配列を返す
     → Immer で簡潔に
```

---

## まとめ

| カテゴリ | 例 | ツール |
|---------|-----|--------|
| ローカル | モーダル開閉、入力値 | useState, useReducer |
| グローバル | テーマ、認証 | Zustand, Context |
| サーバー | API データ | TanStack Query |
| URL | 検索、フィルタ | useSearchParams |

---

## 次に読むべきガイド
→ [[01-zustand-and-jotai.md]] — Zustand / Jotai

---

## 参考文献
1. Kent C. Dodds. "Application State Management with React." kentcdodds.com, 2020.
2. TkDodo. "Practical React Query." tkdodo.eu, 2024.
3. Zustand. "Documentation." github.com/pmndrs/zustand, 2024.
