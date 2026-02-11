# ピッチコントロール

Tempo Faderとキーコントロールの使い方を学びます。

## この章で学ぶこと

- ピッチコントロールとは
- Tempo Faderの使い方
- Master Tempo機能
- キーチェンジの影響
- DDJ-FLX4での操作
- ハーモニックミキシング基礎

## ピッチコントロールとは

**定義:**
```
ピッチコントロール = BPMと音程（キー）の調整

Tempo Fader:
- BPM を変更
- キーも変わる（通常）

Master Tempo:
- BPMを変更
- キーは変わらない
```

---

## 1. Tempo Fader

### DDJ-FLX4のTempo Fader

```
位置:
上: マイナス（遅く）
中央: ±0%（原曲BPM）
下: プラス（速く）

範囲: ±8%

例:
128 BPM × 1.08 = 138.24 BPM
128 BPM × 0.92 = 117.76 BPM
```

### 基本操作

```
ビートマッチング時:
1. 目標BPMを確認
2. Tempo Faderで調整
3. 微調整（0.1%単位）
4. 完璧に合わせる
```

---

## 2. Master Tempo

### 機能

```
Master Tempo ON:
BPM変更 → キー変わらない

Master Tempo OFF:
BPM変更 → キーも変わる
```

### DDJ-FLX4での操作

```
Master Tempo ボタン:
各デッキに1つずつ

ON: ボタンが光る
OFF: 消灯
```

### 使い分け

**OFF（通常）:**
```
メリット:
✓ 自然な音
✓ CPU負荷小
✓ 音質劣化なし

デメリット:
✗ キーが変わる
✗ ハーモニー崩れる可能性
```

**ON:**
```
メリット:
✓ キー維持
✓ ハーモニー保持
✓ 大きなBPM変更も可能

デメリット:
✗ 音質わずかに劣化
✗ CPU負荷大
```

---

## 3. キーチェンジの影響

### BPM変更とキーの関係

```
BPM上げる（+8%）:
→ キーが半音上がる

BPM下げる（-8%）:
→ キーが半音下がる

例:
128 BPM, Key: Am
→ 138 BPM（+8%）: A#m
→ 118 BPM（-8%）: G#m
```

### 聞こえ方の違い

**キー変化（Master Tempo OFF）:**
```
+8%: 高く、明るく聞こえる
-8%: 低く、暗く聞こえる

許容範囲: ±3%程度
→ それ以上は違和感
```

**キー固定（Master Tempo ON）:**
```
+8%でも: 原曲のキーのまま
-8%でも: 原曲のキーのまま

音質: わずかに劣化（通常気づかない）
```

---

## 4. 実践的な使い方

### ビートマッチング時

**Step 1: Tempo Fader調整**
```
1. Master Tempo OFF
2. Tempo Faderで BPM 合わせる
3. ±3%以内なら問題なし
```

**Step 2: キー確認**
```
Rekordbox でキー表示:
曲A: 8A（Am）
曲B: 8A（Am）
→ 同じキー、完璧

曲A: 8A（Am）
曲B: 7A（Gm）
→ 隣接キー、OK

曲A: 8A（Am）
曲B: 1A（Abm）
→ 離れすぎ、注意
```

---

## 5. ハーモニックミキシング基礎

### Camelot Wheel（復習）

```
同じ番号: 完全互換（例: 8A + 8A）
±1番号: 良い（例: 8A + 7A, 8A + 9A）
A ↔ B: 良い（例: 8A + 8B）

それ以外: 避ける
```

### Master Tempo活用

**大きなBPM差の場合:**
```
曲A: 125 BPM, 8A
曲B: 132 BPM, 8A

通常（Master Tempo OFF）:
曲Bを125に → キーが下がる → 8Aでなくなる

Master Tempo ON:
曲Bを125に → キー維持 → 8Aのまま
→ 完璧なハーモニー
```

---

## 6. DDJ-FLX4での練習

### 練習1: Master Tempo OFF

```
目標: 自然な音を学ぶ

1. 同じBPMの曲2曲
2. Master Tempo OFF
3. ミックス
4. キーの変化を聞く
```

### 練習2: Master Tempo ON

```
目標: キー固定を学ぶ

1. 同じキー、違うBPM の曲2曲
2. Master Tempo ON
3. BPM合わせる
4. キーが変わらないことを確認
```

### 練習3: 比較

```
同じミックスを2回:
1回目: Master Tempo OFF
2回目: Master Tempo ON

違いを聞き比べる
```

---

## まとめ

- **Tempo Fader**: BPM調整、±8%
- **Master Tempo OFF**: 自然、キー変化あり
- **Master Tempo ON**: キー固定、音質わずかに劣化
- **使い分け**: ±3%以内ならOFF、それ以上はON
- **ハーモニック**: Camelot Wheel参照
- **DDJ-FLX4**: Master Tempoボタンで切替

**次のステップ:** [キューイング](./cueing.md)

---

## 参考リンク

- [ビートマッチング](./beatmatching.md)
- [キーとスケール](../../00-fundamentals/key-scales.md)
- [BPMとテンポ](../01-basics/bpm-tempo.md)
