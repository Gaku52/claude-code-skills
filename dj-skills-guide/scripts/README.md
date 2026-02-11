# DJ Skills Scripts

DJ練習・本番で使用するスクリプト集

## スクリプト一覧

### optimize_artwork.py

**用途:** Beatportで購入した曲のアートワークを最適化

**機能:**
- アートワークを300x300px、JPEG、RGBに変換
- ファイルサイズを500KB以下に圧縮
- Pioneer DJ CDJ対応100%保証
- 複数アートワーク削除
- ファイル名・パス検証
- ID3v2.3互換（古いCDJ対応）
- エラー詳細レポート

**使い方:**

```bash
# 基本使用
python3 ~/.claude/skills/dj-skills-guide/scripts/optimize_artwork.py <フォルダパス>

# 例: Beatportダウンロードフォルダを最適化
python3 ~/.claude/skills/dj-skills-guide/scripts/optimize_artwork.py ~/Downloads/Beatport

# 例: デスクトップの曲を最適化
python3 ~/.claude/skills/dj-skills-guide/scripts/optimize_artwork.py ~/Desktop
```

**出力:**
- 入力フォルダ内に「超安全版_300px」フォルダが作成される
- 最適化された曲がそこに保存される
- 元のファイルは変更されない

**対応フォーマット:**
- MP3
- AIFF
- WAV

**エラー回避:**
- アートワークサイズ超過
- 複数アートワーク埋め込み
- 非対応フォーマット（PNG等）
- ファイル名問題
- ID3バージョン非互換

**推奨ワークフロー:**

1. Beatportで曲購入（MP3 320kbps）
2. ダウンロード
3. このスクリプトで最適化
4. Rekordboxにインポート
5. Hot Cue設定
6. USBにエクスポート
7. クラブで使用

**注意:**
- 必ず購入した曲のみ使用
- SoundCloud Freeも使用可
- もらったデータは使用しない（著作権）
