# バージョン管理

> Git を使いこなすことは現代のソフトウェアエンジニアにとって「読み書き」と同じレベルの基本スキルである。

## この章で学ぶこと

- [ ] Gitの内部モデル（DAG）を理解する
- [ ] ブランチ戦略を説明できる
- [ ] 実務でのGitワークフローを知る

---

## 1. Gitの内部モデル

```
Git = 有向非巡回グラフ（DAG）+ コンテンツアドレスストレージ

  コミット: スナップショット（ファイル全体のツリー）
  ブランチ: コミットへのポインタ（ただの参照）
  HEAD: 現在のブランチ/コミットへのポインタ

  DAG構造:
  C1 ← C2 ← C3 ← C4      main
                 ↖
                  C5 ← C6  feature

  各コミットは親コミットのハッシュを保持
  → 改ざん不可能な履歴チェーン（ブロックチェーンと同原理）

  オブジェクトモデル:
  - blob: ファイルの内容（SHA-1ハッシュ）
  - tree: ディレクトリ（blobとtreeへの参照）
  - commit: ツリー + 親コミット + メタデータ
  - tag: コミットへの名前付き参照
```

---

## 2. ブランチ戦略

```
主要なブランチ戦略:

  1. GitHub Flow（シンプル）:
     main ──●──●──●──●──●──●──
                ↖     ↗
     feature    ●──●──●

     → mainは常にデプロイ可能
     → featureブランチからPRを作成
     → マージ後にデプロイ

  2. Git Flow（厳格）:
     main    ──●────────────●──
     develop ──●──●──●──●──●──
                ↖  ↗     ↖  ↗
     feature    ●──●      ●──●
     release              ●──●──●

     → 開発用(develop), 安定版(main), リリース用(release)
     → 大規模プロジェクト向け

  3. Trunk-Based Development（モダン）:
     main ──●──●──●──●──●──●──●──
               ↖↗  ↖↗
     short     ●    ●    ← 短命ブランチ（1日以内）

     → mainに頻繁にマージ（1日数回）
     → フィーチャーフラグで未完成機能を隠蔽
     → Google, Facebook が採用
```

---

## 3. 実務のGitコマンド

```bash
# 日常的に使うコマンド
git status              # 状態確認
git add -p              # 対話的にステージング
git commit -m "msg"     # コミット
git push origin branch  # プッシュ
git pull --rebase       # プル（リベースマージ）

# ブランチ操作
git switch -c feature   # ブランチ作成+切替
git merge main          # mainをマージ
git rebase main         # mainにリベース

# 履歴の確認
git log --oneline --graph  # グラフ表示
git blame file.py          # 各行の最終変更者

# トラブルシューティング
git stash               # 作業を一時退避
git bisect              # バグ導入コミットを二分探索
git reflog              # HEAD の移動履歴（復旧に使う）
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Gitの内部 | DAG + コンテンツアドレス。改ざん不可能 |
| ブランチ | ただのポインタ。軽量に作成・削除 |
| 戦略 | GitHub Flow(シンプル), Trunk-Based(モダン) |
| 必須スキル | rebase, stash, bisect, reflog |

---

## 次に読むべきガイド
→ [[../08-advanced-topics/00-distributed-systems.md]] — 分散システム

---

## 参考文献
1. Chacon, S. & Straub, B. "Pro Git." 2nd Edition, Apress, 2014.
2. Driessen, V. "A Successful Git Branching Model." 2010.
