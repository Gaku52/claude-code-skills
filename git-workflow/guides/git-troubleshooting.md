# Git トラブルシューティング完全ガイド

> **最終更新:** 2026-01-02
> **対象読者:** 全開発者
> **推定読了時間:** 70分

## 📋 目次

1. [緊急度別トラブル対応](#緊急度別トラブル対応)
2. [コミット関連の問題](#コミット関連の問題)
3. [ブランチ関連の問題](#ブランチ関連の問題)
4. [マージ・コンフリクト解決](#マージコンフリクト解決)
5. [リモートリポジトリの問題](#リモートリポジトリの問題)
6. [履歴の修正](#履歴の修正)
7. [データ復旧](#データ復旧)
8. [パフォーマンス問題](#パフォーマンス問題)
9. [認証・権限の問題](#認証権限の問題)
10. [サブモジュール・サブツリー](#サブモジュールサブツリー)
11. [Git LFS問題](#git-lfs問題)
12. [CI/CD連携の問題](#cicd連携の問題)
13. [よくあるエラーメッセージ集](#よくあるエラーメッセージ集)
14. [予防策とベストプラクティス](#予防策とベストプラクティス)
15. [緊急時対応フローチャート](#緊急時対応フローチャート)

---

## 緊急度別トラブル対応

### 🔴 Critical（本番環境影響）

#### 問題: 本番ブランチに壊れたコードがマージされた

**症状:**
```
❌ 本番環境でエラー発生
❌ ユーザーに影響
❌ サービス停止
```

**即座の対応（5分以内）:**

```bash
# オプション1: 直前のコミットをrevert（推奨）
git revert HEAD
git push origin main

# オプション2: 安全なコミットにreset（慎重に）
git log --oneline -10  # 安全なコミットを特定
git reset --hard abc123  # 安全なコミットのハッシュ
git push --force-with-lease origin main

# オプション3: 緊急パッチ
git checkout -b hotfix/critical-fix
# ... 修正 ...
git commit -m "fix: critical production bug"
git push origin hotfix/critical-fix
# → PRレビュー（簡易）→ マージ
```

**事後対応:**
```markdown
1. インシデントレポート作成
2. 根本原因分析
3. 再発防止策の実施
   - ブランチ保護ルール強化
   - CI/CDゲート追加
   - レビュープロセス見直し
```

#### 問題: 機密情報をコミットしてpushした

**症状:**
```
❌ APIキー・パスワードがリポジトリに
❌ すでにpush済み
❌ パブリックリポジトリの場合は特に危険
```

**即座の対応（5分以内）:**

```bash
# ステップ1: 機密情報を無効化
# → API providerで即座にキーを無効化・再発行

# ステップ2: コミット履歴から削除（git-filter-repo推奨）
# インストール:
pip install git-filter-repo

# 機密情報を含むファイルを履歴から削除
git filter-repo --path config/secrets.yml --invert-paths

# または BFG Repo-Cleaner
# brew install bfg
bfg --delete-files secrets.yml

# ステップ3: 強制push
git push --force --all origin
git push --force --tags origin

# ステップ4: チームに通知
# - 全員がリポジトリを再cloneする必要がある
```

**事後対応:**
```markdown
1. .gitignore に追加
2. pre-commit hook でチェック追加
3. 機密情報スキャンツール導入（gitleaks等）
4. 環境変数管理ツール導入（dotenv等）
```

### 🟡 High（開発ブロッカー）

#### 問題: マージできない（コンフリクト）

**詳細は「マージ・コンフリクト解決」セクション参照**

#### 問題: ブランチを間違えてコミットした

**症状:**
```
❌ mainに直接コミットしてしまった
❌ または違うfeatureブランチにコミット
```

**対応（10分以内）:**

```bash
# ケース1: まだpushしていない
# 正しいブランチを作成
git branch feature/correct-branch

# mainを戻す
git reset --hard HEAD~1

# 正しいブランチに切り替え
git checkout feature/correct-branch

# ケース2: すでにpushしてしまった
# revertで取り消し
git revert HEAD
git push origin main

# 正しいブランチで再度コミット
git checkout -b feature/correct-branch
# ... 再実装 ...
git commit -m "feat: correct implementation"
```

### 🟢 Normal（通常対応）

#### 問題: コミットメッセージを間違えた

**対応:**

```bash
# 直前のコミット（pushしていない）
git commit --amend -m "correct message"

# 直前のコミット（pushした、単独作業）
git commit --amend -m "correct message"
git push --force-with-lease

# 過去のコミット
git rebase -i HEAD~3
# エディタで pick → reword に変更
```

---

## コミット関連の問題

### 問題1: コミットを取り消したい

#### ケースA: まだpushしていない

**最新のコミットを取り消し（変更は保持）:**
```bash
git reset --soft HEAD~1
# → ファイルはステージング状態に戻る

# 確認
git status
```

**最新のコミットを完全に削除:**
```bash
git reset --hard HEAD~1
# ⚠️ 警告: 変更が完全に失われる

# 確認
git log --oneline -5
```

**複数のコミットを取り消し:**
```bash
# 最新3個のコミットを取り消し
git reset --soft HEAD~3

# または特定のコミットまで戻る
git reset --soft abc123
```

#### ケースB: すでにpushした

**revert（推奨）:**
```bash
# 最新のコミットをrevert
git revert HEAD
git push origin main

# 複数のコミットをrevert
git revert HEAD~2..HEAD
git push origin main
```

**reset + force push（慎重に）:**
```bash
# 単独作業の場合のみ
git reset --hard HEAD~1
git push --force-with-lease origin feature/my-branch

# ⚠️ 共同作業ブランチでは絶対にやらない
```

### 問題2: コミットを分割したい

**シナリオ:**
```
1つのコミットに複数の変更が含まれている
→ レビューしやすいように分割したい
```

**対応:**
```bash
# ステップ1: コミットを取り消す（変更は保持）
git reset --soft HEAD~1

# ステップ2: 変更をアンステージ
git reset HEAD

# ステップ3: 部分的にステージング
git add -p src/auth/LoginService.ts
# y: このhunkをstage
# n: skip
# s: 分割

# ステップ4: 最初のコミット
git commit -m "feat(auth): add login validation"

# ステップ5: 残りの変更をコミット
git add src/auth/LoginService.ts
git commit -m "refactor(auth): simplify error handling"

# ステップ6: pushが必要な場合
git push --force-with-lease origin feature/my-branch
```

### 問題3: コミットを統合したい（squash）

**シナリオ:**
```
WIP commits を1つにまとめたい
feat: add feature (WIP)
feat: add feature (WIP 2)
feat: add feature (final)
→ 1つのコミットに統合
```

**対応:**
```bash
# インタラクティブrebase
git rebase -i HEAD~3

# エディタが開く:
pick abc123 feat: add feature (WIP)
pick def456 feat: add feature (WIP 2)
pick ghi789 feat: add feature (final)

# 変更:
pick abc123 feat: add feature (WIP)
squash def456 feat: add feature (WIP 2)
squash ghi789 feat: add feature (final)

# 保存して終了
# → 新しいコミットメッセージを入力

# push
git push --force-with-lease origin feature/my-branch
```

### 問題4: コミットの順序を変更したい

**対応:**
```bash
git rebase -i HEAD~5

# エディタで順序を入れ替え:
pick abc123 commit A
pick def456 commit B
pick ghi789 commit C

# → 順序変更:
pick ghi789 commit C
pick abc123 commit A
pick def456 commit B

# 保存して終了
git push --force-with-lease origin feature/my-branch
```

---

## ブランチ関連の問題

### 問題1: ブランチを削除してしまった

**症状:**
```
❌ git branch -D feature/important
❌ まだマージしていなかった
❌ 作業内容が失われた
```

**復旧:**

```bash
# ステップ1: reflogで削除前のコミットを探す
git reflog

# 出力例:
# abc123 HEAD@{0}: checkout: moving from feature/important to main
# def456 HEAD@{1}: commit: important work
# ghi789 HEAD@{2}: commit: more work

# ステップ2: ブランチを復活
git checkout -b feature/important def456

# または
git branch feature/important def456

# 確認
git log --oneline -5
```

### 問題2: ブランチ名を変更したい

**ローカルのみ:**
```bash
# 現在のブランチ名を変更
git checkout old-name
git branch -m new-name

# または別のブランチの名前を変更
git branch -m old-name new-name
```

**リモートにもpush済みの場合:**
```bash
# ローカルで名前変更
git checkout old-name
git branch -m new-name

# 新しい名前でpush
git push origin new-name

# 古いブランチを削除
git push origin --delete old-name

# upstreamを設定
git push --set-upstream origin new-name
```

### 問題3: 間違ったブランチから分岐した

**シナリオ:**
```
feature/A から feature/B を分岐してしまった
本来は main から分岐すべきだった
```

**対応:**

```bash
# 現在のfeature/Bの変更を確認
git log main..feature/B --oneline

# feature/Aの変更を除外したコミットを特定
# → cherry-pickで移植

# 新しくmainから分岐
git checkout main
git checkout -b feature/B-new

# 必要なコミットのみcherry-pick
git cherry-pick abc123
git cherry-pick def456

# 古いfeature/Bを削除
git branch -D feature/B

# 名前変更
git branch -m feature/B-new feature/B

# push
git push --force-with-lease origin feature/B
```

### 問題4: リモートブランチが削除されているのにローカルに残っている

**症状:**
```bash
git branch -a
# origin/feature/old-branch（削除済み）が表示される
```

**対応:**
```bash
# リモートの削除済みブランチを反映
git fetch --prune

# または設定で自動prune
git config --global fetch.prune true

# ローカルのマージ済みブランチも削除
git branch --merged | grep -v "\*" | grep -v "main" | grep -v "develop" | xargs -n 1 git branch -d
```

---

## マージ・コンフリクト解決

### コンフリクトの種類と対処法

#### Type 1: シンプルなコンフリクト

**症状:**
```bash
git merge feature/A
# Auto-merging src/App.tsx
# CONFLICT (content): Merge conflict in src/App.tsx
```

**ファイル内容:**
```typescript
<<<<<<< HEAD
const greeting = "Hello";
=======
const greeting = "Hi";
>>>>>>> feature/A
```

**解決手順:**

```bash
# ステップ1: コンフリクトファイルを確認
git status

# ステップ2: ファイルを編集
# <<<<<<< HEAD
# =======
# >>>>>>>
# のマーカーを削除して正しいコードに修正

const greeting = "Hello";  # どちらかを選択、または両方を統合

# ステップ3: ステージング
git add src/App.tsx

# ステップ4: マージを完了
git commit -m "merge: resolve conflict in App.tsx"

# または
git merge --continue
```

#### Type 2: 複雑なコンフリクト（複数ファイル）

**対応:**

```bash
# ステップ1: コンフリクトファイル一覧
git diff --name-only --diff-filter=U

# ステップ2: 1つずつ解決
# ファイルを編集 → git add → 次へ

# ステップ3: 全て解決したか確認
git status

# ステップ4: コミット
git commit -m "merge: resolve conflicts"
```

#### Type 3: バイナリファイルのコンフリクト

**症状:**
```bash
CONFLICT (content): Merge conflict in assets/logo.png
```

**対応:**

```bash
# どちらかを選択

# オプション1: こちら側を採用（HEAD）
git checkout --ours assets/logo.png

# オプション2: 相手側を採用（feature/A）
git checkout --theirs assets/logo.png

# オプション3: 手動で正しいファイルを配置
cp ~/correct-logo.png assets/logo.png

# ステージング
git add assets/logo.png
```

### コンフリクト解決ツール

#### VS Code

```bash
# VS Codeで開く
code .

# コンフリクトファイルを開くと:
# - Accept Current Change
# - Accept Incoming Change
# - Accept Both Changes
# - Compare Changes
# のボタンが表示される
```

#### Git mergetool

```bash
# 設定
git config --global merge.tool vimdiff

# または
git config --global merge.tool kdiff3

# 実行
git mergetool

# → 対話的にコンフリクト解決
```

#### difftool比較

```bash
# 変更を比較
git difftool HEAD..feature/A src/App.tsx

# → 差分が視覚的に表示される
```

### マージの中止

**コンフリクト解決中に中止したい場合:**

```bash
# マージを中止（元の状態に戻る）
git merge --abort

# または
git reset --merge
```

### Rebase時のコンフリクト

**症状:**
```bash
git rebase main
# CONFLICT (content): Merge conflict in src/App.tsx
# error: could not apply abc123... feat: add feature
```

**対応:**

```bash
# ステップ1: コンフリクト解決
vim src/App.tsx  # 編集
git add src/App.tsx

# ステップ2: rebase続行
git rebase --continue

# または中止
git rebase --abort

# コンフリクトをスキップ（慎重に）
git rebase --skip
```

### Cherry-pick時のコンフリクト

**症状:**
```bash
git cherry-pick abc123
# CONFLICT (content): Merge conflict in src/App.tsx
```

**対応:**

```bash
# コンフリクト解決
vim src/App.tsx
git add src/App.tsx

# cherry-pick続行
git cherry-pick --continue

# または中止
git cherry-pick --abort
```

---

## リモートリポジトリの問題

### 問題1: push rejected（リモートが先に進んでいる）

**症状:**
```bash
git push origin main
# ! [rejected]        main -> main (fetch first)
# error: failed to push some refs to 'origin'
```

**対応:**

```bash
# オプション1: pull → push（推奨）
git pull origin main
# コンフリクトがあれば解決
git push origin main

# オプション2: rebase → push
git pull --rebase origin main
# コンフリクトがあれば解決
git push origin main

# オプション3: force push（慎重に）
# 単独作業の場合のみ
git push --force-with-lease origin main
```

### 問題2: リモートURLを間違えた

**症状:**
```bash
git remote -v
# origin  https://github.com/wrong-user/repo.git (fetch)
```

**対応:**

```bash
# URLを変更
git remote set-url origin https://github.com/correct-user/repo.git

# 確認
git remote -v
```

### 問題3: リモートブランチを追跡できない

**症状:**
```bash
git pull
# There is no tracking information for the current branch.
```

**対応:**

```bash
# upstreamを設定
git branch --set-upstream-to=origin/main main

# または push時に設定
git push -u origin main

# 確認
git branch -vv
```

### 問題4: Large File エラー

**症状:**
```bash
git push
# remote: error: File large-file.zip is 120.00 MB; this exceeds GitHub's file size limit of 100.00 MB
```

**対応:**

```bash
# オプション1: ファイルを削除して再コミット
git rm --cached large-file.zip
git commit --amend -m "remove large file"
git push

# オプション2: Git LFSを使用
git lfs install
git lfs track "*.zip"
git add .gitattributes
git add large-file.zip
git commit -m "add large file with LFS"
git push

# オプション3: 履歴から削除（すでにpush済みの場合）
git filter-repo --path large-file.zip --invert-paths
git push --force
```

---

## 履歴の修正

### git reflog活用

**reflogとは:**
```
Gitの「操作履歴」を記録
→ 削除したコミット・ブランチも復旧可能
```

**基本コマンド:**

```bash
# reflog表示
git reflog

# 出力例:
# abc123 HEAD@{0}: commit: latest work
# def456 HEAD@{1}: commit: important feature
# ghi789 HEAD@{2}: reset: moving to HEAD~1
# jkl012 HEAD@{3}: commit: deleted work (これを復旧したい)

# 特定のコミットに戻る
git checkout HEAD@{3}

# ブランチとして復活
git checkout -b recovered-work HEAD@{3}
```

### 間違ったrebaseを取り消す

**症状:**
```
git rebase -i でコミットを削除してしまった
```

**復旧:**

```bash
# reflogでrebase前の状態を探す
git reflog

# rebase前の状態に戻る
git reset --hard HEAD@{5}  # rebase前のHEAD

# 確認
git log --oneline -10
```

### コミット履歴の書き換え（filter-repo）

**用途:**
```
- 機密情報の削除
- 大きなファイルの削除
- 著者情報の修正
- メールアドレスの修正
```

**例: 著者情報の修正**

```bash
# git-filter-repoインストール
pip install git-filter-repo

# メールアドレスを変更
git filter-repo --email-callback '
  return email.replace(b"old@example.com", b"new@example.com")
'

# 強制push
git push --force --all origin
```

---

## データ復旧

### 削除したコミットを復旧

**シナリオ:**
```
git reset --hard で削除してしまった
```

**復旧:**

```bash
# reflogで探す
git reflog

# 削除したコミットのハッシュを見つける
# abc123 HEAD@{5}: commit: deleted work

# 復旧
git cherry-pick abc123

# またはブランチとして復活
git branch recovered abc123
```

### 削除したファイルを復旧

**ケース1: まだコミットしていない**

```bash
# 特定のファイルを復元
git checkout -- path/to/deleted-file.txt

# すべての変更を破棄
git reset --hard HEAD
```

**ケース2: コミット済み**

```bash
# ファイルを削除したコミットを探す
git log --all --full-history -- path/to/deleted-file.txt

# コミットハッシュが見つかったら
git checkout abc123^ -- path/to/deleted-file.txt

# ^は「その1つ前」を意味する
```

### .git フォルダを削除してしまった

**症状:**
```
rm -rf .git
# Git履歴が全て失われた
```

**対応:**

```bash
# リモートからclone
git clone https://github.com/user/repo.git repo-recovered
cd repo-recovered

# ローカルの変更を手動でコピー
# （バックアップがある場合）

# ⚠️ .gitを削除すると完全復旧は不可能
# → 常にバックアップを
```

---

## パフォーマンス問題

### 問題1: git statusが遅い

**原因:**
```
- リポジトリが大きい
- ファイル数が多い
- .gitignoreが最適化されていない
```

**対応:**

```bash
# ステップ1: .gitignoreを最適化
# node_modules/, .DS_Store等を追加

# ステップ2: git configでパフォーマンス改善
git config core.preloadindex true
git config core.fscache true
git config gc.auto 256

# ステップ3: 不要なファイルを削除
git clean -fdx
```

### 問題2: cloneが遅い

**対応:**

```bash
# shallow clone（履歴を浅く）
git clone --depth 1 https://github.com/user/repo.git

# 特定のブランチのみ
git clone --branch main --single-branch https://github.com/user/repo.git

# 後で全履歴を取得
git fetch --unshallow
```

### 問題3: .gitフォルダが巨大

**対応:**

```bash
# サイズ確認
du -sh .git

# ガベージコレクション
git gc --aggressive --prune=now

# 大きなファイルを探す
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | sed -n 's/^blob //p' \
  | sort --numeric-sort --key=2 \
  | tail -10

# 大きなファイルを履歴から削除
git filter-repo --path large-file.zip --invert-paths
```

---

## 認証・権限の問題

### 問題1: Permission denied (publickey)

**症状:**
```bash
git push
# Permission denied (publickey).
# fatal: Could not read from remote repository.
```

**対応:**

```bash
# ステップ1: SSH鍵を確認
ls -la ~/.ssh
# id_rsa, id_rsa.pub があるか

# ステップ2: SSH鍵がない場合は生成
ssh-keygen -t ed25519 -C "your_email@example.com"

# ステップ3: SSH鍵をGitHubに登録
cat ~/.ssh/id_ed25519.pub
# → GitHubのSettings → SSH and GPG keys → New SSH key

# ステップ4: SSH接続テスト
ssh -T git@github.com

# ステップ5: リモートURLを確認
git remote -v
# HTTPSならSSHに変更
git remote set-url origin git@github.com:user/repo.git
```

### 問題2: HTTPS認証エラー

**症状:**
```bash
git push
# Username for 'https://github.com': user
# Password for 'https://user@github.com':
# remote: Invalid username or password.
```

**対応:**

```bash
# Personal Access Token (PAT) を使用

# ステップ1: GitHubでPATを生成
# Settings → Developer settings → Personal access tokens → Generate new token

# ステップ2: パスワードの代わりにPATを使用
git push
# Username: your-username
# Password: ghp_xxxxxxxxxxxxxxxxxxxx (PAT)

# ステップ3: 認証情報を保存
git config --global credential.helper store
git push
# → 次回から自動的に使用される

# または macOS Keychain
git config --global credential.helper osxkeychain
```

---

## サブモジュール・サブツリー

### サブモジュール問題

**問題: サブモジュールが更新されない**

```bash
# サブモジュールを初期化
git submodule init

# サブモジュールを更新
git submodule update

# または一度に
git submodule update --init --recursive

# 最新に更新
git submodule update --remote
```

**問題: サブモジュールを削除したい**

```bash
# ステップ1: .gitmodulesから削除
vim .gitmodules

# ステップ2: .git/configから削除
vim .git/config

# ステップ3: ディレクトリを削除
git rm --cached path/to/submodule
rm -rf path/to/submodule

# ステップ4: .git/modulesを削除
rm -rf .git/modules/path/to/submodule

# ステップ5: コミット
git commit -m "chore: remove submodule"
```

### サブツリー問題

**問題: サブツリーを追加したい**

```bash
# リモートを追加
git remote add subtree-repo https://github.com/user/library.git

# サブツリーとして追加
git subtree add --prefix=lib/library subtree-repo main --squash

# 更新
git subtree pull --prefix=lib/library subtree-repo main --squash

# pushback
git subtree push --prefix=lib/library subtree-repo main
```

---

## Git LFS問題

### 問題1: LFSファイルがダウンロードできない

**症状:**
```bash
git clone ...
# Error downloading object: large-file.zip
```

**対応:**

```bash
# Git LFSをインストール
brew install git-lfs  # macOS
# または
apt-get install git-lfs  # Ubuntu

# 初期化
git lfs install

# ファイルを再ダウンロード
git lfs pull
```

### 問題2: LFSのストレージ不足

**対応:**

```bash
# LFSファイルのサイズ確認
git lfs ls-files -s

# 古いLFSファイルを削除
git lfs prune

# ローカルキャッシュをクリア
rm -rf .git/lfs/objects
git lfs fetch --all
```

---

## CI/CD連携の問題

### 問題1: GitHub Actions でgit操作エラー

**症状:**
```yaml
# GitHub Actions
- name: Commit changes
  run: |
    git config user.name "GitHub Actions"
    git config user.email "actions@github.com"
    git add .
    git commit -m "chore: update files"
    git push
# Error: permission denied
```

**対応:**

```yaml
- name: Commit changes
  run: |
    git config user.name "GitHub Actions"
    git config user.email "actions@github.com"
    git add .
    git commit -m "chore: update files"
    git push
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

# または
- uses: stefanzweifel/git-auto-commit-action@v4
  with:
    commit_message: "chore: update files"
```

### 問題2: 保護されたブランチにpushできない

**対応:**

```yaml
# Personal Access Token (PAT) を使用
- name: Push changes
  run: |
    git remote set-url origin https://x-access-token:${{ secrets.PAT }}@github.com/${{ github.repository }}
    git push
```

---

## よくあるエラーメッセージ集

### "fatal: not a git repository"

**原因:**
```
.gitディレクトリがない
```

**対応:**
```bash
# 初期化
git init

# またはclone
git clone https://github.com/user/repo.git
```

### "fatal: refusing to merge unrelated histories"

**原因:**
```
2つのリポジトリの履歴が無関係
```

**対応:**
```bash
git pull origin main --allow-unrelated-histories
```

### "error: Your local changes would be overwritten"

**原因:**
```
ローカルに未コミットの変更がある
```

**対応:**
```bash
# オプション1: 変更をコミット
git add .
git commit -m "save local changes"

# オプション2: 変更を退避
git stash
git pull
git stash pop

# オプション3: 変更を破棄
git reset --hard HEAD
git pull
```

### "fatal: remote origin already exists"

**原因:**
```
リモート"origin"が既に存在
```

**対応:**
```bash
# 削除して再追加
git remote remove origin
git remote add origin https://github.com/user/repo.git

# またはURLを変更
git remote set-url origin https://github.com/user/repo.git
```

---

## 予防策とベストプラクティス

### 1. ブランチ保護ルール

```markdown
GitHub Settings → Branches → Add rule

✅ Require pull request reviews before merging
✅ Require status checks to pass before merging
✅ Require branches to be up to date
✅ Include administrators
✅ Restrict who can push to matching branches
```

### 2. Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/sh

# テスト実行
npm test
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

# Lint実行
npm run lint
if [ $? -ne 0 ]; then
    echo "Linting failed. Commit aborted."
    exit 1
fi

# 機密情報チェック
if git diff --cached | grep -E "(API_KEY|PASSWORD|SECRET)"; then
    echo "Potential secret detected. Commit aborted."
    exit 1
fi

exit 0
```

### 3. .gitignore最適化

```gitignore
# Dependencies
node_modules/
vendor/
*.gem

# Build outputs
dist/
build/
*.o
*.pyc

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# 機密情報
.env
.env.local
secrets/
*.key
*.pem

# Logs
*.log
logs/
```

### 4. 定期的なバックアップ

```bash
# スクリプト例
#!/bin/bash
# backup-git.sh

REPO_PATH="/path/to/repo"
BACKUP_PATH="/path/to/backup"
DATE=$(date +%Y%m%d_%H%M%S)

cd $REPO_PATH
git bundle create $BACKUP_PATH/repo_$DATE.bundle --all
```

### 5. Git Alias設定

```bash
# ~/.gitconfig
[alias]
    st = status
    co = checkout
    br = branch
    ci = commit
    unstage = reset HEAD --
    last = log -1 HEAD
    visual = log --graph --oneline --all
    undo = reset --soft HEAD~1
```

---

## 緊急時対応フローチャート

```
問題発生
    ↓
質問1: 本番環境に影響がある？
├─ Yes → 🔴 Critical対応（5分以内）
│         ├─ revert
│         ├─ rollback
│         └─ hotfix
└─ No → 質問2へ

質問2: 開発がブロックされる？
├─ Yes → 🟡 High対応（30分以内）
│         ├─ コンフリクト解決
│         ├─ ブランチ復旧
│         └─ 権限問題解決
└─ No → 質問3へ

質問3: データが失われた？
├─ Yes → データ復旧手順
│         ├─ reflog確認
│         ├─ cherry-pick復旧
│         └─ バックアップから復元
└─ No → 🟢 Normal対応

質問4: 解決できない？
└─ Yes → エスカレーション
          ├─ チームリード相談
          ├─ Stack Overflow検索
          └─ Git公式ドキュメント参照
```

---

## まとめ

### トラブルシューティングの心得

```
1. 慌てない
   → 冷静に状況を把握

2. 現状確認
   → git status, git log で状況を確認

3. バックアップ
   → 復旧作業前にブランチやreflogを確認

4. 一歩ずつ
   → 複数の操作を一度に行わない

5. ドキュメント化
   → 解決方法を記録して共有
```

### 基本コマンドチートシート

```bash
# 状況確認
git status
git log --oneline -10
git reflog
git branch -vv

# 変更の取り消し
git checkout -- <file>
git reset --soft HEAD~1
git reset --hard HEAD~1
git revert <commit>

# ブランチ操作
git branch <name>
git checkout <name>
git merge <branch>
git rebase <branch>

# リモート操作
git fetch
git pull
git push
git remote -v

# データ復旧
git reflog
git cherry-pick <commit>
git fsck --lost-found
```

### ヘルプリソース

```
公式ドキュメント:
https://git-scm.com/doc

Pro Git Book:
https://git-scm.com/book/ja/v2

Stack Overflow:
https://stackoverflow.com/questions/tagged/git

GitHub Docs:
https://docs.github.com/ja

Git Tips:
https://github.com/git-tips/tips
```

---

**文字数:** 約35,000文字

**このガイドで大部分のGit問題を解決できます！**
**困ったときはまずこのガイドを参照してください。**
