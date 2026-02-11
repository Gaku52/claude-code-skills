# アクセス制御

> OSのセキュリティの基盤は「誰が何にアクセスできるか」を厳密に管理するアクセス制御である。

## この章で学ぶこと

- [ ] DAC、MAC、RBACの違いを説明できる
- [ ] Unixパーミッションモデルを理解する
- [ ] 最小権限の原則を実践できる

---

## 1. アクセス制御モデル

```
3つのアクセス制御モデル:

  1. DAC（Discretionary Access Control）— 任意アクセス制御:
     ファイルの所有者がアクセス権を自由に設定
     → Unixの標準パーミッション（rwx）
     → Windows NTFS のACL
     → 柔軟だが、所有者が誤った設定をすると脆弱

  2. MAC（Mandatory Access Control）— 強制アクセス制御:
     システム管理者が定めたポリシーに従う（所有者でも変更不可）
     → SELinux, AppArmor
     → 軍事・政府システムで必須
     → Bell-LaPadula モデル: 上位の情報は下位に読めない

  3. RBAC（Role-Based Access Control）— ロールベース:
     ユーザーにロール（役割）を割り当て、ロールに権限を付与
     → 管理者、開発者、閲覧者 等の役割
     → AWS IAM, Kubernetes RBAC
     → 大規模組織の標準

  比較:
  ┌──────┬──────────────┬──────────┬──────────┐
  │ モデル│ 決定者       │ 柔軟性   │ セキュリティ│
  ├──────┼──────────────┼──────────┼──────────┤
  │ DAC  │ ファイル所有者│ 高       │ 中        │
  │ MAC  │ システム管理者│ 低       │ 高        │
  │ RBAC │ 管理者(ロール)│ 中       │ 高        │
  └──────┴──────────────┴──────────┴──────────┘
```

---

## 2. Unixパーミッション

```
パーミッションの表示:
  $ ls -la
  -rwxr-xr-- 1 user group 4096 Jan 1 file.txt
  │├─┤├─┤├─┤
  │ │  │  └── other（その他）: r-- (読み取りのみ)
  │ │  └── group（グループ）: r-x (読取+実行)
  │ └── owner（所有者）: rwx (読取+書込+実行)
  └── タイプ: - (ファイル), d (ディレクトリ), l (リンク)

  数値表現:
  rwx = 4+2+1 = 7
  r-x = 4+0+1 = 5
  r-- = 4+0+0 = 4
  → chmod 754 file.txt

  特殊パーミッション:
  SUID (4xxx): 実行時に所有者の権限で動作
  → /usr/bin/passwd は SUID root（一般ユーザーが/etc/shadowを更新）

  SGID (2xxx): 実行時にグループの権限で動作
  Sticky (1xxx): ディレクトリ内の削除を所有者のみに制限
  → /tmp は sticky bit が設定されている

  ACL（Access Control List）:
  標準パーミッションを超える細かい制御
  $ setfacl -m u:alice:rw file.txt  # aliceに読み書き許可
  $ getfacl file.txt                 # ACL確認
```

---

## 3. Linux セキュリティモジュール

```
SELinux（Security-Enhanced Linux, NSA開発）:
  MAC の実装。全プロセスとファイルにラベルを付与

  タイプエンフォースメント:
  httpd_t プロセスは httpd_sys_content_t ファイルのみアクセス可能
  → Webサーバーが乗っ取られても他のファイルにアクセスできない

  モード:
  - Enforcing: ポリシー違反を拒否+ログ
  - Permissive: ログのみ（デバッグ用）
  - Disabled: 無効

  $ getenforce                # 現在のモード確認
  $ sudo setenforce 0         # Permissive に一時変更

AppArmor（Ubuntu デフォルト）:
  パスベースのMAC。SELinuxより設定が簡単
  → プロファイルでプロセスのアクセスを制限
  → /etc/apparmor.d/ にプロファイル

seccomp（Secure Computing Mode）:
  プロセスが使用できるシステムコールを制限
  → Dockerコンテナのデフォルトセキュリティ
  → Chromeのサンドボックスでも使用
```

---

## 4. 最小権限の原則

```
最小権限の原則（Principle of Least Privilege）:
  プロセスやユーザーに必要最小限の権限のみを付与

  実践例:
  ┌──────────────────────────────────────────┐
  │ ✗ root で Web サーバーを実行            │
  │ ✓ 専用ユーザー（www-data）で実行        │
  │                                          │
  │ ✗ chmod 777 で全開放                    │
  │ ✓ 必要な権限のみ設定                    │
  │                                          │
  │ ✗ アプリにroot権限を付与                │
  │ ✓ capabilities で必要な権限のみ付与     │
  └──────────────────────────────────────────┘

  Linux Capabilities:
  root の権限を細分化して必要なものだけ付与
  → CAP_NET_BIND_SERVICE: 1024未満のポートにバインド
  → CAP_SYS_PTRACE: 他プロセスのデバッグ
  → CAP_DAC_OVERRIDE: ファイルパーミッション無視

  $ sudo setcap cap_net_bind_service=+ep ./server
  → rootなしで80番ポートにバインド可能
```

---

## 実践演習

### 演習1: [基礎] — パーミッション操作

```bash
# ファイルの権限を操作
touch test.txt
chmod 644 test.txt && ls -la test.txt
chmod u+x test.txt && ls -la test.txt
chmod o-r test.txt && ls -la test.txt

# 所有者変更
sudo chown root:root test.txt
```

### 演習2: [応用] — セキュリティ監査

```bash
# SUID ファイルの検索（セキュリティ監査で重要）
find / -perm -4000 -type f 2>/dev/null

# 書き込み可能なファイルの検索
find /etc -writable -type f 2>/dev/null
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| DAC | 所有者が権限設定。Unix標準。柔軟 |
| MAC | 管理者が強制。SELinux, AppArmor |
| RBAC | ロールベース。AWS IAM, K8s |
| 最小権限 | 必要最小限の権限のみ付与 |

---

## 次に読むべきガイド
→ [[01-sandboxing.md]] — サンドボックスと隔離

---

## 参考文献
1. Bishop, M. "Computer Security: Art and Science." 2nd Ed, Addison-Wesley, 2018.
