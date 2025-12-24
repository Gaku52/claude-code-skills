---
name: ios-project-setup
description: 新規iOSプロジェクト作成時の初期設定、フォルダ構成、Xcode設定、依存関係管理、ビルド設定の最適化、チーム開発環境構築まで、プロジェクト開始時の全てをカバー。
---

# iOS Project Setup Skill

## 📋 目次

1. [概要](#概要)
2. [いつ使うか](#いつ使うか)
3. [プロジェクト作成](#プロジェクト作成)
4. [フォルダ構成](#フォルダ構成)
5. [Xcode設定](#xcode設定)
6. [依存関係管理](#依存関係管理)
7. [チーム環境構築](#チーム環境構築)
8. [Agent連携](#agent連携)

---

## 概要

- ✅ Xcodeプロジェクト作成・初期設定
- ✅ フォルダ構成ベストプラクティス
- ✅ ビルド設定最適化
- ✅ Scheme・Configuration管理
- ✅ 依存関係管理（SPM, CocoaPods, Carthage）
- ✅ チーム開発環境統一
- ✅ CI/CD初期設定

---

## いつ使うか

- 新規iOSプロジェクト開始時
- プロジェクト構成の見直し時
- チームメンバーの環境構築時

---

## プロジェクト作成

### 1. Xcodeプロジェクト作成

[guides/01-xcode-project-creation.md](guides/01-xcode-project-creation.md)

### 2. Git初期化

```bash
git init
git add .
git commit -m "feat(init): initial project setup"
```

### 3. .gitignore設定

[templates/.gitignore](templates/.gitignore)

詳細: [guides/02-git-setup.md](guides/02-git-setup.md)

---

## フォルダ構成

### 推奨構成（MVVM）

```
YourApp/
├── App/
│   ├── AppDelegate.swift
│   └── SceneDelegate.swift
├── Models/
│   ├── User.swift
│   └── ...
├── ViewModels/
│   ├── UserViewModel.swift
│   └── ...
├── Views/
│   ├── Home/
│   ├── Profile/
│   └── ...
├── Repositories/
│   ├── UserRepository.swift
│   └── ...
├── Services/
│   ├── APIClient.swift
│   ├── DatabaseService.swift
│   └── ...
├── Utilities/
│   ├── Extensions/
│   └── Helpers/
└── Resources/
    ├── Assets.xcassets
    └── Localizable.strings
```

詳細: [guides/03-folder-structure.md](guides/03-folder-structure.md)

---

## Xcode設定

### Build Settings最適化

[guides/04-build-settings.md](guides/04-build-settings.md)

### Scheme管理

- Debug
- Staging
- Release

[guides/05-scheme-configuration.md](guides/05-scheme-configuration.md)

### Xcconfig活用

[templates/Configs/](templates/Configs/)

詳細: [guides/06-xcconfig.md](guides/06-xcconfig.md)

---

## 依存関係管理

### SPM vs CocoaPods vs Carthage

| 機能 | SPM | CocoaPods | Carthage |
|------|-----|-----------|----------|
| 公式サポート | ✅ | ❌ | ❌ |
| 設定の簡単さ | ✅ | ⭕ | ⭕ |
| ビルド速度 | ✅ | ⭕ | ✅ |
| 推奨度 | ⭐⭐⭐ | ⭐⭐ | ⭐ |

詳細: [guides/07-dependency-management.md](guides/07-dependency-management.md)

---

## チーム環境構築

### README作成

[templates/README.md](templates/README.md)

### 環境構築手順書

[guides/08-onboarding.md](guides/08-onboarding.md)

---

## Agent連携

### 使用するAgents

1. **project-initializer-agent** - プロジェクト自動作成
2. **dependency-setup-agent** - 依存関係自動セットアップ

---

## 詳細ドキュメント

### Guides
1. [Xcodeプロジェクト作成](guides/01-xcode-project-creation.md)
2. [Git初期設定](guides/02-git-setup.md)
3. [フォルダ構成](guides/03-folder-structure.md)
4. [ビルド設定](guides/04-build-settings.md)
5. [Scheme/Configuration](guides/05-scheme-configuration.md)
6. [Xcconfig活用](guides/06-xcconfig.md)
7. [依存関係管理](guides/07-dependency-management.md)
8. [オンボーディング](guides/08-onboarding.md)

### Checklists
- [プロジェクト作成時](checklists/project-creation.md)
- [環境構築時](checklists/environment-setup.md)

### Templates
- [.gitignore](templates/.gitignore)
- [README.md](templates/README.md)
- [Xcconfig](templates/Configs/)

---

## 関連Skills

- `dependency-management` - 依存関係詳細
- `git-workflow` - Git運用
- `ci-cd-automation` - CI/CD設定
