# 初めてのアプリ作成 - 総合演習

## 目次

1. [概要](#概要)
2. [プロジェクトのゴール](#プロジェクトのゴール)
3. [プロジェクトセットアップ](#プロジェクトセットアップ)
4. [データモデル設計](#データモデル設計)
5. [ViewModel実装](#viewmodel実装)
6. [UI実装](#ui実装)
7. [機能追加](#機能追加)
8. [まとめ](#まとめ)

---

## 概要

### 何を学ぶか

このチュートリアルでは、これまで学んだ全ての概念を統合して、**メモアプリ**を実装します。

### 実装する機能

- ✅ メモの一覧表示
- ✅ メモの追加
- ✅ メモの編集
- ✅ メモの削除
- ✅ お気に入り機能
- ✅ データの永続化

### 学習時間：2〜3時間

---

## プロジェクトのゴール

### 完成するアプリ

```
メモアプリ
├── ホーム画面（メモ一覧）
├── メモ追加画面
└── メモ詳細・編集画面
```

---

## プロジェクトセットアップ

### ステップ1：新規プロジェクト作成

```
1. Xcode起動
2. Create New Project
3. iOS → App
4. Product Name: SimpleNotes
5. Interface: SwiftUI
6. Language: Swift
7. プロジェクトを保存
```

### ステップ2：プロジェクト構成

```
SimpleNotes/
├── Models/
│   └── Note.swift
├── ViewModels/
│   └── NotesViewModel.swift
├── Views/
│   ├── ContentView.swift
│   ├── AddNoteView.swift
│   └── NoteDetailView.swift
└── SimpleNotesApp.swift
```

---

## データモデル設計

### Note モデル

新規ファイル作成：`Models/Note.swift`

```swift
import Foundation

struct Note: Identifiable, Codable {
    let id: UUID
    var title: String
    var content: String
    var createdAt: Date
    var isFavorite: Bool

    init(id: UUID = UUID(), title: String, content: String, isFavorite: Bool = false) {
        self.id = id
        self.title = title
        self.content = content
        self.createdAt = Date()
        self.isFavorite = isFavorite
    }
}
```

**ポイント**：
- `Identifiable`：リスト表示に必要
- `Codable`：データ永続化に必要
- `UUID`：一意なID
- `Date`：作成日時

---

## ViewModel実装

### NotesViewModel

新規ファイル作成：`ViewModels/NotesViewModel.swift`

```swift
import Foundation
import Combine

class NotesViewModel: ObservableObject {
    @Published var notes: [Note] = []

    private let savePath = FileManager.documentsDirectory.appendingPathComponent("notes.json")

    init() {
        loadNotes()
    }

    // メモを追加
    func addNote(title: String, content: String) {
        let note = Note(title: title, content: content)
        notes.insert(note, at: 0)
        saveNotes()
    }

    // メモを更新
    func updateNote(_ note: Note) {
        if let index = notes.firstIndex(where: { $0.id == note.id }) {
            notes[index] = note
            saveNotes()
        }
    }

    // メモを削除
    func deleteNote(at offsets: IndexSet) {
        notes.remove(atOffsets: offsets)
        saveNotes()
    }

    // お気に入りをトグル
    func toggleFavorite(_ note: Note) {
        if let index = notes.firstIndex(where: { $0.id == note.id }) {
            notes[index].isFavorite.toggle()
            saveNotes()
        }
    }

    // データを保存
    private func saveNotes() {
        do {
            let data = try JSONEncoder().encode(notes)
            try data.write(to: savePath)
        } catch {
            print("保存エラー: \(error.localizedDescription)")
        }
    }

    // データを読み込み
    private func loadNotes() {
        do {
            let data = try Data(contentsOf: savePath)
            notes = try JSONDecoder().decode([Note].self, from: data)
        } catch {
            // ファイルがない場合はサンプルデータを作成
            notes = [
                Note(title: "ようこそ", content: "初めてのメモアプリです"),
                Note(title: "買い物リスト", content: "牛乳、卵、パン")
            ]
        }
    }
}

// FileManager拡張
extension FileManager {
    static var documentsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
}
```

---

## UI実装

### 1. メイン画面（ContentView）

`Views/ContentView.swift`を編集：

```swift
import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = NotesViewModel()
    @State private var showAddNote = false

    var body: some View {
        NavigationView {
            Group {
                if viewModel.notes.isEmpty {
                    EmptyStateView()
                } else {
                    notesList
                }
            }
            .navigationTitle("メモ")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showAddNote = true }) {
                        Image(systemName: "plus")
                    }
                }
            }
            .sheet(isPresented: $showAddNote) {
                AddNoteView(viewModel: viewModel)
            }
        }
    }

    private var notesList: some View {
        List {
            // お気に入り
            if viewModel.notes.contains(where: { $0.isFavorite }) {
                Section(header: Text("お気に入り")) {
                    ForEach(viewModel.notes.filter { $0.isFavorite }) { note in
                        NavigationLink(destination: NoteDetailView(note: note, viewModel: viewModel)) {
                            NoteRow(note: note, viewModel: viewModel)
                        }
                    }
                }
            }

            // 全てのメモ
            Section(header: Text("全てのメモ")) {
                ForEach(viewModel.notes) { note in
                    NavigationLink(destination: NoteDetailView(note: note, viewModel: viewModel)) {
                        NoteRow(note: note, viewModel: viewModel)
                    }
                }
                .onDelete(perform: viewModel.deleteNote)
            }
        }
    }
}

// メモ行
struct NoteRow: View {
    let note: Note
    @ObservedObject var viewModel: NotesViewModel

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 5) {
                Text(note.title)
                    .font(.headline)

                Text(note.content)
                    .font(.subheadline)
                    .foregroundColor(.gray)
                    .lineLimit(2)

                Text(note.createdAt, style: .date)
                    .font(.caption)
                    .foregroundColor(.gray)
            }

            Spacer()

            Button(action: {
                viewModel.toggleFavorite(note)
            }) {
                Image(systemName: note.isFavorite ? "star.fill" : "star")
                    .foregroundColor(note.isFavorite ? .yellow : .gray)
            }
            .buttonStyle(.plain)
        }
    }
}

// 空の状態
struct EmptyStateView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "note.text")
                .font(.system(size: 80))
                .foregroundColor(.gray)

            Text("メモがありません")
                .font(.title2)
                .foregroundColor(.gray)

            Text("右上の + ボタンから\n新しいメモを追加しましょう")
                .font(.subheadline)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
        }
    }
}

#Preview {
    ContentView()
}
```

### 2. メモ追加画面

新規ファイル作成：`Views/AddNoteView.swift`

```swift
import SwiftUI

struct AddNoteView: View {
    @Environment(\.dismiss) var dismiss
    @ObservedObject var viewModel: NotesViewModel

    @State private var title = ""
    @State private var content = ""
    @FocusState private var focusedField: Field?

    enum Field {
        case title, content
    }

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("タイトル")) {
                    TextField("タイトルを入力", text: $title)
                        .focused($focusedField, equals: .title)
                }

                Section(header: Text("内容")) {
                    TextEditor(text: $content)
                        .frame(minHeight: 200)
                        .focused($focusedField, equals: .content)
                }
            }
            .navigationTitle("新規メモ")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("キャンセル") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("保存") {
                        viewModel.addNote(title: title.isEmpty ? "無題" : title, content: content)
                        dismiss()
                    }
                    .disabled(content.isEmpty)
                }
            }
            .onAppear {
                focusedField = .title
            }
        }
    }
}

#Preview {
    AddNoteView(viewModel: NotesViewModel())
}
```

### 3. メモ詳細・編集画面

新規ファイル作成：`Views/NoteDetailView.swift`

```swift
import SwiftUI

struct NoteDetailView: View {
    let note: Note
    @ObservedObject var viewModel: NotesViewModel
    @Environment(\.dismiss) var dismiss

    @State private var title: String
    @State private var content: String
    @State private var isEditing = false

    init(note: Note, viewModel: NotesViewModel) {
        self.note = note
        self.viewModel = viewModel
        _title = State(initialValue: note.title)
        _content = State(initialValue: note.content)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // タイトル
                if isEditing {
                    TextField("タイトル", text: $title)
                        .font(.title)
                        .fontWeight(.bold)
                } else {
                    Text(note.title)
                        .font(.title)
                        .fontWeight(.bold)
                }

                // 日付
                Text(note.createdAt, style: .date)
                    .font(.caption)
                    .foregroundColor(.gray)

                Divider()

                // 内容
                if isEditing {
                    TextEditor(text: $content)
                        .frame(minHeight: 300)
                } else {
                    Text(note.content)
                        .font(.body)
                }

                Spacer()
            }
            .padding()
        }
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(isEditing ? "完了" : "編集") {
                    if isEditing {
                        // 保存
                        var updatedNote = note
                        updatedNote.title = title
                        updatedNote.content = content
                        viewModel.updateNote(updatedNote)
                    }
                    isEditing.toggle()
                }
            }

            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: {
                    viewModel.toggleFavorite(note)
                }) {
                    Image(systemName: note.isFavorite ? "star.fill" : "star")
                        .foregroundColor(note.isFavorite ? .yellow : .gray)
                }
            }
        }
    }
}

#Preview {
    NavigationView {
        NoteDetailView(
            note: Note(title: "サンプル", content: "これはサンプルのメモです"),
            viewModel: NotesViewModel()
        )
    }
}
```

---

## 機能追加

### 1. 検索機能

`ContentView.swift`に追加：

```swift
@State private var searchText = ""

var filteredNotes: [Note] {
    if searchText.isEmpty {
        return viewModel.notes
    } else {
        return viewModel.notes.filter { note in
            note.title.localizedCaseInsensitiveContains(searchText) ||
            note.content.localizedCaseInsensitiveContains(searchText)
        }
    }
}

// ビューに追加
.searchable(text: $searchText, prompt: "メモを検索")
```

### 2. ソート機能

`NotesViewModel.swift`に追加：

```swift
enum SortOption {
    case date, title
}

@Published var sortOption: SortOption = .date

var sortedNotes: [Note] {
    switch sortOption {
    case .date:
        return notes.sorted { $0.createdAt > $1.createdAt }
    case .title:
        return notes.sorted { $0.title < $1.title }
    }
}
```

---

## テストとデバッグ

### テスト項目

- [ ] メモを追加できる
- [ ] メモを編集できる
- [ ] メモを削除できる
- [ ] お気に入りを切り替えられる
- [ ] アプリを再起動してもデータが残る
- [ ] 検索が動作する

### デバッグのヒント

```swift
// データの確認
print("Notes count: \(viewModel.notes.count)")
print("Save path: \(savePath)")

// Previewでのテスト
#Preview {
    ContentView()
}
```

---

## まとめ

### このチュートリアルで学んだこと

- ✅ MVVMアーキテクチャ
- ✅ データモデル設計
- ✅ CRUD操作の実装
- ✅ データの永続化
- ✅ ナビゲーション
- ✅ リスト表示
- ✅ フォーム入力

### 拡張アイデア

1. **カテゴリー機能**
   - メモをカテゴリー分け
   - カテゴリー別フィルター

2. **画像添付**
   - メモに画像を追加
   - Photo Pickerの統合

3. **共有機能**
   - メモをテキストとして共有
   - Share Sheetの実装

4. **パスワードロック**
   - アプリロック機能
   - Face ID / Touch ID認証

5. **iCloudSync**
   - 複数デバイス間の同期
   - CloudKit統合

---

## 次のステップ

### 学習の振り返り

このガイドシリーズで学んだこと：
1. ✅ iOS開発の基礎
2. ✅ Swift言語
3. ✅ Xcode操作
4. ✅ SwiftUI
5. ✅ レイアウト
6. ✅ 状態管理
7. ✅ 実践的なアプリ開発

### さらに学ぶべきこと

**中級トピック**：
- Navigation Stack
- async/await（非同期処理）
- API通信（URLSession）
- Core Data（高度なデータ管理）
- アニメーション
- カスタムビュー

**上級トピック**：
- Combine フレームワーク
- Widget 開発
- App Clip
- App Store 申請

---

**前のガイド**：[06-state-management.md](./06-state-management.md)

**親ガイド**：[iOS Development - SKILL.md](../../SKILL.md)

**おめでとうございます！** iOS開発の基礎を全て学びました。
