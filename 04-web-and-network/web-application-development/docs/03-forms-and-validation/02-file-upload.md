# ファイルアップロード

> ファイルアップロードはWebフォームの中でも特に複雑な領域である。ドラッグ&ドロップ、プログレス表示、プレビュー、S3直接アップロード、画像リサイズ、チャンクアップロード、セキュリティ対策まで、プロダクション品質のファイルアップロード実装を体系的に習得する。本ガイドではフロントエンドからバックエンドまで、実際のプロジェクトで必要となるあらゆるパターンを網羅する。

## この章で学ぶこと

- [ ] HTML5 File API の基礎と `<input type="file">` の詳細な挙動を理解する
- [ ] ドラッグ&ドロップアップロードの実装パターンを習得する
- [ ] プログレスバー表示付きアップロードを XMLHttpRequest と Fetch API で実装する
- [ ] S3プリサインドURLによる直接アップロードを把握する
- [ ] 画像プレビュー・リサイズ・バリデーションの実装を学ぶ
- [ ] チャンクアップロード（分割アップロード）の仕組みと実装を理解する
- [ ] マルチファイルアップロードの UX 設計と実装を習得する
- [ ] サーバーサイドでのファイル受信・検証・保存のベストプラクティスを学ぶ
- [ ] セキュリティ対策（MIME検証、ウイルススキャン、パストラバーサル防止）を実装する
- [ ] 大規模ファイルアップロードのアーキテクチャ設計を理解する

---

## 1. HTML5 File API の基礎

### 1.1 `<input type="file">` の基本

ファイルアップロードの最も基本的な要素は HTML の `<input type="file">` である。このシンプルな要素が提供する属性と動作を正しく理解することが、高度なアップロード機能を実装する上での基盤となる。

```html
<!-- 基本的なファイル入力 -->
<input type="file" name="document" />

<!-- 複数ファイル選択を許可 -->
<input type="file" name="photos" multiple />

<!-- 受け入れるファイルタイプを制限 -->
<input type="file" accept=".pdf,.doc,.docx" />
<input type="file" accept="image/*" />
<input type="file" accept="image/jpeg,image/png,image/webp" />
<input type="file" accept="video/*" />
<input type="file" accept="audio/*" />

<!-- カメラを直接起動（モバイル） -->
<input type="file" accept="image/*" capture="environment" />
<input type="file" accept="image/*" capture="user" />
<input type="file" accept="video/*" capture="environment" />

<!-- ディレクトリ選択（Chrome/Edge） -->
<input type="file" webkitdirectory />
```

### 1.2 accept 属性の詳細

`accept` 属性はブラウザのファイル選択ダイアログでフィルタリングを行うが、セキュリティ上の制約ではない点に注意が必要である。ユーザーは「すべてのファイル」を選択してフィルタを回避できるため、サーバーサイドでの検証は必須となる。

```typescript
// accept 属性で指定可能な形式一覧
const acceptFormats = {
  // MIME タイプによる指定
  'image/jpeg': 'JPEG画像',
  'image/png': 'PNG画像',
  'image/webp': 'WebP画像',
  'image/gif': 'GIF画像',
  'image/svg+xml': 'SVG画像',
  'application/pdf': 'PDFファイル',
  'application/json': 'JSONファイル',
  'text/csv': 'CSVファイル',
  'text/plain': 'テキストファイル',
  'application/zip': 'ZIPアーカイブ',
  'video/mp4': 'MP4動画',
  'audio/mpeg': 'MP3音声',

  // ワイルドカードによる指定
  'image/*': 'すべての画像形式',
  'video/*': 'すべての動画形式',
  'audio/*': 'すべての音声形式',

  // 拡張子による指定
  '.pdf': 'PDFファイル',
  '.xlsx': 'Excelファイル',
  '.docx': 'Wordファイル',
  '.pptx': 'PowerPointファイル',
};
```

### 1.3 File オブジェクトと FileList

`<input type="file">` から取得できる File オブジェクトと FileList の構造を理解することが重要である。

```typescript
// File オブジェクトのプロパティ
interface FileInfo {
  name: string;            // ファイル名（例: "photo.jpg"）
  size: number;            // ファイルサイズ（バイト）
  type: string;            // MIME タイプ（例: "image/jpeg"）
  lastModified: number;    // 最終更新日時（UNIXタイムスタンプ）
  webkitRelativePath: string; // webkitdirectory使用時の相対パス
}

// FileList の操作
function handleFileInput(event: Event) {
  const input = event.target as HTMLInputElement;
  const files = input.files; // FileList オブジェクト

  if (!files || files.length === 0) {
    console.log('ファイルが選択されていません');
    return;
  }

  // FileList は配列ではないが、イテラブル
  // Array.from() で配列に変換可能
  const fileArray = Array.from(files);

  fileArray.forEach(file => {
    console.log(`名前: ${file.name}`);
    console.log(`サイズ: ${formatFileSize(file.size)}`);
    console.log(`タイプ: ${file.type}`);
    console.log(`最終更新: ${new Date(file.lastModified).toLocaleString()}`);
  });
}

// ファイルサイズのフォーマット関数
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}
```

### 1.4 FileReader API

FileReader API を使用すると、ファイルの内容をブラウザ上で読み取ることができる。プレビュー表示やクライアントサイドの処理に不可欠である。

```typescript
// FileReader の読み取りメソッド一覧
class FileReaderExample {
  // テキストファイルを読み取り
  readAsText(file: File, encoding = 'UTF-8'): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(reader.error);
      reader.readAsText(file, encoding);
    });
  }

  // 画像をData URLとして読み取り（プレビュー用）
  readAsDataURL(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    });
  }

  // バイナリデータとして読み取り
  readAsArrayBuffer(file: File): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as ArrayBuffer);
      reader.onerror = () => reject(reader.error);
      reader.readAsArrayBuffer(file);
    });
  }

  // 進捗付き読み取り
  readWithProgress(
    file: File,
    onProgress: (percent: number) => void
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onprogress = (event) => {
        if (event.lengthComputable) {
          const percent = Math.round((event.loaded / event.total) * 100);
          onProgress(percent);
        }
      };

      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    });
  }
}
```

### 1.5 Blob API とファイル操作

Blob（Binary Large Object）は File オブジェクトの親クラスであり、バイナリデータの操作に使用する。

```typescript
// Blob の生成と操作
class BlobOperations {
  // テキストからBlobを生成
  createTextBlob(content: string, type = 'text/plain'): Blob {
    return new Blob([content], { type });
  }

  // JSONからBlobを生成
  createJsonBlob(data: unknown): Blob {
    const json = JSON.stringify(data, null, 2);
    return new Blob([json], { type: 'application/json' });
  }

  // Blob をスライス（部分的に読み取り）
  sliceBlob(blob: Blob, start: number, end: number): Blob {
    return blob.slice(start, end, blob.type);
  }

  // Blob から File に変換
  blobToFile(blob: Blob, filename: string): File {
    return new File([blob], filename, {
      type: blob.type,
      lastModified: Date.now(),
    });
  }

  // Blob URLの生成と解放
  createObjectURL(blob: Blob): string {
    return URL.createObjectURL(blob);
  }

  revokeObjectURL(url: string): void {
    URL.revokeObjectURL(url);
  }

  // ArrayBuffer から Blob へ変換
  arrayBufferToBlob(buffer: ArrayBuffer, type: string): Blob {
    return new Blob([buffer], { type });
  }

  // Blob をダウンロード
  downloadBlob(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}
```

---

## 2. 基本的なファイルアップロード実装

### 2.1 React Hook Form + ファイル入力

React Hook Form を使用した基本的なファイルアップロードフォームの実装パターンを示す。

```typescript
import { useForm, SubmitHandler } from 'react-hook-form';
import { useState, useCallback } from 'react';

// フォームの型定義
interface UploadFormData {
  title: string;
  description: string;
  category: string;
  file: FileList;
}

// バリデーションルール
const FILE_VALIDATION = {
  maxSize: 5 * 1024 * 1024, // 5MB
  allowedTypes: ['image/jpeg', 'image/png', 'image/webp'] as const,
  allowedExtensions: ['.jpg', '.jpeg', '.png', '.webp'] as const,
};

function FileUploadForm() {
  const [preview, setPreview] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<
    'idle' | 'uploading' | 'success' | 'error'
  >('idle');

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    watch,
    reset,
    setError,
  } = useForm<UploadFormData>();

  // ファイル変更を監視してプレビュー生成
  const watchFile = watch('file');

  React.useEffect(() => {
    if (watchFile?.[0]) {
      const file = watchFile[0];
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setPreview(null);
    }
  }, [watchFile]);

  const onSubmit: SubmitHandler<UploadFormData> = async (data) => {
    try {
      setUploadStatus('uploading');
      const formData = new FormData();
      formData.append('file', data.file[0]);
      formData.append('title', data.title);
      formData.append('description', data.description);
      formData.append('category', data.category);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
        // Content-Type ヘッダーは設定しない
        // ブラウザが自動的に multipart/form-data を設定する
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'アップロードに失敗しました');
      }

      const result = await response.json();
      setUploadStatus('success');
      reset();
      setPreview(null);
      console.log('Upload successful:', result);
    } catch (error) {
      setUploadStatus('error');
      setError('root', {
        message: error instanceof Error ? error.message : 'アップロードエラー',
      });
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      {/* タイトル入力 */}
      <div>
        <label htmlFor="title" className="block text-sm font-medium">
          タイトル
        </label>
        <input
          id="title"
          type="text"
          {...register('title', {
            required: 'タイトルは必須です',
            maxLength: { value: 100, message: '100文字以内で入力してください' },
          })}
          className="mt-1 block w-full border rounded-md px-3 py-2"
        />
        {errors.title && (
          <p className="mt-1 text-sm text-red-600">{errors.title.message}</p>
        )}
      </div>

      {/* 説明入力 */}
      <div>
        <label htmlFor="description" className="block text-sm font-medium">
          説明
        </label>
        <textarea
          id="description"
          rows={3}
          {...register('description', {
            maxLength: { value: 500, message: '500文字以内で入力してください' },
          })}
          className="mt-1 block w-full border rounded-md px-3 py-2"
        />
        {errors.description && (
          <p className="mt-1 text-sm text-red-600">{errors.description.message}</p>
        )}
      </div>

      {/* カテゴリ選択 */}
      <div>
        <label htmlFor="category" className="block text-sm font-medium">
          カテゴリ
        </label>
        <select
          id="category"
          {...register('category', { required: 'カテゴリを選択してください' })}
          className="mt-1 block w-full border rounded-md px-3 py-2"
        >
          <option value="">選択してください</option>
          <option value="profile">プロフィール画像</option>
          <option value="document">ドキュメント</option>
          <option value="gallery">ギャラリー</option>
        </select>
        {errors.category && (
          <p className="mt-1 text-sm text-red-600">{errors.category.message}</p>
        )}
      </div>

      {/* ファイル入力 */}
      <div>
        <label htmlFor="file" className="block text-sm font-medium">
          ファイル
        </label>
        <input
          id="file"
          type="file"
          accept="image/jpeg,image/png,image/webp"
          {...register('file', {
            required: 'ファイルを選択してください',
            validate: {
              size: (files) =>
                !files[0] ||
                files[0].size <= FILE_VALIDATION.maxSize ||
                `ファイルサイズは${formatFileSize(FILE_VALIDATION.maxSize)}以下にしてください`,
              type: (files) =>
                !files[0] ||
                (FILE_VALIDATION.allowedTypes as readonly string[]).includes(
                  files[0].type
                ) ||
                'JPEG、PNG、WebP形式のみ対応しています',
              notEmpty: (files) =>
                !files[0] ||
                files[0].size > 0 ||
                '空のファイルはアップロードできません',
            },
          })}
          className="mt-1 block w-full text-sm file:mr-4 file:py-2 file:px-4
            file:rounded-md file:border-0 file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />
        {errors.file && (
          <p className="mt-1 text-sm text-red-600">{errors.file.message}</p>
        )}
        <p className="mt-1 text-xs text-gray-500">
          JPEG, PNG, WebP（最大5MB）
        </p>
      </div>

      {/* プレビュー表示 */}
      {preview && (
        <div className="mt-4">
          <p className="text-sm font-medium mb-2">プレビュー:</p>
          <img
            src={preview}
            alt="プレビュー"
            className="max-w-xs max-h-48 object-contain rounded-lg border"
          />
        </div>
      )}

      {/* エラーメッセージ */}
      {errors.root && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-md">
          <p className="text-sm text-red-600">{errors.root.message}</p>
        </div>
      )}

      {/* 送信ボタン */}
      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full py-2 px-4 bg-blue-600 text-white rounded-md
          hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isSubmitting ? 'アップロード中...' : 'アップロード'}
      </button>

      {/* ステータス表示 */}
      {uploadStatus === 'success' && (
        <p className="text-sm text-green-600">アップロードが完了しました</p>
      )}
    </form>
  );
}
```

### 2.2 FormData の詳細な使い方

FormData はファイルアップロードの中核となる API であり、`multipart/form-data` 形式でデータを構築する。

```typescript
// FormData の高度な使い方
class FormDataBuilder {
  private formData: FormData;

  constructor() {
    this.formData = new FormData();
  }

  // 単一ファイルの追加
  addFile(key: string, file: File): this {
    this.formData.append(key, file, file.name);
    return this;
  }

  // 複数ファイルの追加（同じキーに複数）
  addFiles(key: string, files: File[]): this {
    files.forEach(file => {
      this.formData.append(key, file, file.name);
    });
    return this;
  }

  // Blobの追加（ファイル名を指定）
  addBlob(key: string, blob: Blob, filename: string): this {
    this.formData.append(key, blob, filename);
    return this;
  }

  // テキストデータの追加
  addField(key: string, value: string | number | boolean): this {
    this.formData.append(key, String(value));
    return this;
  }

  // JSONデータをフィールドとして追加
  addJson(key: string, data: unknown): this {
    this.formData.append(key, JSON.stringify(data));
    return this;
  }

  // FormData の内容をログ出力（デバッグ用）
  debug(): void {
    for (const [key, value] of this.formData.entries()) {
      if (value instanceof File) {
        console.log(`${key}: [File] ${value.name} (${formatFileSize(value.size)})`);
      } else {
        console.log(`${key}: ${value}`);
      }
    }
  }

  build(): FormData {
    return this.formData;
  }
}

// 使用例
async function uploadWithMetadata(
  files: File[],
  metadata: { userId: string; tags: string[] }
) {
  const formData = new FormDataBuilder()
    .addFiles('files', files)
    .addField('userId', metadata.userId)
    .addJson('tags', metadata.tags)
    .addField('uploadedAt', new Date().toISOString())
    .build();

  const response = await fetch('/api/upload', {
    method: 'POST',
    body: formData,
    // 注意: Content-Type は設定しない
    // ブラウザが boundary パラメータ付きで自動設定する
  });

  return response.json();
}
```

### 2.3 Fetch API vs XMLHttpRequest の比較

ファイルアップロードにおける Fetch API と XMLHttpRequest の違いを理解することは重要である。

```typescript
// Fetch API によるアップロード
// メリット: モダンなPromiseベースAPI、シンプルなコード
// デメリット: アップロード進捗を標準ではサポートしない
async function uploadWithFetch(file: File): Promise<UploadResult> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/api/upload', {
    method: 'POST',
    body: formData,
    signal: AbortSignal.timeout(60000), // 60秒タイムアウト
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

// XMLHttpRequest によるアップロード
// メリット: アップロード進捗イベントをサポート
// デメリット: コールバックベース、やや冗長
function uploadWithXHR(
  file: File,
  options: {
    onProgress?: (percent: number) => void;
    onComplete?: (result: UploadResult) => void;
    onError?: (error: Error) => void;
    timeout?: number;
  }
): { abort: () => void } {
  const xhr = new XMLHttpRequest();
  const formData = new FormData();
  formData.append('file', file);

  // アップロード進捗
  xhr.upload.addEventListener('progress', (event) => {
    if (event.lengthComputable) {
      const percent = Math.round((event.loaded / event.total) * 100);
      options.onProgress?.(percent);
    }
  });

  // 完了
  xhr.addEventListener('load', () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      const result = JSON.parse(xhr.responseText);
      options.onComplete?.(result);
    } else {
      options.onError?.(new Error(`Upload failed: ${xhr.status}`));
    }
  });

  // エラー
  xhr.addEventListener('error', () => {
    options.onError?.(new Error('Network error during upload'));
  });

  // タイムアウト
  xhr.addEventListener('timeout', () => {
    options.onError?.(new Error('Upload timeout'));
  });

  xhr.timeout = options.timeout ?? 60000;
  xhr.open('POST', '/api/upload');
  xhr.send(formData);

  return { abort: () => xhr.abort() };
}

// Fetch API + ReadableStream でプログレスを取得する方法（ダウンロードのみ）
// 注意: Fetch API ではアップロードの進捗は取得できないが、
// レスポンスのダウンロード進捗は取得可能
async function fetchWithDownloadProgress(url: string): Promise<Blob> {
  const response = await fetch(url);
  const contentLength = Number(response.headers.get('content-length'));
  const reader = response.body!.getReader();
  const chunks: Uint8Array[] = [];
  let receivedLength = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    chunks.push(value);
    receivedLength += value.length;

    if (contentLength) {
      const percent = Math.round((receivedLength / contentLength) * 100);
      console.log(`Download progress: ${percent}%`);
    }
  }

  return new Blob(chunks);
}
```

| 機能 | Fetch API | XMLHttpRequest |
|------|-----------|---------------|
| Promise サポート | ネイティブ | 手動ラップが必要 |
| アップロード進捗 | 非サポート | `upload.onprogress` |
| ダウンロード進捗 | ReadableStream | `onprogress` |
| キャンセル | AbortController | `abort()` |
| タイムアウト | AbortSignal.timeout() | `timeout` プロパティ |
| ストリーミング | ReadableStream | 非サポート |
| Service Worker | 対応 | 制限あり |
| 構文 | 簡潔 | 冗長 |
| ブラウザ互換性 | IE非対応 | すべて対応 |

---

## 3. ドラッグ&ドロップアップロード

### 3.1 HTML5 Drag and Drop API の基礎

ドラッグ&ドロップは、ファイルアップロードのUXを大幅に改善する機能である。HTML5のDrag and Drop APIの仕組みを理解することが重要だ。

```typescript
// 素のHTML5 Drag and Drop APIでの実装
function createDropZone(element: HTMLElement) {
  // ドラッグイベントの処理
  // 重要: dragover と dragenter で preventDefault() を呼ぶ必要がある
  // これをしないと drop イベントが発火しない

  let dragCounter = 0; // ネストされた子要素での dragenter/dragleave 対策

  element.addEventListener('dragenter', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter++;
    element.classList.add('drag-active');
  });

  element.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter--;
    if (dragCounter === 0) {
      element.classList.remove('drag-active');
    }
  });

  element.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    // dropEffect を設定してカーソルを変更
    if (e.dataTransfer) {
      e.dataTransfer.dropEffect = 'copy';
    }
  });

  element.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter = 0;
    element.classList.remove('drag-active');

    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      handleFiles(Array.from(files));
    }
  });

  // ウィンドウ全体でのドラッグ防止（ブラウザのデフォルト動作を抑止）
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => e.preventDefault());
}

function handleFiles(files: File[]) {
  files.forEach(file => {
    console.log(`Dropped: ${file.name} (${formatFileSize(file.size)})`);
  });
}
```

### 3.2 react-dropzone を使った高度な実装

```typescript
import { useDropzone, FileRejection, DropEvent } from 'react-dropzone';
import { useState, useCallback, useMemo, useEffect } from 'react';

// アップロードファイルの型定義
interface UploadFile {
  id: string;
  file: File;
  preview: string;
  status: 'pending' | 'uploading' | 'success' | 'error';
  progress: number;
  error?: string;
  url?: string;
}

// Dropzone の設定型
interface DropzoneConfig {
  maxFiles: number;
  maxSize: number;
  acceptedTypes: Record<string, string[]>;
  onUpload: (files: File[]) => Promise<void>;
}

function AdvancedFileDropzone({ maxFiles, maxSize, acceptedTypes, onUpload }: DropzoneConfig) {
  const [uploadFiles, setUploadFiles] = useState<UploadFile[]>([]);

  // ファイルが追加されたときの処理
  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
      // 既存ファイル数 + 新規ファイル数がmaxFilesを超えないか確認
      const remainingSlots = maxFiles - uploadFiles.length;
      const filesToAdd = acceptedFiles.slice(0, remainingSlots);

      if (acceptedFiles.length > remainingSlots) {
        toast.warning(
          `最大${maxFiles}ファイルまでです。${acceptedFiles.length - remainingSlots}ファイルが除外されました。`
        );
      }

      // 承認されたファイルをステートに追加
      const newUploadFiles: UploadFile[] = filesToAdd.map(file => ({
        id: crypto.randomUUID(),
        file,
        preview: file.type.startsWith('image/')
          ? URL.createObjectURL(file)
          : '',
        status: 'pending' as const,
        progress: 0,
      }));

      setUploadFiles(prev => [...prev, ...newUploadFiles]);

      // 拒否されたファイルのエラー表示
      rejectedFiles.forEach(({ file, errors }) => {
        const messages = errors.map(e => {
          switch (e.code) {
            case 'file-too-large':
              return `${file.name}: ファイルサイズが大きすぎます（最大${formatFileSize(maxSize)}）`;
            case 'file-invalid-type':
              return `${file.name}: サポートされていないファイル形式です`;
            case 'too-many-files':
              return `ファイル数が上限を超えています`;
            default:
              return `${file.name}: ${e.message}`;
          }
        });
        messages.forEach(msg => toast.error(msg));
      });
    },
    [uploadFiles.length, maxFiles, maxSize]
  );

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    isDragAccept,
    isDragReject,
    isFocused,
    open,
  } = useDropzone({
    accept: acceptedTypes,
    maxSize,
    maxFiles: maxFiles - uploadFiles.length,
    onDrop,
    noClick: false,
    noKeyboard: false,
    preventDropOnDocument: true,
    // ドラッグ&ドロップのカスタムバリデーション
    validator: (file) => {
      // ファイル名に特殊文字が含まれていないか確認
      const invalidChars = /[<>:"/\\|?*\x00-\x1F]/;
      if (invalidChars.test(file.name)) {
        return {
          code: 'invalid-filename',
          message: 'ファイル名に使用できない文字が含まれています',
        };
      }
      return null;
    },
  });

  // ドロップゾーンのスタイル
  const dropzoneStyle = useMemo(() => {
    let className = 'border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ';
    if (isDragReject) {
      className += 'border-red-500 bg-red-50 ';
    } else if (isDragAccept) {
      className += 'border-green-500 bg-green-50 ';
    } else if (isDragActive) {
      className += 'border-blue-500 bg-blue-50 ';
    } else if (isFocused) {
      className += 'border-blue-400 bg-blue-25 ';
    } else {
      className += 'border-gray-300 hover:border-gray-400 ';
    }
    return className;
  }, [isDragActive, isDragAccept, isDragReject, isFocused]);

  // ファイルの削除
  const removeFile = useCallback((id: string) => {
    setUploadFiles(prev => {
      const file = prev.find(f => f.id === id);
      if (file?.preview) {
        URL.revokeObjectURL(file.preview);
      }
      return prev.filter(f => f.id !== id);
    });
  }, []);

  // すべてのファイルをアップロード
  const uploadAll = useCallback(async () => {
    const pendingFiles = uploadFiles.filter(f => f.status === 'pending');
    if (pendingFiles.length === 0) return;

    for (const uploadFile of pendingFiles) {
      setUploadFiles(prev =>
        prev.map(f =>
          f.id === uploadFile.id ? { ...f, status: 'uploading' as const } : f
        )
      );

      try {
        await onUpload([uploadFile.file]);
        setUploadFiles(prev =>
          prev.map(f =>
            f.id === uploadFile.id
              ? { ...f, status: 'success' as const, progress: 100 }
              : f
          )
        );
      } catch (error) {
        setUploadFiles(prev =>
          prev.map(f =>
            f.id === uploadFile.id
              ? {
                  ...f,
                  status: 'error' as const,
                  error: error instanceof Error ? error.message : 'アップロードに失敗しました',
                }
              : f
          )
        );
      }
    }
  }, [uploadFiles, onUpload]);

  // メモリリーク防止: コンポーネントのアンマウント時にObject URLを解放
  useEffect(() => {
    return () => {
      uploadFiles.forEach(file => {
        if (file.preview) {
          URL.revokeObjectURL(file.preview);
        }
      });
    };
  }, [uploadFiles]);

  return (
    <div className="space-y-4">
      {/* ドロップゾーン */}
      <div {...getRootProps()} className={dropzoneStyle}>
        <input {...getInputProps()} />
        <div className="space-y-3">
          {/* アイコン */}
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>

          {isDragReject ? (
            <p className="text-red-600 font-medium">
              このファイル形式はサポートされていません
            </p>
          ) : isDragActive ? (
            <p className="text-blue-600 font-medium">ここにドロップしてください</p>
          ) : (
            <>
              <p className="text-gray-600">
                ファイルをドラッグ&ドロップ、または
                <button
                  type="button"
                  onClick={open}
                  className="text-blue-600 hover:text-blue-700 font-medium mx-1"
                >
                  クリックして選択
                </button>
              </p>
              <p className="text-xs text-gray-500">
                JPEG, PNG, WebP（最大{formatFileSize(maxSize)}、最大{maxFiles}ファイル）
              </p>
            </>
          )}
        </div>
      </div>

      {/* ファイルリスト */}
      {uploadFiles.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium">
              ファイル一覧 ({uploadFiles.length}/{maxFiles})
            </h4>
            <button
              type="button"
              onClick={uploadAll}
              disabled={!uploadFiles.some(f => f.status === 'pending')}
              className="text-sm text-blue-600 hover:text-blue-700 disabled:text-gray-400"
            >
              すべてアップロード
            </button>
          </div>

          {uploadFiles.map(uploadFile => (
            <FileListItem
              key={uploadFile.id}
              file={uploadFile}
              onRemove={() => removeFile(uploadFile.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ファイルリストの各アイテム
function FileListItem({
  file,
  onRemove,
}: {
  file: UploadFile;
  onRemove: () => void;
}) {
  return (
    <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
      {/* プレビュー */}
      {file.preview ? (
        <img
          src={file.preview}
          alt={file.file.name}
          className="w-12 h-12 object-cover rounded"
        />
      ) : (
        <div className="w-12 h-12 bg-gray-200 rounded flex items-center justify-center">
          <span className="text-xs text-gray-500">FILE</span>
        </div>
      )}

      {/* ファイル情報 */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{file.file.name}</p>
        <p className="text-xs text-gray-500">{formatFileSize(file.file.size)}</p>

        {/* プログレスバー */}
        {file.status === 'uploading' && (
          <div className="mt-1 w-full bg-gray-200 rounded-full h-1.5">
            <div
              className="bg-blue-500 h-1.5 rounded-full transition-all"
              style={{ width: `${file.progress}%` }}
            />
          </div>
        )}

        {/* エラーメッセージ */}
        {file.status === 'error' && (
          <p className="text-xs text-red-600 mt-1">{file.error}</p>
        )}
      </div>

      {/* ステータスアイコン / 削除ボタン */}
      <div className="flex-shrink-0">
        {file.status === 'success' ? (
          <span className="text-green-500">完了</span>
        ) : file.status === 'uploading' ? (
          <span className="text-blue-500">アップロード中...</span>
        ) : (
          <button
            type="button"
            onClick={onRemove}
            className="text-gray-400 hover:text-red-500 transition-colors"
            aria-label="ファイルを削除"
          >
            &times;
          </button>
        )}
      </div>
    </div>
  );
}
```

### 3.3 ページ全体のドラッグ&ドロップオーバーレイ

ページ全体にドラッグ&ドロップ領域を広げるパターンは、多くのアプリケーションで採用されている。

```typescript
import { useState, useEffect, useCallback, useRef } from 'react';

// ページ全体のドラッグ&ドロップを検知するカスタムフック
function usePageDragDrop(onFilesDropped: (files: File[]) => void) {
  const [isDragging, setIsDragging] = useState(false);
  const dragCounterRef = useRef(0);

  const handleDragEnter = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current++;

    // ファイルがドラッグされているか確認
    if (e.dataTransfer?.types.includes('Files')) {
      setIsDragging(true);
    }
  }, []);

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current--;

    if (dragCounterRef.current === 0) {
      setIsDragging(false);
    }
  }, []);

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounterRef.current = 0;
      setIsDragging(false);

      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        onFilesDropped(Array.from(files));
      }
    },
    [onFilesDropped]
  );

  useEffect(() => {
    document.addEventListener('dragenter', handleDragEnter);
    document.addEventListener('dragleave', handleDragLeave);
    document.addEventListener('dragover', handleDragOver);
    document.addEventListener('drop', handleDrop);

    return () => {
      document.removeEventListener('dragenter', handleDragEnter);
      document.removeEventListener('dragleave', handleDragLeave);
      document.removeEventListener('dragover', handleDragOver);
      document.removeEventListener('drop', handleDrop);
    };
  }, [handleDragEnter, handleDragLeave, handleDragOver, handleDrop]);

  return isDragging;
}

// ドラッグ&ドロップオーバーレイコンポーネント
function DragDropOverlay({ onFilesDropped }: { onFilesDropped: (files: File[]) => void }) {
  const isDragging = usePageDragDrop(onFilesDropped);

  if (!isDragging) return null;

  return (
    <div className="fixed inset-0 z-50 bg-blue-500/20 backdrop-blur-sm flex items-center justify-center">
      <div className="bg-white rounded-2xl p-12 shadow-2xl border-2 border-dashed border-blue-500">
        <div className="text-center space-y-4">
          <svg
            className="mx-auto h-16 w-16 text-blue-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
          <p className="text-xl font-semibold text-blue-700">
            ファイルをここにドロップしてアップロード
          </p>
          <p className="text-sm text-blue-500">
            対応形式: JPEG, PNG, WebP, PDF
          </p>
        </div>
      </div>
    </div>
  );
}
```

---

## 4. プログレス付きアップロード

### 4.1 XMLHttpRequest でのプログレス取得

プログレスバーはユーザーにアップロードの進行状況を視覚的に伝える重要なUI要素である。XMLHttpRequest の `upload.onprogress` イベントを使用して実装する。

```typescript
import { useState, useRef, useCallback } from 'react';

// アップロード状態の型定義
interface UploadState {
  status: 'idle' | 'uploading' | 'processing' | 'success' | 'error' | 'cancelled';
  progress: number;      // 0-100
  loaded: number;        // アップロード済みバイト数
  total: number;         // 合計バイト数
  speed: number;         // バイト/秒
  remainingTime: number; // 残り秒数
  error?: string;
  result?: UploadResult;
}

interface UploadResult {
  url: string;
  key: string;
  size: number;
  mimeType: string;
}

// 高度なファイルアップロードフック
function useFileUpload(uploadUrl: string) {
  const [state, setState] = useState<UploadState>({
    status: 'idle',
    progress: 0,
    loaded: 0,
    total: 0,
    speed: 0,
    remainingTime: 0,
  });

  const xhrRef = useRef<XMLHttpRequest | null>(null);
  const startTimeRef = useRef<number>(0);
  const lastLoadedRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);

  const upload = useCallback(
    async (file: File): Promise<UploadResult> => {
      return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhrRef.current = xhr;
        startTimeRef.current = Date.now();
        lastLoadedRef.current = 0;
        lastTimeRef.current = Date.now();

        // アップロード進捗イベント
        xhr.upload.addEventListener('progress', (event) => {
          if (!event.lengthComputable) return;

          const now = Date.now();
          const elapsedSinceLastUpdate = (now - lastTimeRef.current) / 1000;

          // 速度計算（移動平均）
          let speed = 0;
          if (elapsedSinceLastUpdate > 0) {
            const bytesInPeriod = event.loaded - lastLoadedRef.current;
            speed = bytesInPeriod / elapsedSinceLastUpdate;
          }

          // 残り時間計算
          const remaining = event.total - event.loaded;
          const remainingTime = speed > 0 ? remaining / speed : 0;

          setState({
            status: 'uploading',
            progress: Math.round((event.loaded / event.total) * 100),
            loaded: event.loaded,
            total: event.total,
            speed,
            remainingTime,
          });

          lastLoadedRef.current = event.loaded;
          lastTimeRef.current = now;
        });

        // アップロード完了後のサーバー処理待ち
        xhr.upload.addEventListener('load', () => {
          setState(prev => ({
            ...prev,
            status: 'processing',
            progress: 100,
          }));
        });

        // レスポンス受信完了
        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            const result = JSON.parse(xhr.responseText) as UploadResult;
            setState(prev => ({
              ...prev,
              status: 'success',
              result,
            }));
            resolve(result);
          } else {
            const errorMessage = `アップロード失敗: ${xhr.status} ${xhr.statusText}`;
            setState(prev => ({
              ...prev,
              status: 'error',
              error: errorMessage,
            }));
            reject(new Error(errorMessage));
          }
        });

        // ネットワークエラー
        xhr.addEventListener('error', () => {
          const errorMessage = 'ネットワークエラーが発生しました';
          setState(prev => ({
            ...prev,
            status: 'error',
            error: errorMessage,
          }));
          reject(new Error(errorMessage));
        });

        // キャンセル
        xhr.addEventListener('abort', () => {
          setState(prev => ({
            ...prev,
            status: 'cancelled',
          }));
          reject(new Error('アップロードがキャンセルされました'));
        });

        // タイムアウト
        xhr.addEventListener('timeout', () => {
          const errorMessage = 'アップロードがタイムアウトしました';
          setState(prev => ({
            ...prev,
            status: 'error',
            error: errorMessage,
          }));
          reject(new Error(errorMessage));
        });

        // FormData を構築
        const formData = new FormData();
        formData.append('file', file);

        // リクエスト設定
        xhr.timeout = 5 * 60 * 1000; // 5分
        xhr.open('POST', uploadUrl);
        xhr.send(formData);
      });
    },
    [uploadUrl]
  );

  // キャンセル機能
  const cancel = useCallback(() => {
    if (xhrRef.current) {
      xhrRef.current.abort();
      xhrRef.current = null;
    }
  }, []);

  // リセット機能
  const reset = useCallback(() => {
    cancel();
    setState({
      status: 'idle',
      progress: 0,
      loaded: 0,
      total: 0,
      speed: 0,
      remainingTime: 0,
    });
  }, [cancel]);

  return { state, upload, cancel, reset };
}
```

### 4.2 高機能プログレス表示コンポーネント

```typescript
// 詳細なプログレス表示コンポーネント
function UploadProgressDisplay({ state }: { state: UploadState }) {
  if (state.status === 'idle') return null;

  const getStatusColor = () => {
    switch (state.status) {
      case 'uploading': return 'bg-blue-500';
      case 'processing': return 'bg-yellow-500';
      case 'success': return 'bg-green-500';
      case 'error': return 'bg-red-500';
      case 'cancelled': return 'bg-gray-500';
      default: return 'bg-gray-300';
    }
  };

  const getStatusMessage = () => {
    switch (state.status) {
      case 'uploading':
        return `アップロード中... ${state.progress}%`;
      case 'processing':
        return 'サーバーで処理中...';
      case 'success':
        return 'アップロード完了';
      case 'error':
        return state.error || 'エラーが発生しました';
      case 'cancelled':
        return 'キャンセルされました';
      default:
        return '';
    }
  };

  const formatSpeed = (bytesPerSecond: number): string => {
    if (bytesPerSecond === 0) return '計算中...';
    return `${formatFileSize(bytesPerSecond)}/s`;
  };

  const formatTime = (seconds: number): string => {
    if (seconds <= 0 || !isFinite(seconds)) return '計算中...';
    if (seconds < 60) return `残り ${Math.ceil(seconds)} 秒`;
    if (seconds < 3600) return `残り ${Math.ceil(seconds / 60)} 分`;
    return `残り ${Math.ceil(seconds / 3600)} 時間`;
  };

  return (
    <div className="space-y-2 p-4 bg-gray-50 rounded-lg">
      {/* ステータスメッセージ */}
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium">{getStatusMessage()}</span>
        {state.status === 'uploading' && (
          <span className="text-gray-500">
            {formatFileSize(state.loaded)} / {formatFileSize(state.total)}
          </span>
        )}
      </div>

      {/* プログレスバー */}
      <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
        <div
          className={`h-2.5 rounded-full transition-all duration-300 ${getStatusColor()}`}
          style={{ width: `${state.progress}%` }}
        />
      </div>

      {/* 詳細情報 */}
      {state.status === 'uploading' && (
        <div className="flex justify-between text-xs text-gray-500">
          <span>速度: {formatSpeed(state.speed)}</span>
          <span>{formatTime(state.remainingTime)}</span>
        </div>
      )}

      {/* 処理中アニメーション */}
      {state.status === 'processing' && (
        <div className="flex items-center gap-2 text-xs text-yellow-600">
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
            <circle
              cx="12" cy="12" r="10"
              stroke="currentColor" strokeWidth="4"
              fill="none" opacity="0.25"
            />
            <path
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
          <span>画像の最適化・リサイズを実行中...</span>
        </div>
      )}
    </div>
  );
}

// 円形プログレス表示
function CircularProgress({
  progress,
  size = 64,
  strokeWidth = 4,
}: {
  progress: number;
  size?: number;
  strokeWidth?: number;
}) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="transform -rotate-90">
        {/* 背景円 */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={strokeWidth}
        />
        {/* プログレス円 */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#3b82f6"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-300"
        />
      </svg>
      <span className="absolute text-xs font-semibold">{progress}%</span>
    </div>
  );
}
```

### 4.3 複数ファイルの並列・逐次アップロード

```typescript
// 複数ファイルを同時にアップロード（並列数制限あり）
async function uploadFilesWithConcurrency(
  files: File[],
  uploadFn: (file: File) => Promise<UploadResult>,
  options: {
    concurrency?: number;
    onProgress?: (completed: number, total: number) => void;
    onFileComplete?: (file: File, result: UploadResult) => void;
    onFileError?: (file: File, error: Error) => void;
  } = {}
): Promise<Map<string, UploadResult | Error>> {
  const { concurrency = 3, onProgress, onFileComplete, onFileError } = options;
  const results = new Map<string, UploadResult | Error>();
  let completedCount = 0;

  // セマフォ実装（同時アップロード数制限）
  const semaphore = {
    count: concurrency,
    queue: [] as (() => void)[],
    acquire(): Promise<void> {
      return new Promise(resolve => {
        if (this.count > 0) {
          this.count--;
          resolve();
        } else {
          this.queue.push(resolve);
        }
      });
    },
    release(): void {
      if (this.queue.length > 0) {
        const next = this.queue.shift()!;
        next();
      } else {
        this.count++;
      }
    },
  };

  const uploadWithSemaphore = async (file: File) => {
    await semaphore.acquire();
    try {
      const result = await uploadFn(file);
      results.set(file.name, result);
      onFileComplete?.(file, result);
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      results.set(file.name, err);
      onFileError?.(file, err);
    } finally {
      completedCount++;
      onProgress?.(completedCount, files.length);
      semaphore.release();
    }
  };

  await Promise.all(files.map(uploadWithSemaphore));
  return results;
}

// React で使う並列アップロードフック
function useMultiFileUpload(options: {
  uploadUrl: string;
  concurrency?: number;
}) {
  const [files, setFiles] = useState<Map<string, {
    file: File;
    status: 'pending' | 'uploading' | 'success' | 'error';
    progress: number;
    error?: string;
    result?: UploadResult;
  }>>(new Map());

  const [overallProgress, setOverallProgress] = useState(0);

  const addFiles = useCallback((newFiles: File[]) => {
    setFiles(prev => {
      const next = new Map(prev);
      newFiles.forEach(file => {
        next.set(file.name, {
          file,
          status: 'pending',
          progress: 0,
        });
      });
      return next;
    });
  }, []);

  const uploadAll = useCallback(async () => {
    const pendingFiles = Array.from(files.entries())
      .filter(([, f]) => f.status === 'pending')
      .map(([, f]) => f.file);

    if (pendingFiles.length === 0) return;

    await uploadFilesWithConcurrency(
      pendingFiles,
      async (file) => {
        setFiles(prev => {
          const next = new Map(prev);
          const entry = next.get(file.name);
          if (entry) {
            entry.status = 'uploading';
          }
          return next;
        });

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(options.uploadUrl, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
        return response.json();
      },
      {
        concurrency: options.concurrency ?? 3,
        onProgress: (completed, total) => {
          setOverallProgress(Math.round((completed / total) * 100));
        },
        onFileComplete: (file, result) => {
          setFiles(prev => {
            const next = new Map(prev);
            const entry = next.get(file.name);
            if (entry) {
              entry.status = 'success';
              entry.progress = 100;
              entry.result = result;
            }
            return next;
          });
        },
        onFileError: (file, error) => {
          setFiles(prev => {
            const next = new Map(prev);
            const entry = next.get(file.name);
            if (entry) {
              entry.status = 'error';
              entry.error = error.message;
            }
            return next;
          });
        },
      }
    );
  }, [files, options.uploadUrl, options.concurrency]);

  return { files, addFiles, uploadAll, overallProgress };
}
```

---

## 5. S3 直接アップロード（プリサインドURL）

### 5.1 アーキテクチャの概要

S3プリサインドURLを使用した直接アップロードは、サーバーの負荷を大幅に削減できるアーキテクチャである。ファイルがアプリケーションサーバーを経由せず、クライアントから直接S3にアップロードされるため、帯域幅の節約とスケーラビリティの向上を実現する。

```
S3直接アップロードのフロー:

  [クライアント]                [アプリサーバー]              [AWS S3]
      |                             |                          |
      |-- 1. URL生成リクエスト -->  |                          |
      |                             |-- 2. PutObject署名 -->   |
      |                             |<-- 3. プリサインドURL --  |
      |<-- 4. プリサインドURL ----  |                          |
      |                             |                          |
      |-- 5. ファイルを直接PUT -------------------------------->|
      |<-- 6. 200 OK -------------------------------------------|
      |                             |                          |
      |-- 7. アップロード完了通知 ->|                          |
      |                             |-- 8. メタデータ保存 -->  |
      |<-- 9. 完了レスポンス -------|                          |

メリット:
- サーバーの帯域幅を節約
- 大ファイルもタイムアウトなしで処理
- スケーラビリティの向上
- CDNとの統合が容易

デメリット:
- CORS設定が必要
- クライアントにAWSのエンドポイントが露出
- ファイルのサーバーサイド処理が即座にできない
```

### 5.2 サーバーサイド: プリサインドURL生成

```typescript
// Next.js App Router: app/api/upload/presign/route.ts
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@/lib/auth';
import { z } from 'zod';
import crypto from 'crypto';

// S3 クライアントの初期化
const s3Client = new S3Client({
  region: process.env.AWS_REGION || 'ap-northeast-1',
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});

// リクエストのバリデーションスキーマ
const presignRequestSchema = z.object({
  filename: z.string().min(1).max(255),
  contentType: z.string().regex(/^(image|video|audio|application)\//),
  fileSize: z.number().positive().max(100 * 1024 * 1024), // 最大100MB
});

// 許可するファイルタイプ
const ALLOWED_CONTENT_TYPES: Record<string, string[]> = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/webp': ['.webp'],
  'image/gif': ['.gif'],
  'application/pdf': ['.pdf'],
  'video/mp4': ['.mp4'],
};

// 最大ファイルサイズ（Content-Type別）
const MAX_FILE_SIZES: Record<string, number> = {
  'image/': 10 * 1024 * 1024,    // 画像: 10MB
  'video/': 100 * 1024 * 1024,   // 動画: 100MB
  'application/pdf': 50 * 1024 * 1024, // PDF: 50MB
};

export async function POST(request: NextRequest) {
  try {
    // 認証チェック
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json(
        { error: '認証が必要です' },
        { status: 401 }
      );
    }

    // リクエストボディの解析
    const body = await request.json();
    const validation = presignRequestSchema.safeParse(body);

    if (!validation.success) {
      return NextResponse.json(
        { error: 'バリデーションエラー', details: validation.error.flatten() },
        { status: 400 }
      );
    }

    const { filename, contentType, fileSize } = validation.data;

    // Content-Type の検証
    if (!ALLOWED_CONTENT_TYPES[contentType]) {
      return NextResponse.json(
        { error: `サポートされていないファイル形式です: ${contentType}` },
        { status: 400 }
      );
    }

    // ファイルサイズの検証（Content-Type別）
    const maxSize = Object.entries(MAX_FILE_SIZES).find(
      ([prefix]) => contentType.startsWith(prefix)
    )?.[1] ?? 10 * 1024 * 1024;

    if (fileSize > maxSize) {
      return NextResponse.json(
        { error: `ファイルサイズが上限（${formatFileSize(maxSize)}）を超えています` },
        { status: 400 }
      );
    }

    // 安全なファイル名の生成
    const sanitizedFilename = filename
      .replace(/[^a-zA-Z0-9._-]/g, '_')
      .substring(0, 100);
    const uniqueId = crypto.randomUUID();
    const date = new Date().toISOString().split('T')[0]; // YYYY-MM-DD
    const key = `uploads/${session.user.id}/${date}/${uniqueId}/${sanitizedFilename}`;

    // プリサインドURLの生成
    const command = new PutObjectCommand({
      Bucket: process.env.S3_BUCKET!,
      Key: key,
      ContentType: contentType,
      ContentLength: fileSize,
      // メタデータ
      Metadata: {
        'uploaded-by': session.user.id,
        'original-filename': encodeURIComponent(filename),
        'upload-timestamp': new Date().toISOString(),
      },
      // サーバーサイド暗号化
      ServerSideEncryption: 'AES256',
    });

    const presignedUrl = await getSignedUrl(s3Client, command, {
      expiresIn: 900, // 15分で期限切れ
    });

    // アップロードレコードをDBに保存（ステータス: pending）
    // await db.upload.create({
    //   data: {
    //     key,
    //     userId: session.user.id,
    //     filename,
    //     contentType,
    //     fileSize,
    //     status: 'pending',
    //   },
    // });

    return NextResponse.json({
      presignedUrl,
      key,
      expiresAt: new Date(Date.now() + 900 * 1000).toISOString(),
    });
  } catch (error) {
    console.error('Presign URL generation error:', error);
    return NextResponse.json(
      { error: 'URLの生成に失敗しました' },
      { status: 500 }
    );
  }
}
```

### 5.3 S3 CORS 設定

S3バケットに適切なCORS設定を行わないと、ブラウザからの直接アップロードがブロックされる。

```json
// S3バケットのCORS設定
{
  "CORSRules": [
    {
      "AllowedOrigins": [
        "https://yourdomain.com",
        "https://staging.yourdomain.com"
      ],
      "AllowedMethods": ["PUT", "POST", "GET", "HEAD"],
      "AllowedHeaders": [
        "Content-Type",
        "Content-Length",
        "x-amz-meta-*",
        "x-amz-server-side-encryption"
      ],
      "ExposeHeaders": ["ETag", "x-amz-request-id"],
      "MaxAgeSeconds": 3600
    }
  ]
}
```

```typescript
// AWS CDK での CORS 設定
import * as s3 from 'aws-cdk-lib/aws-s3';

const uploadBucket = new s3.Bucket(this, 'UploadBucket', {
  bucketName: 'my-app-uploads',
  cors: [
    {
      allowedOrigins: ['https://yourdomain.com'],
      allowedMethods: [
        s3.HttpMethods.PUT,
        s3.HttpMethods.POST,
        s3.HttpMethods.GET,
        s3.HttpMethods.HEAD,
      ],
      allowedHeaders: ['*'],
      exposedHeaders: ['ETag', 'x-amz-request-id'],
      maxAge: 3600,
    },
  ],
  encryption: s3.BucketEncryption.S3_MANAGED,
  blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
  lifecycleRules: [
    {
      // 未完了のマルチパートアップロードを7日後に削除
      abortIncompleteMultipartUploadAfter: Duration.days(7),
    },
    {
      // 一時アップロードフォルダは30日後に削除
      prefix: 'uploads/temp/',
      expiration: Duration.days(30),
    },
  ],
});
```

### 5.4 クライアントサイド: S3直接アップロード実装

```typescript
// S3 直接アップロードのクライアントサイド実装
interface S3UploadOptions {
  file: File;
  onProgress?: (percent: number) => void;
  onComplete?: (url: string) => void;
  onError?: (error: Error) => void;
  signal?: AbortSignal;
}

async function uploadToS3({
  file,
  onProgress,
  onComplete,
  onError,
  signal,
}: S3UploadOptions): Promise<string> {
  try {
    // 1. プリサインドURL取得
    const presignResponse = await fetch('/api/upload/presign', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filename: file.name,
        contentType: file.type,
        fileSize: file.size,
      }),
      signal,
    });

    if (!presignResponse.ok) {
      const error = await presignResponse.json();
      throw new Error(error.error || 'プリサインドURLの取得に失敗しました');
    }

    const { presignedUrl, key } = await presignResponse.json();

    // 2. S3に直接アップロード（XMLHttpRequestでプログレス取得）
    await new Promise<void>((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      // AbortSignal の処理
      if (signal) {
        signal.addEventListener('abort', () => {
          xhr.abort();
          reject(new Error('アップロードがキャンセルされました'));
        });
      }

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const percent = Math.round((event.loaded / event.total) * 100);
          onProgress?.(percent);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve();
        } else {
          reject(new Error(`S3アップロード失敗: ${xhr.status}`));
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('S3アップロード中にネットワークエラーが発生しました'));
      });

      xhr.open('PUT', presignedUrl);
      xhr.setRequestHeader('Content-Type', file.type);
      xhr.send(file);
    });

    // 3. アップロード完了をサーバーに通知
    const confirmResponse = await fetch('/api/upload/confirm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key }),
      signal,
    });

    if (!confirmResponse.ok) {
      throw new Error('アップロード完了の通知に失敗しました');
    }

    const { url } = await confirmResponse.json();
    onComplete?.(url);
    return url;
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error));
    onError?.(err);
    throw err;
  }
}

// React フック
function useS3Upload() {
  const [state, setState] = useState<{
    status: 'idle' | 'getting-url' | 'uploading' | 'confirming' | 'success' | 'error';
    progress: number;
    error?: string;
    url?: string;
  }>({
    status: 'idle',
    progress: 0,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const upload = useCallback(async (file: File) => {
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setState({ status: 'getting-url', progress: 0 });

    try {
      const url = await uploadToS3({
        file,
        signal: controller.signal,
        onProgress: (percent) => {
          setState(prev => ({ ...prev, status: 'uploading', progress: percent }));
        },
      });

      setState({ status: 'success', progress: 100, url });
      return url;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'アップロードに失敗しました';
      setState({ status: 'error', progress: 0, error: message });
      throw error;
    }
  }, []);

  const cancel = useCallback(() => {
    abortControllerRef.current?.abort();
    setState({ status: 'idle', progress: 0 });
  }, []);

  return { ...state, upload, cancel };
}
```

---

## 6. 画像最適化とクライアントサイド処理

### 6.1 Canvas API による画像リサイズ

大きな画像をアップロード前にクライアントサイドでリサイズすることで、アップロード時間とサーバーの負荷を削減できる。Canvas API を使用した画像リサイズの完全な実装を示す。

```typescript
// 画像リサイズの設定
interface ImageResizeOptions {
  maxWidth: number;
  maxHeight: number;
  quality: number;         // 0-1（JPEG/WebP の品質）
  outputFormat: 'image/jpeg' | 'image/png' | 'image/webp';
  maintainAspectRatio: boolean;
  backgroundColor?: string; // PNG透過の場合の背景色
}

const DEFAULT_RESIZE_OPTIONS: ImageResizeOptions = {
  maxWidth: 1920,
  maxHeight: 1080,
  quality: 0.85,
  outputFormat: 'image/jpeg',
  maintainAspectRatio: true,
};

// 画像リサイズ関数
async function resizeImage(
  file: File,
  options: Partial<ImageResizeOptions> = {}
): Promise<File> {
  const opts = { ...DEFAULT_RESIZE_OPTIONS, ...options };

  // 画像をロード
  const img = await loadImage(file);

  // 新しいサイズを計算
  const { width, height } = calculateDimensions(
    img.naturalWidth,
    img.naturalHeight,
    opts.maxWidth,
    opts.maxHeight,
    opts.maintainAspectRatio
  );

  // リサイズが不要な場合はそのまま返す
  if (width === img.naturalWidth && height === img.naturalHeight) {
    return file;
  }

  // Canvas でリサイズ
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;

  // 背景色の設定（PNG→JPEG変換時に透過部分を埋める）
  if (opts.backgroundColor) {
    ctx.fillStyle = opts.backgroundColor;
    ctx.fillRect(0, 0, width, height);
  }

  // 高品質リサイズのための設定
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';

  // 描画
  ctx.drawImage(img, 0, 0, width, height);

  // Blob に変換
  const blob = await canvasToBlob(canvas, opts.outputFormat, opts.quality);

  // File オブジェクトとして返す
  const extension = opts.outputFormat.split('/')[1];
  const newFilename = file.name.replace(/\.[^.]+$/, `.${extension}`);

  return new File([blob], newFilename, {
    type: opts.outputFormat,
    lastModified: Date.now(),
  });
}

// 画像のロード
function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(img.src);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(img.src);
      reject(new Error('画像の読み込みに失敗しました'));
    };
    img.src = URL.createObjectURL(file);
  });
}

// サイズの計算
function calculateDimensions(
  originalWidth: number,
  originalHeight: number,
  maxWidth: number,
  maxHeight: number,
  maintainAspectRatio: boolean
): { width: number; height: number } {
  if (!maintainAspectRatio) {
    return {
      width: Math.min(originalWidth, maxWidth),
      height: Math.min(originalHeight, maxHeight),
    };
  }

  let width = originalWidth;
  let height = originalHeight;

  // アスペクト比を維持してリサイズ
  if (width > maxWidth) {
    height = Math.round((height * maxWidth) / width);
    width = maxWidth;
  }

  if (height > maxHeight) {
    width = Math.round((width * maxHeight) / height);
    height = maxHeight;
  }

  return { width, height };
}

// Canvas を Blob に変換
function canvasToBlob(
  canvas: HTMLCanvasElement,
  type: string,
  quality: number
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Canvas to Blob 変換に失敗しました'));
        }
      },
      type,
      quality
    );
  });
}
```

### 6.2 EXIF データの処理

スマートフォンで撮影された写真には EXIF データが含まれており、画像の回転情報（Orientation）を正しく処理しないと、表示が意図しない方向になることがある。

```typescript
// EXIF Orientation の読み取り
async function getExifOrientation(file: File): Promise<number> {
  const buffer = await file.slice(0, 65536).arrayBuffer();
  const view = new DataView(buffer);

  // JPEG マーカーの確認
  if (view.getUint16(0) !== 0xFFD8) {
    return 1; // JPEG ではない
  }

  let offset = 2;
  while (offset < view.byteLength) {
    const marker = view.getUint16(offset);
    offset += 2;

    if (marker === 0xFFE1) {
      // APP1 (EXIF) マーカー
      const length = view.getUint16(offset);
      offset += 2;

      // "Exif\0\0" の確認
      if (view.getUint32(offset) !== 0x45786966) {
        return 1;
      }
      offset += 6;

      const tiffOffset = offset;
      const bigEndian = view.getUint16(offset) === 0x4D4D;
      offset += 2;

      // マジックナンバー 42 の確認
      const magic = bigEndian
        ? view.getUint16(offset)
        : view.getUint16(offset, true);
      if (magic !== 42) return 1;
      offset += 2;

      // IFD0 オフセット
      const ifdOffset = bigEndian
        ? view.getUint32(offset)
        : view.getUint32(offset, true);
      offset = tiffOffset + ifdOffset;

      // IFD エントリ数
      const entries = bigEndian
        ? view.getUint16(offset)
        : view.getUint16(offset, true);
      offset += 2;

      for (let i = 0; i < entries; i++) {
        const tag = bigEndian
          ? view.getUint16(offset)
          : view.getUint16(offset, true);

        if (tag === 0x0112) {
          // Orientation タグ
          return bigEndian
            ? view.getUint16(offset + 8)
            : view.getUint16(offset + 8, true);
        }

        offset += 12;
      }

      return 1;
    } else if ((marker & 0xFF00) === 0xFF00) {
      const length = view.getUint16(offset);
      offset += length;
    } else {
      break;
    }
  }

  return 1;
}

// EXIF Orientation に基づいて画像を修正
async function fixImageOrientation(file: File): Promise<File> {
  const orientation = await getExifOrientation(file);

  // Orientation が 1（正常）の場合は処理不要
  if (orientation <= 1) return file;

  const img = await loadImage(file);
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;

  // Orientation に基づいてキャンバスサイズと変換を設定
  switch (orientation) {
    case 2: // 水平反転
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.transform(-1, 0, 0, 1, img.naturalWidth, 0);
      break;
    case 3: // 180度回転
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.transform(-1, 0, 0, -1, img.naturalWidth, img.naturalHeight);
      break;
    case 4: // 垂直反転
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.transform(1, 0, 0, -1, 0, img.naturalHeight);
      break;
    case 5: // 90度時計回り回転 + 水平反転
      canvas.width = img.naturalHeight;
      canvas.height = img.naturalWidth;
      ctx.transform(0, 1, 1, 0, 0, 0);
      break;
    case 6: // 90度時計回り回転
      canvas.width = img.naturalHeight;
      canvas.height = img.naturalWidth;
      ctx.transform(0, 1, -1, 0, img.naturalHeight, 0);
      break;
    case 7: // 90度反時計回り回転 + 水平反転
      canvas.width = img.naturalHeight;
      canvas.height = img.naturalWidth;
      ctx.transform(0, -1, -1, 0, img.naturalHeight, img.naturalWidth);
      break;
    case 8: // 90度反時計回り回転
      canvas.width = img.naturalHeight;
      canvas.height = img.naturalWidth;
      ctx.transform(0, -1, 1, 0, 0, img.naturalWidth);
      break;
  }

  ctx.drawImage(img, 0, 0);
  const blob = await canvasToBlob(canvas, file.type, 0.92);
  return new File([blob], file.name, { type: file.type, lastModified: Date.now() });
}
```

### 6.3 画像プレビューコンポーネント

```typescript
import { useState, useEffect, useCallback } from 'react';

// 画像プレビューフック
function useImagePreview() {
  const [previews, setPreviews] = useState<Map<string, string>>(new Map());

  const generatePreview = useCallback(async (file: File): Promise<string> => {
    // 既にプレビューがある場合はキャッシュを返す
    const existingPreview = previews.get(file.name);
    if (existingPreview) return existingPreview;

    // 画像の場合はサムネイルを生成
    if (file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      setPreviews(prev => new Map(prev).set(file.name, url));
      return url;
    }

    // 画像以外はアイコンURLを返す
    return getFileTypeIcon(file.type);
  }, [previews]);

  const removePreview = useCallback((filename: string) => {
    setPreviews(prev => {
      const next = new Map(prev);
      const url = next.get(filename);
      if (url && url.startsWith('blob:')) {
        URL.revokeObjectURL(url);
      }
      next.delete(filename);
      return next;
    });
  }, []);

  // クリーンアップ
  useEffect(() => {
    return () => {
      previews.forEach(url => {
        if (url.startsWith('blob:')) {
          URL.revokeObjectURL(url);
        }
      });
    };
  }, [previews]);

  return { previews, generatePreview, removePreview };
}

// ファイルタイプごとのアイコン取得
function getFileTypeIcon(mimeType: string): string {
  const iconMap: Record<string, string> = {
    'application/pdf': '/icons/pdf.svg',
    'application/zip': '/icons/zip.svg',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '/icons/doc.svg',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '/icons/xls.svg',
    'text/plain': '/icons/txt.svg',
    'text/csv': '/icons/csv.svg',
    'video/mp4': '/icons/video.svg',
    'audio/mpeg': '/icons/audio.svg',
  };

  return iconMap[mimeType] || '/icons/file.svg';
}

// 画像プレビューコンポーネント（ライトボックス付き）
function ImagePreviewGallery({
  files,
  onRemove,
}: {
  files: { id: string; file: File; previewUrl: string }[];
  onRemove: (id: string) => void;
}) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  return (
    <>
      {/* サムネイルグリッド */}
      <div className="grid grid-cols-4 gap-3">
        {files.map((item, index) => (
          <div key={item.id} className="relative group">
            <button
              type="button"
              onClick={() => setSelectedIndex(index)}
              className="w-full aspect-square overflow-hidden rounded-lg border
                hover:ring-2 hover:ring-blue-500 transition-all"
            >
              <img
                src={item.previewUrl}
                alt={item.file.name}
                className="w-full h-full object-cover"
              />
            </button>
            {/* 削除ボタン */}
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onRemove(item.id);
              }}
              className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white
                rounded-full text-xs flex items-center justify-center
                opacity-0 group-hover:opacity-100 transition-opacity"
              aria-label="削除"
            >
              x
            </button>
            {/* ファイル名 */}
            <p className="mt-1 text-xs text-gray-500 truncate">{item.file.name}</p>
            <p className="text-xs text-gray-400">{formatFileSize(item.file.size)}</p>
          </div>
        ))}
      </div>

      {/* ライトボックス */}
      {selectedIndex !== null && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center"
          onClick={() => setSelectedIndex(null)}
        >
          <div className="relative max-w-4xl max-h-[90vh]">
            <img
              src={files[selectedIndex].previewUrl}
              alt={files[selectedIndex].file.name}
              className="max-w-full max-h-[90vh] object-contain"
            />
            {/* ナビゲーション */}
            {selectedIndex > 0 && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedIndex(selectedIndex - 1);
                }}
                className="absolute left-4 top-1/2 -translate-y-1/2 text-white text-3xl"
              >
                &lt;
              </button>
            )}
            {selectedIndex < files.length - 1 && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedIndex(selectedIndex + 1);
                }}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-white text-3xl"
              >
                &gt;
              </button>
            )}
            {/* 閉じるボタン */}
            <button
              type="button"
              onClick={() => setSelectedIndex(null)}
              className="absolute top-4 right-4 text-white text-2xl"
            >
              x
            </button>
          </div>
        </div>
      )}
    </>
  );
}
```

### 6.4 WebP/AVIF 変換

モダンなフォーマットへの変換を行うことで、ファイルサイズを大幅に削減できる。

```typescript
// 画像フォーマット変換ユーティリティ
class ImageConverter {
  // WebP に変換
  static async toWebP(file: File, quality = 0.85): Promise<File> {
    return this.convert(file, 'image/webp', quality, '.webp');
  }

  // JPEG に変換
  static async toJPEG(file: File, quality = 0.9): Promise<File> {
    return this.convert(file, 'image/jpeg', quality, '.jpg');
  }

  // PNG に変換（無損失）
  static async toPNG(file: File): Promise<File> {
    return this.convert(file, 'image/png', 1, '.png');
  }

  // 汎用変換メソッド
  private static async convert(
    file: File,
    outputType: string,
    quality: number,
    extension: string
  ): Promise<File> {
    const img = await loadImage(file);
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0);

    const blob = await canvasToBlob(canvas, outputType, quality);
    const newFilename = file.name.replace(/\.[^.]+$/, extension);

    return new File([blob], newFilename, {
      type: outputType,
      lastModified: Date.now(),
    });
  }

  // ブラウザの WebP サポートを確認
  static async isWebPSupported(): Promise<boolean> {
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    return canvas.toDataURL('image/webp').startsWith('data:image/webp');
  }

  // ブラウザの AVIF サポートを確認
  static isAVIFSupported(): Promise<boolean> {
    return new Promise(resolve => {
      const img = new Image();
      img.onload = () => resolve(true);
      img.onerror = () => resolve(false);
      // 最小の AVIF 画像（1x1ピクセル）
      img.src = 'data:image/avif;base64,AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAADybWV0YQAAAAAAAAAoaGRscgAAAAAAAAAAcGljdAAAAAAAAAAAAAAAAGxpYmF2aWYAAAAADnBpdG0AAAAAAAEAAAAeaWxvYwAAAABEAAABAAEAAAABAAABGgAAAB0AAAAoaWluZgAAAAAAAQAAABppbmZlAgAAAAABAABhdjAxQ29sb3IAAAAAamlwcnAAAABLaXBjbwAAABRpc3BlAAAAAAAAAAIAAAACAAAAEHBpeGkAAAAAAwgICAAAAAxhdjFDgQ0MAAAAABNjb2xybmNseAACAAIAAYAAAAAXaXBtYQAAAAAAAAABAAEEAQKDBAAAACVtZGF0EgAKCBgANogQEAwgMg8f8D///8WfhwB8+ErZ42';
    });
  }

  // 最適なフォーマットで圧縮
  static async optimizeImage(
    file: File,
    options: {
      maxWidth?: number;
      maxHeight?: number;
      targetSizeKB?: number;
      preferredFormat?: 'webp' | 'jpeg' | 'auto';
    } = {}
  ): Promise<File> {
    const {
      maxWidth = 1920,
      maxHeight = 1080,
      targetSizeKB,
      preferredFormat = 'auto',
    } = options;

    // 1. リサイズ
    let processed = await resizeImage(file, { maxWidth, maxHeight });

    // 2. フォーマット選択
    let format: 'image/webp' | 'image/jpeg' = 'image/jpeg';
    if (preferredFormat === 'webp' || (preferredFormat === 'auto' && await this.isWebPSupported())) {
      format = 'image/webp';
    }

    // 3. 品質調整（ターゲットサイズ指定時）
    if (targetSizeKB) {
      let quality = 0.92;
      let result = await this.convert(processed, format, quality, format === 'image/webp' ? '.webp' : '.jpg');

      // バイナリサーチで最適な品質を見つける
      let minQuality = 0.1;
      let maxQuality = 0.95;

      for (let i = 0; i < 8; i++) {
        if (result.size > targetSizeKB * 1024) {
          maxQuality = quality;
        } else {
          minQuality = quality;
        }
        quality = (minQuality + maxQuality) / 2;
        result = await this.convert(processed, format, quality, format === 'image/webp' ? '.webp' : '.jpg');
      }

      return result;
    }

    // 4. デフォルト品質で変換
    return this.convert(processed, format, 0.85, format === 'image/webp' ? '.webp' : '.jpg');
  }
}
```

| フォーマット | 圧縮率 | 品質 | ブラウザ対応 | 用途 |
|-------------|--------|------|-------------|------|
| JPEG | 高 | やや劣化 | すべて | 写真・自然画像 |
| PNG | 低 | 無劣化 | すべて | アイコン・透過画像 |
| WebP | 非常に高 | 良好 | モダンブラウザ | 汎用（推奨） |
| AVIF | 最高 | 最良 | 限定的 | 次世代フォーマット |
| GIF | 低 | 制限的 | すべて | アニメーション |
| SVG | N/A | 完璧 | すべて | ベクター画像 |

---

## 7. チャンクアップロード（分割アップロード）

### 7.1 チャンクアップロードの概要

大容量ファイル（数百MB〜数GB）をアップロードする場合、ファイルを小さなチャンク（断片）に分割して順次アップロードする方式が有効である。ネットワーク障害時のレジューム、メモリ使用量の最適化、プログレス表示の精度向上などのメリットがある。

```
チャンクアップロードのフロー:

  [クライアント]                   [サーバー]                  [ストレージ]
      |                              |                           |
      |-- 1. アップロード開始 ------>|                           |
      |<-- 2. uploadId を返却 -------|                           |
      |                              |                           |
      |-- 3. チャンク1 送信 -------->|-- 一時保存 -------------->|
      |<-- 4. チャンク1 受理 --------|                           |
      |                              |                           |
      |-- 5. チャンク2 送信 -------->|-- 一時保存 -------------->|
      |<-- 6. チャンク2 受理 --------|                           |
      |                              |                           |
      |   ... 中断発生 ...           |                           |
      |                              |                           |
      |-- 7. レジューム要求 -------->|                           |
      |<-- 8. 完了チャンク情報 ------|                           |
      |                              |                           |
      |-- 9. チャンクN 送信 -------->|-- 一時保存 -------------->|
      |<-- 10. チャンクN 受理 -------|                           |
      |                              |                           |
      |-- 11. アップロード完了 ----->|-- チャンク結合 ---------->|
      |<-- 12. 最終URL返却 ----------|                           |
```

### 7.2 クライアントサイドのチャンクアップロード実装

```typescript
// チャンクアップロードの設定
interface ChunkUploadConfig {
  chunkSize: number;          // チャンクサイズ（バイト）
  maxRetries: number;         // チャンクごとの最大リトライ回数
  retryDelay: number;         // リトライ間隔（ミリ秒）
  concurrentChunks: number;   // 同時アップロードチャンク数
  apiEndpoint: string;        // APIエンドポイント
}

const DEFAULT_CHUNK_CONFIG: ChunkUploadConfig = {
  chunkSize: 5 * 1024 * 1024,  // 5MB
  maxRetries: 3,
  retryDelay: 1000,
  concurrentChunks: 3,
  apiEndpoint: '/api/upload/chunk',
};

// チャンク情報
interface ChunkInfo {
  index: number;
  start: number;
  end: number;
  size: number;
  blob: Blob;
  status: 'pending' | 'uploading' | 'success' | 'error';
  retries: number;
  etag?: string;
}

// チャンクアップロードクラス
class ChunkedUploader {
  private config: ChunkUploadConfig;
  private chunks: ChunkInfo[] = [];
  private uploadId: string | null = null;
  private abortController: AbortController | null = null;

  constructor(config: Partial<ChunkUploadConfig> = {}) {
    this.config = { ...DEFAULT_CHUNK_CONFIG, ...config };
  }

  // ファイルをチャンクに分割
  private createChunks(file: File): ChunkInfo[] {
    const chunks: ChunkInfo[] = [];
    const totalChunks = Math.ceil(file.size / this.config.chunkSize);

    for (let i = 0; i < totalChunks; i++) {
      const start = i * this.config.chunkSize;
      const end = Math.min(start + this.config.chunkSize, file.size);

      chunks.push({
        index: i,
        start,
        end,
        size: end - start,
        blob: file.slice(start, end),
        status: 'pending',
        retries: 0,
      });
    }

    return chunks;
  }

  // アップロード開始
  async upload(
    file: File,
    callbacks: {
      onProgress?: (progress: {
        percent: number;
        uploadedBytes: number;
        totalBytes: number;
        uploadedChunks: number;
        totalChunks: number;
        speed: number;
      }) => void;
      onComplete?: (result: { url: string; key: string }) => void;
      onError?: (error: Error) => void;
      onChunkComplete?: (chunkIndex: number) => void;
    } = {}
  ): Promise<{ url: string; key: string }> {
    this.abortController = new AbortController();
    this.chunks = this.createChunks(file);

    try {
      // 1. アップロードセッションの初期化
      const initResponse = await fetch(`${this.config.apiEndpoint}/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: file.name,
          fileSize: file.size,
          contentType: file.type,
          totalChunks: this.chunks.length,
          chunkSize: this.config.chunkSize,
        }),
        signal: this.abortController.signal,
      });

      if (!initResponse.ok) {
        throw new Error('アップロードの初期化に失敗しました');
      }

      const { uploadId } = await initResponse.json();
      this.uploadId = uploadId;

      // 2. チャンクを並列アップロード
      let uploadedBytes = 0;
      const startTime = Date.now();

      const uploadChunk = async (chunk: ChunkInfo): Promise<void> => {
        chunk.status = 'uploading';

        for (let retry = 0; retry <= this.config.maxRetries; retry++) {
          try {
            const formData = new FormData();
            formData.append('chunk', chunk.blob);
            formData.append('chunkIndex', String(chunk.index));
            formData.append('uploadId', uploadId);

            const response = await fetch(`${this.config.apiEndpoint}/chunk`, {
              method: 'POST',
              body: formData,
              signal: this.abortController!.signal,
            });

            if (!response.ok) {
              throw new Error(`チャンク ${chunk.index} のアップロードに失敗しました`);
            }

            const result = await response.json();
            chunk.etag = result.etag;
            chunk.status = 'success';
            uploadedBytes += chunk.size;

            // プログレスコールバック
            const elapsed = (Date.now() - startTime) / 1000;
            const speed = uploadedBytes / elapsed;
            const completedChunks = this.chunks.filter(c => c.status === 'success').length;

            callbacks.onProgress?.({
              percent: Math.round((uploadedBytes / file.size) * 100),
              uploadedBytes,
              totalBytes: file.size,
              uploadedChunks: completedChunks,
              totalChunks: this.chunks.length,
              speed,
            });

            callbacks.onChunkComplete?.(chunk.index);
            return;
          } catch (error) {
            chunk.retries++;
            if (retry < this.config.maxRetries) {
              // リトライ前に待機（指数バックオフ）
              await new Promise(resolve =>
                setTimeout(resolve, this.config.retryDelay * Math.pow(2, retry))
              );
            } else {
              chunk.status = 'error';
              throw error;
            }
          }
        }
      };

      // セマフォで並列数を制限
      await this.runWithConcurrency(
        this.chunks.map(chunk => () => uploadChunk(chunk)),
        this.config.concurrentChunks
      );

      // 3. アップロード完了を通知
      const completeResponse = await fetch(`${this.config.apiEndpoint}/complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          uploadId,
          parts: this.chunks.map(c => ({
            index: c.index,
            etag: c.etag,
          })),
        }),
        signal: this.abortController.signal,
      });

      if (!completeResponse.ok) {
        throw new Error('アップロードの完了処理に失敗しました');
      }

      const result = await completeResponse.json();
      callbacks.onComplete?.(result);
      return result;
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      callbacks.onError?.(err);
      throw err;
    }
  }

  // 並列実行制御
  private async runWithConcurrency(
    tasks: (() => Promise<void>)[],
    limit: number
  ): Promise<void> {
    const executing: Promise<void>[] = [];

    for (const task of tasks) {
      const p = task().then(() => {
        executing.splice(executing.indexOf(p), 1);
      });
      executing.push(p);

      if (executing.length >= limit) {
        await Promise.race(executing);
      }
    }

    await Promise.all(executing);
  }

  // レジューム（中断からの再開）
  async resume(
    file: File,
    uploadId: string,
    callbacks: {
      onProgress?: (progress: any) => void;
      onComplete?: (result: any) => void;
      onError?: (error: Error) => void;
    } = {}
  ): Promise<{ url: string; key: string }> {
    // 完了済みチャンクの情報を取得
    const statusResponse = await fetch(
      `${this.config.apiEndpoint}/status/${uploadId}`
    );

    if (!statusResponse.ok) {
      throw new Error('レジューム情報の取得に失敗しました');
    }

    const { completedChunks } = await statusResponse.json();
    const completedSet = new Set(completedChunks.map((c: any) => c.index));

    // 未完了のチャンクのみアップロード
    this.chunks = this.createChunks(file);
    this.chunks.forEach(chunk => {
      if (completedSet.has(chunk.index)) {
        chunk.status = 'success';
      }
    });

    this.uploadId = uploadId;
    // 残りのチャンクをアップロード（上記 upload メソッドと同様のロジック）
    // ... （省略: upload メソッドの後半と同じ処理）

    return { url: '', key: '' }; // プレースホルダー
  }

  // キャンセル
  cancel(): void {
    this.abortController?.abort();

    // サーバー側のアップロードセッションもキャンセル
    if (this.uploadId) {
      fetch(`${this.config.apiEndpoint}/abort`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uploadId: this.uploadId }),
      }).catch(() => {}); // エラーは無視
    }
  }

  // 完了チャンク数を取得
  getProgress(): { completed: number; total: number; percent: number } {
    const completed = this.chunks.filter(c => c.status === 'success').length;
    return {
      completed,
      total: this.chunks.length,
      percent: this.chunks.length > 0
        ? Math.round((completed / this.chunks.length) * 100)
        : 0,
    };
  }
}
```

### 7.3 S3 マルチパートアップロード（サーバーサイド）

```typescript
// Next.js API Routes: S3 マルチパートアップロード
// app/api/upload/chunk/init/route.ts
import {
  S3Client,
  CreateMultipartUploadCommand,
} from '@aws-sdk/client-s3';

const s3 = new S3Client({ region: 'ap-northeast-1' });

export async function POST(request: Request) {
  const { filename, contentType, fileSize } = await request.json();

  // ファイルサイズ上限チェック（1GB）
  if (fileSize > 1024 * 1024 * 1024) {
    return Response.json(
      { error: 'ファイルサイズは1GB以下にしてください' },
      { status: 400 }
    );
  }

  const key = `uploads/${crypto.randomUUID()}/${filename}`;

  const command = new CreateMultipartUploadCommand({
    Bucket: process.env.S3_BUCKET!,
    Key: key,
    ContentType: contentType,
    ServerSideEncryption: 'AES256',
  });

  const { UploadId } = await s3.send(command);

  return Response.json({ uploadId: UploadId, key });
}

// app/api/upload/chunk/presign/route.ts
import { UploadPartCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

export async function POST(request: Request) {
  const { uploadId, key, partNumber } = await request.json();

  const command = new UploadPartCommand({
    Bucket: process.env.S3_BUCKET!,
    Key: key,
    UploadId: uploadId,
    PartNumber: partNumber,
  });

  const presignedUrl = await getSignedUrl(s3, command, { expiresIn: 3600 });

  return Response.json({ presignedUrl });
}

// app/api/upload/chunk/complete/route.ts
import { CompleteMultipartUploadCommand } from '@aws-sdk/client-s3';

export async function POST(request: Request) {
  const { uploadId, key, parts } = await request.json();

  const command = new CompleteMultipartUploadCommand({
    Bucket: process.env.S3_BUCKET!,
    Key: key,
    UploadId: uploadId,
    MultipartUpload: {
      Parts: parts.map((part: any) => ({
        PartNumber: part.partNumber,
        ETag: part.etag,
      })),
    },
  });

  const result = await s3.send(command);

  return Response.json({
    url: result.Location,
    key,
    etag: result.ETag,
  });
}

// app/api/upload/chunk/abort/route.ts
import { AbortMultipartUploadCommand } from '@aws-sdk/client-s3';

export async function POST(request: Request) {
  const { uploadId, key } = await request.json();

  const command = new AbortMultipartUploadCommand({
    Bucket: process.env.S3_BUCKET!,
    Key: key,
    UploadId: uploadId,
  });

  await s3.send(command);

  return Response.json({ success: true });
}
```

### 7.4 tus プロトコルによるレジュームアブルアップロード

tus はオープンプロトコルとして標準化されたレジュームアブルアップロードの仕組みである。

```typescript
// tus-js-client を使用したレジュームアブルアップロード
import * as tus from 'tus-js-client';

function useTusUpload(endpoint: string) {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<'idle' | 'uploading' | 'paused' | 'success' | 'error'>('idle');
  const uploadRef = useRef<tus.Upload | null>(null);

  const upload = useCallback((file: File) => {
    const tusUpload = new tus.Upload(file, {
      endpoint,
      retryDelays: [0, 1000, 3000, 5000], // リトライ間隔
      chunkSize: 5 * 1024 * 1024, // 5MB チャンク
      metadata: {
        filename: file.name,
        filetype: file.type,
        filesize: String(file.size),
      },
      // アップロード前のフック
      onBeforeRequest: (req) => {
        // 認証ヘッダーの追加
        const token = getAuthToken();
        if (token) {
          req.setHeader('Authorization', `Bearer ${token}`);
        }
      },
      // プログレス
      onProgress: (bytesUploaded, bytesTotal) => {
        const percentage = Math.round((bytesUploaded / bytesTotal) * 100);
        setProgress(percentage);
      },
      // 成功
      onSuccess: () => {
        setStatus('success');
        console.log('Upload complete:', tusUpload.url);
      },
      // エラー
      onError: (error) => {
        setStatus('error');
        console.error('Upload error:', error);
      },
      // チャンク成功
      onChunkComplete: (chunkSize, bytesAccepted) => {
        console.log(`Chunk uploaded: ${formatFileSize(bytesAccepted)}`);
      },
    });

    uploadRef.current = tusUpload;
    setStatus('uploading');

    // 以前のアップロードがあれば再開を試みる
    tusUpload.findPreviousUploads().then((previousUploads) => {
      if (previousUploads.length > 0) {
        // 最新のアップロードから再開
        tusUpload.resumeFromPreviousUpload(previousUploads[0]);
      }
      tusUpload.start();
    });
  }, [endpoint]);

  const pause = useCallback(() => {
    uploadRef.current?.abort();
    setStatus('paused');
  }, []);

  const resume = useCallback(() => {
    uploadRef.current?.start();
    setStatus('uploading');
  }, []);

  const cancel = useCallback(() => {
    uploadRef.current?.abort();
    setStatus('idle');
    setProgress(0);
  }, []);

  return { progress, status, upload, pause, resume, cancel };
}
```

---

## 8. サーバーサイドのファイル処理

### 8.1 Next.js App Router でのファイル受信

```typescript
// app/api/upload/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import path from 'path';
import crypto from 'crypto';

// アップロード設定
const UPLOAD_CONFIG = {
  maxFileSize: 10 * 1024 * 1024, // 10MB
  allowedMimeTypes: [
    'image/jpeg',
    'image/png',
    'image/webp',
    'image/gif',
    'application/pdf',
  ],
  uploadDir: path.join(process.cwd(), 'uploads'),
};

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File | null;

    if (!file) {
      return NextResponse.json(
        { error: 'ファイルが送信されていません' },
        { status: 400 }
      );
    }

    // バリデーション
    const validationError = validateFile(file);
    if (validationError) {
      return NextResponse.json(
        { error: validationError },
        { status: 400 }
      );
    }

    // MIME タイプの実際の検証（マジックバイト確認）
    const buffer = Buffer.from(await file.arrayBuffer());
    const actualMimeType = detectMimeType(buffer);

    if (!actualMimeType || !UPLOAD_CONFIG.allowedMimeTypes.includes(actualMimeType)) {
      return NextResponse.json(
        { error: 'ファイル形式が不正です。実際のファイル内容と拡張子が一致しません。' },
        { status: 400 }
      );
    }

    // 安全なファイル名の生成
    const uniqueId = crypto.randomUUID();
    const ext = getExtensionFromMimeType(actualMimeType);
    const safeFilename = `${uniqueId}${ext}`;

    // ディレクトリの作成
    const dateDir = new Date().toISOString().split('T')[0];
    const uploadPath = path.join(UPLOAD_CONFIG.uploadDir, dateDir);
    await mkdir(uploadPath, { recursive: true });

    // ファイルの保存
    const filePath = path.join(uploadPath, safeFilename);
    await writeFile(filePath, buffer);

    // メタデータの保存（DB）
    // await db.file.create({
    //   data: {
    //     originalName: file.name,
    //     storedName: safeFilename,
    //     mimeType: actualMimeType,
    //     size: file.size,
    //     path: filePath,
    //     url: `/uploads/${dateDir}/${safeFilename}`,
    //   },
    // });

    return NextResponse.json({
      url: `/uploads/${dateDir}/${safeFilename}`,
      filename: safeFilename,
      originalName: file.name,
      size: file.size,
      mimeType: actualMimeType,
    });
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: 'アップロード処理中にエラーが発生しました' },
      { status: 500 }
    );
  }
}

// ファイルバリデーション
function validateFile(file: File): string | null {
  // サイズチェック
  if (file.size > UPLOAD_CONFIG.maxFileSize) {
    return `ファイルサイズが上限（${formatFileSize(UPLOAD_CONFIG.maxFileSize)}）を超えています`;
  }

  // 空ファイルチェック
  if (file.size === 0) {
    return '空のファイルはアップロードできません';
  }

  // ファイル名チェック（パストラバーサル防止）
  if (file.name.includes('..') || file.name.includes('/') || file.name.includes('\\')) {
    return '不正なファイル名です';
  }

  // ファイル名の長さチェック
  if (file.name.length > 255) {
    return 'ファイル名が長すぎます（最大255文字）';
  }

  return null;
}

// マジックバイトによるMIMEタイプ検出
function detectMimeType(buffer: Buffer): string | null {
  // JPEG: FF D8 FF
  if (buffer[0] === 0xFF && buffer[1] === 0xD8 && buffer[2] === 0xFF) {
    return 'image/jpeg';
  }

  // PNG: 89 50 4E 47 0D 0A 1A 0A
  if (
    buffer[0] === 0x89 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x4E &&
    buffer[3] === 0x47
  ) {
    return 'image/png';
  }

  // WebP: 52 49 46 46 ... 57 45 42 50
  if (
    buffer[0] === 0x52 &&
    buffer[1] === 0x49 &&
    buffer[2] === 0x46 &&
    buffer[3] === 0x46 &&
    buffer[8] === 0x57 &&
    buffer[9] === 0x45 &&
    buffer[10] === 0x42 &&
    buffer[11] === 0x50
  ) {
    return 'image/webp';
  }

  // GIF: 47 49 46 38
  if (
    buffer[0] === 0x47 &&
    buffer[1] === 0x49 &&
    buffer[2] === 0x46 &&
    buffer[3] === 0x38
  ) {
    return 'image/gif';
  }

  // PDF: 25 50 44 46
  if (
    buffer[0] === 0x25 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x44 &&
    buffer[3] === 0x46
  ) {
    return 'application/pdf';
  }

  return null;
}

// MIMEタイプから拡張子を取得
function getExtensionFromMimeType(mimeType: string): string {
  const extensionMap: Record<string, string> = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/webp': '.webp',
    'image/gif': '.gif',
    'application/pdf': '.pdf',
  };
  return extensionMap[mimeType] || '.bin';
}
```

### 8.2 Express.js + Multer でのファイル受信

```typescript
// Express.js + Multer による実装
import express from 'express';
import multer from 'multer';
import path from 'path';
import crypto from 'crypto';

const app = express();

// Multer の設定
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const dateDir = new Date().toISOString().split('T')[0];
    const uploadPath = path.join(__dirname, 'uploads', dateDir);
    // ディレクトリが存在しない場合は作成
    require('fs').mkdirSync(uploadPath, { recursive: true });
    cb(null, uploadPath);
  },
  filename: (req, file, cb) => {
    // 安全なファイル名の生成
    const uniqueId = crypto.randomUUID();
    const ext = path.extname(file.originalname).toLowerCase();
    cb(null, `${uniqueId}${ext}`);
  },
});

// ファイルフィルター
const fileFilter = (
  req: express.Request,
  file: Express.Multer.File,
  cb: multer.FileFilterCallback
) => {
  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'application/pdf'];

  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error(`サポートされていないファイル形式: ${file.mimetype}`));
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB
    files: 5, // 最大5ファイル
    fields: 10, // 最大10フィールド
    fieldSize: 1 * 1024 * 1024, // フィールドの最大サイズ: 1MB
  },
});

// 単一ファイルアップロード
app.post('/api/upload/single', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'ファイルが送信されていません' });
  }

  res.json({
    filename: req.file.filename,
    originalName: req.file.originalname,
    size: req.file.size,
    mimeType: req.file.mimetype,
    path: req.file.path,
  });
});

// 複数ファイルアップロード
app.post('/api/upload/multiple', upload.array('files', 5), (req, res) => {
  const files = req.files as Express.Multer.File[];

  if (!files || files.length === 0) {
    return res.status(400).json({ error: 'ファイルが送信されていません' });
  }

  res.json({
    files: files.map(file => ({
      filename: file.filename,
      originalName: file.originalname,
      size: file.size,
      mimeType: file.mimetype,
    })),
  });
});

// 複数フィールドのファイルアップロード
app.post(
  '/api/upload/fields',
  upload.fields([
    { name: 'avatar', maxCount: 1 },
    { name: 'gallery', maxCount: 10 },
    { name: 'document', maxCount: 3 },
  ]),
  (req, res) => {
    const files = req.files as { [fieldname: string]: Express.Multer.File[] };
    res.json({
      avatar: files['avatar']?.[0],
      gallery: files['gallery'],
      documents: files['document'],
    });
  }
);

// エラーハンドリングミドルウェア
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  if (err instanceof multer.MulterError) {
    switch (err.code) {
      case 'LIMIT_FILE_SIZE':
        return res.status(400).json({ error: 'ファイルサイズが上限を超えています' });
      case 'LIMIT_FILE_COUNT':
        return res.status(400).json({ error: 'ファイル数が上限を超えています' });
      case 'LIMIT_UNEXPECTED_FILE':
        return res.status(400).json({ error: '不正なフィールド名です' });
      default:
        return res.status(400).json({ error: err.message });
    }
  }

  console.error('Upload error:', err);
  res.status(500).json({ error: 'アップロード処理中にエラーが発生しました' });
});
```

### 8.3 Sharp による画像処理（サーバーサイド）

```typescript
// Sharp を使用したサーバーサイド画像処理
import sharp from 'sharp';
import path from 'path';

// 画像処理パイプライン
interface ImageProcessingOptions {
  sizes: { name: string; width: number; height: number }[];
  formats: ('jpeg' | 'webp' | 'avif')[];
  quality: Record<string, number>;
  watermark?: {
    text: string;
    position: 'center' | 'bottom-right';
    opacity: number;
  };
}

const DEFAULT_PROCESSING: ImageProcessingOptions = {
  sizes: [
    { name: 'thumbnail', width: 150, height: 150 },
    { name: 'small', width: 320, height: 320 },
    { name: 'medium', width: 640, height: 640 },
    { name: 'large', width: 1280, height: 1280 },
    { name: 'original', width: 3840, height: 3840 },
  ],
  formats: ['jpeg', 'webp'],
  quality: {
    jpeg: 85,
    webp: 80,
    avif: 65,
  },
};

class ImageProcessor {
  private options: ImageProcessingOptions;

  constructor(options: Partial<ImageProcessingOptions> = {}) {
    this.options = { ...DEFAULT_PROCESSING, ...options };
  }

  // 画像を複数サイズ・フォーマットで生成
  async processImage(
    inputPath: string,
    outputDir: string
  ): Promise<{
    variants: { name: string; format: string; path: string; size: number }[];
    metadata: sharp.Metadata;
  }> {
    const metadata = await sharp(inputPath).metadata();
    const variants: { name: string; format: string; path: string; size: number }[] = [];

    for (const sizeConfig of this.options.sizes) {
      for (const format of this.options.formats) {
        const outputFilename = `${sizeConfig.name}.${format}`;
        const outputPath = path.join(outputDir, outputFilename);

        let pipeline = sharp(inputPath)
          .rotate() // EXIF の回転情報を適用
          .resize(sizeConfig.width, sizeConfig.height, {
            fit: 'inside',
            withoutEnlargement: true,
          });

        // フォーマット変換
        switch (format) {
          case 'jpeg':
            pipeline = pipeline.jpeg({
              quality: this.options.quality.jpeg,
              progressive: true,
              mozjpeg: true,
            });
            break;
          case 'webp':
            pipeline = pipeline.webp({
              quality: this.options.quality.webp,
              effort: 4,
            });
            break;
          case 'avif':
            pipeline = pipeline.avif({
              quality: this.options.quality.avif,
              effort: 4,
            });
            break;
        }

        const info = await pipeline.toFile(outputPath);

        variants.push({
          name: sizeConfig.name,
          format,
          path: outputPath,
          size: info.size,
        });
      }
    }

    return { variants, metadata: metadata };
  }

  // 画像メタデータの取得
  async getMetadata(inputPath: string): Promise<{
    width: number;
    height: number;
    format: string;
    size: number;
    hasAlpha: boolean;
    orientation?: number;
  }> {
    const metadata = await sharp(inputPath).metadata();
    const stats = await sharp(inputPath).stats();

    return {
      width: metadata.width || 0,
      height: metadata.height || 0,
      format: metadata.format || 'unknown',
      size: metadata.size || 0,
      hasAlpha: metadata.hasAlpha || false,
      orientation: metadata.orientation,
    };
  }

  // EXIF データの除去（プライバシー保護）
  async stripExif(inputPath: string, outputPath: string): Promise<void> {
    await sharp(inputPath)
      .rotate() // 回転を適用してから
      .withMetadata({ orientation: undefined }) // EXIF を除去
      .toFile(outputPath);
  }
}
```

---

## 9. セキュリティ対策

ファイルアップロードはWebアプリケーションの中でも最も攻撃されやすい機能の一つである。適切なセキュリティ対策を怠ると、リモートコード実行（RCE）、ディレクトリトラバーサル、クロスサイトスクリプティング（XSS）、サービス拒否（DoS）などの深刻な脆弱性につながる。

### 9.1 脅威モデルと攻撃ベクトル

```
ファイルアップロードに対する主要な攻撃:

  1. マルウェアアップロード
     → ウイルス、トロイの木馬、ランサムウェアの配布
     → 対策: ウイルススキャン、ファイルタイプ制限

  2. Webシェル（リモートコード実行）
     → .php, .jsp, .aspx ファイルをアップロードしてサーバーで実行
     → 対策: 拡張子ホワイトリスト、実行権限の除去、別ドメインでの配信

  3. ディレクトリトラバーサル
     → ファイル名に "../" を含めて任意のパスに書き込み
     → 対策: ファイル名のサニタイズ、UUIDによるリネーム

  4. XSS（クロスサイトスクリプティング）
     → SVG ファイルや HTML ファイルに悪意のあるスクリプトを埋め込み
     → 対策: Content-Type の厳密な設定、CSP ヘッダー

  5. DoS（サービス拒否）
     → 巨大ファイルのアップロードでサーバーリソースを枯渇させる
     → 対策: ファイルサイズ制限、レート制限、タイムアウト

  6. MIME タイプ偽装
     → .jpg 拡張子の実行ファイルなど
     → 対策: マジックバイトによる実際のファイル内容検証

  7. ZIP爆弾（Zip Bomb）
     → 展開すると巨大になる圧縮ファイル
     → 対策: 展開後サイズの制限、再帰展開の禁止

  8. メタデータ攻撃
     → EXIF データに悪意のあるペイロードを埋め込み
     → 対策: メタデータの除去、再エンコード
```

### 9.2 サーバーサイドのセキュリティ実装

```typescript
// セキュアなファイルアップロードサービス
import crypto from 'crypto';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

class SecureFileUploadService {
  private readonly allowedMimeTypes: Set<string>;
  private readonly maxFileSize: number;
  private readonly uploadDir: string;

  constructor(config: {
    allowedMimeTypes: string[];
    maxFileSize: number;
    uploadDir: string;
  }) {
    this.allowedMimeTypes = new Set(config.allowedMimeTypes);
    this.maxFileSize = config.maxFileSize;
    this.uploadDir = config.uploadDir;
  }

  // 総合的なファイル検証
  async validateFile(buffer: Buffer, originalName: string): Promise<{
    isValid: boolean;
    errors: string[];
    detectedMimeType: string | null;
  }> {
    const errors: string[] = [];

    // 1. ファイルサイズチェック
    if (buffer.length === 0) {
      errors.push('ファイルが空です');
    }
    if (buffer.length > this.maxFileSize) {
      errors.push(`ファイルサイズが上限（${formatFileSize(this.maxFileSize)}）を超えています`);
    }

    // 2. ファイル名のサニタイズチェック
    const filenameErrors = this.validateFilename(originalName);
    errors.push(...filenameErrors);

    // 3. マジックバイトによるMIMEタイプ検証
    const detectedMimeType = this.detectMimeTypeFromMagicBytes(buffer);
    if (!detectedMimeType) {
      errors.push('ファイル形式を判別できませんでした');
    } else if (!this.allowedMimeTypes.has(detectedMimeType)) {
      errors.push(`許可されていないファイル形式です: ${detectedMimeType}`);
    }

    // 4. 拡張子とMIMEタイプの整合性チェック
    if (detectedMimeType) {
      const ext = path.extname(originalName).toLowerCase();
      const expectedExtensions = this.getExpectedExtensions(detectedMimeType);
      if (!expectedExtensions.includes(ext)) {
        errors.push(
          `ファイルの拡張子（${ext}）と実際の内容（${detectedMimeType}）が一致しません`
        );
      }
    }

    // 5. 危険なコンテンツのスキャン
    const dangerousContent = this.scanForDangerousContent(buffer);
    if (dangerousContent.length > 0) {
      errors.push(...dangerousContent);
    }

    return {
      isValid: errors.length === 0,
      errors,
      detectedMimeType,
    };
  }

  // ファイル名のバリデーション
  private validateFilename(filename: string): string[] {
    const errors: string[] = [];

    // パストラバーサル防止
    if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
      errors.push('ファイル名に不正な文字列が含まれています');
    }

    // NULL バイト防止
    if (filename.includes('\0')) {
      errors.push('ファイル名にNULLバイトが含まれています');
    }

    // 長さチェック
    if (filename.length > 255) {
      errors.push('ファイル名が長すぎます（最大255文字）');
    }

    // 危険な拡張子チェック
    const dangerousExtensions = [
      '.php', '.php3', '.php4', '.php5', '.phtml',
      '.jsp', '.jspx', '.jsw', '.jsv',
      '.asp', '.aspx', '.cer', '.csr',
      '.exe', '.dll', '.bat', '.cmd', '.com', '.msi',
      '.ps1', '.vbs', '.js', '.wsh', '.wsf',
      '.sh', '.bash', '.cgi', '.pl', '.py', '.rb',
      '.htaccess', '.htpasswd',
      '.svg', // XSS のリスクがある場合
    ];

    const ext = path.extname(filename).toLowerCase();
    if (dangerousExtensions.includes(ext)) {
      errors.push(`危険なファイル拡張子が検出されました: ${ext}`);
    }

    // 二重拡張子チェック（例: file.php.jpg）
    const parts = filename.split('.');
    if (parts.length > 2) {
      for (let i = 0; i < parts.length - 1; i++) {
        const checkExt = `.${parts[i]}`;
        if (dangerousExtensions.includes(checkExt.toLowerCase())) {
          errors.push(`二重拡張子が検出されました: ${filename}`);
          break;
        }
      }
    }

    return errors;
  }

  // マジックバイトによるMIMEタイプ検出
  private detectMimeTypeFromMagicBytes(buffer: Buffer): string | null {
    const signatures: Array<{
      bytes: number[];
      offset: number;
      mimeType: string;
    }> = [
      // JPEG
      { bytes: [0xFF, 0xD8, 0xFF], offset: 0, mimeType: 'image/jpeg' },
      // PNG
      { bytes: [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A], offset: 0, mimeType: 'image/png' },
      // GIF87a
      { bytes: [0x47, 0x49, 0x46, 0x38, 0x37, 0x61], offset: 0, mimeType: 'image/gif' },
      // GIF89a
      { bytes: [0x47, 0x49, 0x46, 0x38, 0x39, 0x61], offset: 0, mimeType: 'image/gif' },
      // PDF
      { bytes: [0x25, 0x50, 0x44, 0x46], offset: 0, mimeType: 'application/pdf' },
      // ZIP
      { bytes: [0x50, 0x4B, 0x03, 0x04], offset: 0, mimeType: 'application/zip' },
    ];

    // WebP 特殊チェック（RIFF...WEBP）
    if (
      buffer.length >= 12 &&
      buffer[0] === 0x52 && buffer[1] === 0x49 &&
      buffer[2] === 0x46 && buffer[3] === 0x46 &&
      buffer[8] === 0x57 && buffer[9] === 0x45 &&
      buffer[10] === 0x42 && buffer[11] === 0x50
    ) {
      return 'image/webp';
    }

    for (const sig of signatures) {
      if (buffer.length < sig.offset + sig.bytes.length) continue;

      let match = true;
      for (let i = 0; i < sig.bytes.length; i++) {
        if (buffer[sig.offset + i] !== sig.bytes[i]) {
          match = false;
          break;
        }
      }

      if (match) return sig.mimeType;
    }

    return null;
  }

  // 危険なコンテンツのスキャン
  private scanForDangerousContent(buffer: Buffer): string[] {
    const errors: string[] = [];
    const content = buffer.toString('utf-8', 0, Math.min(buffer.length, 10240));

    // スクリプトタグの検出
    if (/<script[\s>]/i.test(content)) {
      errors.push('ファイル内にスクリプトタグが検出されました');
    }

    // PHP タグの検出
    if (/<\?php/i.test(content) || /<\?=/i.test(content)) {
      errors.push('ファイル内にPHPコードが検出されました');
    }

    // イベントハンドラの検出（SVG/HTML）
    if (/\bon\w+\s*=/i.test(content)) {
      errors.push('ファイル内にイベントハンドラが検出されました');
    }

    return errors;
  }

  // 拡張子の期待値マッピング
  private getExpectedExtensions(mimeType: string): string[] {
    const map: Record<string, string[]> = {
      'image/jpeg': ['.jpg', '.jpeg', '.jpe'],
      'image/png': ['.png'],
      'image/webp': ['.webp'],
      'image/gif': ['.gif'],
      'application/pdf': ['.pdf'],
      'application/zip': ['.zip'],
    };
    return map[mimeType] || [];
  }

  // 安全なファイル名の生成
  generateSafeFilename(mimeType: string): string {
    const uuid = crypto.randomUUID();
    const ext = this.getExpectedExtensions(mimeType)[0] || '.bin';
    return `${uuid}${ext}`;
  }

  // ウイルススキャン（ClamAV連携）
  async scanForVirus(filePath: string): Promise<{
    isClean: boolean;
    threat?: string;
  }> {
    try {
      // ClamAV の clamscan コマンドを使用
      const { stdout } = await execFileAsync('clamscan', [
        '--no-summary',
        '--infected',
        filePath,
      ]);

      return { isClean: true };
    } catch (error: any) {
      // clamscan は感染ファイルを見つけた場合、終了コード 1 を返す
      if (error.code === 1) {
        return {
          isClean: false,
          threat: error.stdout?.trim() || 'マルウェアが検出されました',
        };
      }
      // その他のエラー（ClamAVが利用不可など）
      console.error('Virus scan error:', error);
      throw new Error('ウイルススキャンに失敗しました');
    }
  }
}
```

### 9.3 Content-Disposition とセキュリティヘッダー

```typescript
// ファイル配信時のセキュリティヘッダー設定
function getSecureFileHeaders(
  filename: string,
  mimeType: string,
  options: {
    inline?: boolean;
    noSniff?: boolean;
    csp?: string;
  } = {}
): Record<string, string> {
  const {
    inline = false,
    noSniff = true,
    csp = "default-src 'none'; img-src 'self'; style-src 'none'; script-src 'none'",
  } = options;

  // RFC 5987 に準拠したファイル名エンコーディング
  const encodedFilename = encodeURIComponent(filename)
    .replace(/['()]/g, escape)
    .replace(/\*/g, '%2A');

  const headers: Record<string, string> = {
    // Content-Disposition: ダウンロードかインライン表示かを制御
    'Content-Disposition': inline
      ? `inline; filename="${filename}"; filename*=UTF-8''${encodedFilename}`
      : `attachment; filename="${filename}"; filename*=UTF-8''${encodedFilename}`,

    // Content-Type: MIME タイプを明示的に設定
    'Content-Type': mimeType,

    // Cache-Control: プライベートなファイルの場合
    'Cache-Control': 'private, max-age=31536000, immutable',
  };

  // X-Content-Type-Options: MIME スニッフィング防止
  if (noSniff) {
    headers['X-Content-Type-Options'] = 'nosniff';
  }

  // Content-Security-Policy: インライン表示時のXSS防止
  if (inline && csp) {
    headers['Content-Security-Policy'] = csp;
  }

  // X-Frame-Options: クリックジャッキング防止
  headers['X-Frame-Options'] = 'DENY';

  return headers;
}

// Express でのファイル配信
app.get('/files/:id', async (req, res) => {
  const fileRecord = await db.file.findUnique({
    where: { id: req.params.id },
  });

  if (!fileRecord) {
    return res.status(404).json({ error: 'ファイルが見つかりません' });
  }

  // アクセス権限チェック
  if (!canAccessFile(req.user, fileRecord)) {
    return res.status(403).json({ error: 'アクセス権限がありません' });
  }

  const headers = getSecureFileHeaders(
    fileRecord.originalName,
    fileRecord.mimeType,
    {
      // 画像はインライン表示を許可
      inline: fileRecord.mimeType.startsWith('image/'),
    }
  );

  Object.entries(headers).forEach(([key, value]) => {
    res.setHeader(key, value);
  });

  res.sendFile(fileRecord.storedPath);
});
```

### 9.4 レート制限とリソース保護

```typescript
// アップロードのレート制限
import rateLimit from 'express-rate-limit';
import slowDown from 'express-slow-down';

// レート制限: 1ユーザーあたり1分に10回まで
const uploadRateLimit = rateLimit({
  windowMs: 60 * 1000,
  max: 10,
  message: {
    error: 'アップロードが多すぎます。しばらくしてからお試しください。',
  },
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: (req) => {
    return req.user?.id || req.ip;
  },
});

// スローダウン: リクエストが多い場合にレスポンスを遅延
const uploadSlowDown = slowDown({
  windowMs: 60 * 1000,
  delayAfter: 5,
  delayMs: (hits) => hits * 200,
});

// ストレージ使用量の制限
class StorageQuotaService {
  private readonly quotaPerUser: number; // バイト

  constructor(quotaGB: number = 5) {
    this.quotaPerUser = quotaGB * 1024 * 1024 * 1024;
  }

  async checkQuota(userId: string, additionalBytes: number): Promise<{
    allowed: boolean;
    currentUsage: number;
    quota: number;
    remainingBytes: number;
  }> {
    // ユーザーの現在のストレージ使用量を取得
    const currentUsage = await this.getUserStorageUsage(userId);
    const remainingBytes = this.quotaPerUser - currentUsage;

    return {
      allowed: currentUsage + additionalBytes <= this.quotaPerUser,
      currentUsage,
      quota: this.quotaPerUser,
      remainingBytes: Math.max(0, remainingBytes),
    };
  }

  private async getUserStorageUsage(userId: string): Promise<number> {
    // DB からユーザーの全ファイルサイズを合計
    // const result = await db.file.aggregate({
    //   _sum: { size: true },
    //   where: { userId },
    // });
    // return result._sum.size || 0;
    return 0; // プレースホルダー
  }
}

// アップロードミドルウェアへの統合
async function uploadMiddleware(req: any, res: any, next: any) {
  const userId = req.user?.id;
  if (!userId) {
    return res.status(401).json({ error: '認証が必要です' });
  }

  // ストレージ容量チェック
  const contentLength = parseInt(req.headers['content-length'] || '0', 10);
  const quotaService = new StorageQuotaService();
  const quotaCheck = await quotaService.checkQuota(userId, contentLength);

  if (!quotaCheck.allowed) {
    return res.status(413).json({
      error: 'ストレージ容量が不足しています',
      currentUsage: formatFileSize(quotaCheck.currentUsage),
      quota: formatFileSize(quotaCheck.quota),
      remaining: formatFileSize(quotaCheck.remainingBytes),
    });
  }

  next();
}
```

### 9.5 セキュリティチェックリスト

| チェック項目 | 重要度 | 対策 |
|-------------|--------|------|
| ファイルサイズ制限 | 必須 | サーバー側で制限を強制 |
| ファイルタイプ制限 | 必須 | ホワイトリスト方式で許可 |
| マジックバイト検証 | 必須 | 拡張子だけでなくバイナリ内容を検証 |
| ファイル名サニタイズ | 必須 | UUID でリネーム |
| パストラバーサル防止 | 必須 | `..` `/` `\` の除去 |
| NULLバイト除去 | 必須 | ファイル名から `\0` を除去 |
| 実行権限の除去 | 必須 | 保存ファイルに実行権限を付与しない |
| Content-Type 固定 | 必須 | サーバーが正しいMIMEタイプを設定 |
| X-Content-Type-Options | 必須 | `nosniff` ヘッダーを付与 |
| Content-Disposition | 推奨 | ダウンロード時に `attachment` を設定 |
| 別ドメインでの配信 | 推奨 | CDN/別ドメインからファイルを配信 |
| ウイルススキャン | 推奨 | ClamAV などでスキャン |
| レート制限 | 推奨 | ユーザーごとのアップロード頻度制限 |
| ストレージ容量制限 | 推奨 | ユーザーごとのストレージ上限 |
| 二重拡張子チェック | 推奨 | `file.php.jpg` などの検出 |
| EXIF データ除去 | 推奨 | プライバシー保護のため |
| CSP ヘッダー | 推奨 | インライン表示時のXSS防止 |
| 一時ファイルの自動削除 | 推奨 | TTL 付きで一時ファイルを管理 |

---

## 10. アップロードライブラリの比較

### 10.1 フロントエンドライブラリ比較

| ライブラリ | サイズ | 機能 | 特徴 |
|-----------|--------|------|------|
| react-dropzone | 8KB | D&D, バリデーション | シンプル、Reactフック |
| Uppy | 45KB+ | フル機能 | プラグインアーキテクチャ |
| Filepond | 36KB | 画像プレビュー, 変換 | 美しいUI |
| Dropzone.js | 43KB | D&D, プレビュー | jQuery/バニラ両対応 |
| Fine Uploader | 100KB+ | フル機能 | 企業向け |
| tus-js-client | 12KB | レジュームアブル | tusプロトコル |

### 10.2 バックエンドライブラリ比較

| ライブラリ | 言語/FW | 機能 | 用途 |
|-----------|---------|------|------|
| Multer | Node/Express | multipart解析 | Express向けファイルアップロード |
| Busboy | Node | ストリーミング解析 | 低レベルmultipart処理 |
| Formidable | Node | ファイル解析, 進捗 | 汎用Node.jsアップロード |
| Sharp | Node | 画像処理 | リサイズ/変換/最適化 |
| @aws-sdk/client-s3 | Node | S3操作 | AWS S3連携 |
| tusd | Go | tusサーバー | レジュームアブルアップロードサーバー |

### 10.3 Uppy を使った高機能アップロードUI

```typescript
import Uppy from '@uppy/core';
import Dashboard from '@uppy/dashboard';
import XHRUpload from '@uppy/xhr-upload';
import ImageEditor from '@uppy/image-editor';
import Webcam from '@uppy/webcam';
import '@uppy/core/dist/style.css';
import '@uppy/dashboard/dist/style.css';
import '@uppy/image-editor/dist/style.css';
import '@uppy/webcam/dist/style.css';

// Uppy の設定
function createUppy(options: {
  uploadEndpoint: string;
  maxFiles: number;
  maxFileSize: number;
  allowedFileTypes: string[];
  onComplete: (results: any[]) => void;
}) {
  const uppy = new Uppy({
    id: 'file-uploader',
    autoProceed: false,
    restrictions: {
      maxFileSize: options.maxFileSize,
      maxNumberOfFiles: options.maxFiles,
      allowedFileTypes: options.allowedFileTypes,
    },
    locale: {
      strings: {
        dropPasteFiles: 'ファイルをここにドラッグ&ドロップ、%{browseFiles}',
        browseFiles: 'ファイルを選択',
        uploadComplete: 'アップロード完了',
        xFilesSelected: {
          0: '%{smart_count}ファイルが選択されています',
          1: '%{smart_count}ファイルが選択されています',
        },
      },
    },
  })
    // ダッシュボードUI
    .use(Dashboard, {
      inline: true,
      target: '#uppy-dashboard',
      width: '100%',
      height: 400,
      showProgressDetails: true,
      proudlyDisplayPoweredByUppy: false,
      note: `最大${options.maxFiles}ファイル、各${formatFileSize(options.maxFileSize)}まで`,
    })
    // 画像エディター
    .use(ImageEditor, {
      target: Dashboard,
      quality: 0.85,
      cropperOptions: {
        viewMode: 1,
        background: false,
        autoCropArea: 1,
        responsive: true,
      },
    })
    // Webカメラ
    .use(Webcam, {
      target: Dashboard,
      modes: ['picture'],
      mirror: true,
    })
    // XHRアップロード
    .use(XHRUpload, {
      endpoint: options.uploadEndpoint,
      formData: true,
      fieldName: 'file',
      headers: {
        Authorization: `Bearer ${getAuthToken()}`,
      },
    });

  // イベントハンドリング
  uppy.on('complete', (result) => {
    const successful = result.successful?.map((file: any) => ({
      name: file.name,
      size: file.size,
      url: file.response?.body?.url,
    }));

    options.onComplete(successful || []);
  });

  uppy.on('upload-error', (file, error) => {
    console.error(`Upload error for ${file?.name}:`, error);
  });

  return uppy;
}

// React コンポーネントでの使用
function UppyUploader() {
  useEffect(() => {
    const uppy = createUppy({
      uploadEndpoint: '/api/upload',
      maxFiles: 10,
      maxFileSize: 10 * 1024 * 1024,
      allowedFileTypes: ['image/*', '.pdf'],
      onComplete: (results) => {
        console.log('Uploaded files:', results);
      },
    });

    return () => {
      uppy.close();
    };
  }, []);

  return <div id="uppy-dashboard" />;
}
```

---

## 11. トラブルシューティング

### 11.1 よくある問題と解決策

```
問題1: Content-Type ヘッダーの設定ミス

  症状: ファイルがサーバーで正しく受信されない
  原因: fetch() に Content-Type: multipart/form-data を手動設定している

  // NG: Content-Type を手動設定すると boundary が含まれない
  fetch('/api/upload', {
    method: 'POST',
    headers: {
      'Content-Type': 'multipart/form-data', // これが原因
    },
    body: formData,
  });

  // OK: Content-Type を設定しない（ブラウザが自動設定）
  fetch('/api/upload', {
    method: 'POST',
    body: formData,  // Content-Type は自動で設定される
  });

---

問題2: CORS エラー

  症状: S3 直接アップロード時に CORS エラーが発生する
  原因: S3 バケットの CORS 設定が不適切

  確認事項:
  - AllowedOrigins にフロントエンドのドメインが含まれているか
  - AllowedMethods に PUT が含まれているか
  - AllowedHeaders に Content-Type が含まれているか
  - ブラウザの開発者ツールで実際のエラーメッセージを確認

---

問題3: 大きなファイルでタイムアウト

  症状: 数MB以上のファイルでアップロードがタイムアウトする
  原因: サーバーのボディサイズ制限、プロキシのタイムアウト設定

  対策:
  - Next.js: route の config で bodySize を設定
    export const config = { api: { bodyParser: { sizeLimit: '50mb' } } };
  - Nginx: client_max_body_size を設定
    client_max_body_size 50M;
    proxy_read_timeout 300s;
  - チャンクアップロードの採用を検討

---

問題4: メモリリーク（Object URL）

  症状: プレビュー画像が増えるとメモリ使用量が増加する
  原因: URL.createObjectURL() で生成した URL を解放していない

  対策: コンポーネントのアンマウント時やファイル削除時に
  URL.revokeObjectURL(url) を呼び出す

---

問題5: iOS Safari での画像回転

  症状: iOS Safari で撮影した画像が回転して表示される
  原因: EXIF Orientation タグの処理が不足

  対策:
  - CSS の image-orientation: from-image; を使用
  - アップロード前に Canvas で回転を修正
  - サーバーサイドで Sharp の .rotate() を使用

---

問題6: ファイル入力のリセットがうまくいかない

  症状: 同じファイルを連続で選択しても onChange が発火しない
  原因: input 要素の value がクリアされていない

  対策:
  inputRef.current.value = ''; // 明示的にクリア
  または React のキー属性を変更して input を再マウント

---

問題7: Multer で req.file が undefined

  症状: Express + Multer でファイルが受信できない
  原因: フィールド名の不一致、Content-Type の問題

  確認事項:
  - FormData の append キーと Multer の .single('キー名') が一致しているか
  - Content-Type を手動設定していないか
  - body-parser が multipart を処理しようとしていないか
```

### 11.2 ブラウザ互換性の注意点

| 機能 | Chrome | Firefox | Safari | Edge | iOS Safari |
|------|--------|---------|--------|------|-----------|
| File API | 6+ | 3.6+ | 5.1+ | 12+ | 6+ |
| FileReader | 6+ | 3.6+ | 6+ | 12+ | 7+ |
| FormData | 7+ | 4+ | 5+ | 12+ | 5+ |
| Drag & Drop | 4+ | 3.5+ | 3.1+ | 12+ | 11+ |
| multiple 属性 | 5+ | 3.6+ | 4+ | 12+ | 5+ |
| accept 属性 | 8+ | 4+ | 6+ | 12+ | 6+ |
| capture 属性 | 25+ | - | 6+ | 12+ | 6+ |
| webkitdirectory | 11+ | 50+ | 11.1+ | 13+ | 14+ |
| Blob.slice | 13+ | 13+ | 7+ | 12+ | 7+ |
| URL.createObjectURL | 8+ | 4+ | 6+ | 12+ | 6+ |
| AbortController | 66+ | 57+ | 11.1+ | 16+ | 11.3+ |
| ReadableStream | 43+ | 65+ | 10.1+ | 14+ | 10.3+ |

### 11.3 パフォーマンス最適化のヒント

```typescript
// パフォーマンス最適化チェックリスト
const performanceChecklist = {
  // クライアントサイド
  client: [
    'Web Worker でのファイル処理（メインスレッドをブロックしない）',
    'サムネイル生成時は低解像度で行う（例: Canvas 150x150）',
    'Object URL は使い終わったら即座に revokeObjectURL する',
    'FileReader は大きなファイルには使わない（createObjectURL を使う）',
    'プレビュー画像は遅延ロードする',
    'リサイズは OffscreenCanvas を使用する（対応ブラウザのみ）',
  ],

  // ネットワーク
  network: [
    'クライアントサイドで画像を圧縮してからアップロードする',
    'チャンクアップロードで大ファイルを分割する',
    '同時アップロード数を制限する（3〜5接続）',
    'HTTP/2 を使用してヘッドオブラインブロッキングを回避',
    'CDN にアップロードエンドポイントを配置する',
    'S3 Transfer Acceleration を検討する',
  ],

  // サーバーサイド
  server: [
    'ストリーミング処理でメモリ使用量を抑える',
    '画像処理は非同期ジョブキューに委譲する',
    'シャーディングでファイル保存を分散する',
    'Lambda/Cloud Functions で画像処理をスケール',
    'CDN で静的ファイルをキャッシュする',
    'オンデマンド画像変換（imgix, Cloudinary）を検討する',
  ],
};

// Web Worker での画像リサイズ（メインスレッドをブロックしない）
// worker.ts
self.onmessage = async (event: MessageEvent) => {
  const { imageData, maxWidth, maxHeight, quality, format } = event.data;

  const canvas = new OffscreenCanvas(maxWidth, maxHeight);
  const ctx = canvas.getContext('2d')!;

  // ImageBitmap を使って描画
  const bitmap = await createImageBitmap(imageData);
  const { width, height } = calculateDimensions(
    bitmap.width,
    bitmap.height,
    maxWidth,
    maxHeight,
    true
  );

  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();

  const blob = await canvas.convertToBlob({ type: format, quality });
  self.postMessage({ blob }, [await blob.arrayBuffer()]);
};

// メインスレッドからの使用
function resizeInWorker(file: File, options: any): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const worker = new Worker('/workers/image-resize.js');

    worker.onmessage = (event) => {
      resolve(event.data.blob);
      worker.terminate();
    };

    worker.onerror = (error) => {
      reject(error);
      worker.terminate();
    };

    worker.postMessage({
      imageData: file,
      ...options,
    });
  });
}
```

---

## 12. アーキテクチャパターンと設計指針

### 12.1 ファイルアップロードのアーキテクチャ選択

```
方式1: サーバー経由アップロード
  クライアント → サーバー → ストレージ
  メリット: シンプルな実装、サーバーでの処理が容易
  デメリット: サーバーの帯域幅を消費、スケーラビリティに制限
  用途: 小〜中規模アプリ、ファイルサイズ数MB以下

方式2: S3 プリサインドURL直接アップロード
  クライアント → S3 (プリサインドURL)
  メリット: サーバー帯域を節約、S3のスケーラビリティを活用
  デメリット: CORS設定が必要、クライアントサイド処理が複雑
  用途: 中〜大規模アプリ、画像/動画アップロード

方式3: チャンクアップロード
  クライアント → サーバー (チャンク × N)
  メリット: レジューム可能、大ファイルに対応
  デメリット: 実装が複雑、サーバーサイドの状態管理が必要
  用途: 大ファイル（100MB+）、動画アップロード

方式4: CDN/エッジアップロード
  クライアント → CDN Edge → Origin Storage
  メリット: 地理的に近いエッジにアップロード、最高のパフォーマンス
  デメリット: コストが高い、CDN プロバイダーへの依存
  用途: グローバルサービス、リアルタイム処理
```

### 12.2 画像処理パイプラインの設計

```
推奨アーキテクチャ:

  [アップロード]
      ↓
  [オリジナル保存] → S3 (originals/)
      ↓
  [イベント発火] → S3 Event / SQS
      ↓
  [Lambda 画像処理]
      ├── リサイズ: thumbnail (150x150)
      ├── リサイズ: small (320x320)
      ├── リサイズ: medium (640x640)
      ├── リサイズ: large (1280x1280)
      ├── フォーマット変換: WebP
      ├── フォーマット変換: AVIF
      └── メタデータ除去: EXIF strip
      ↓
  [加工済み保存] → S3 (processed/)
      ↓
  [CDN配信] → CloudFront

代替: オンデマンド変換
  [リクエスト] → CDN → [Lambda@Edge / Cloudinary]
                             ↓
                    リアルタイム変換（キャッシュ）
```

### 12.3 データベースモデル設計

```typescript
// Prisma スキーマ: ファイル管理
// prisma/schema.prisma

/*
model File {
  id            String   @id @default(cuid())
  userId        String
  user          User     @relation(fields: [userId], references: [id])

  // オリジナルファイル情報
  originalName  String
  storedName    String   @unique
  mimeType      String
  size          Int      // バイト数
  hash          String   // SHA-256 ハッシュ（重複検出用）

  // ストレージ情報
  storageKey    String   @unique  // S3 キー
  bucket        String
  region        String

  // メタデータ
  width         Int?     // 画像の場合
  height        Int?     // 画像の場合
  duration      Float?   // 動画/音声の場合（秒）

  // ステータス
  status        FileStatus @default(PENDING)
  scanResult    ScanResult?
  scanDate      DateTime?

  // バリアント（リサイズ版等）
  variants      FileVariant[]
  parentId      String?
  parent        File?    @relation("FileVariants", fields: [parentId], references: [id])
  children      File[]   @relation("FileVariants")

  // タイムスタンプ
  createdAt     DateTime @default(now())
  updatedAt     DateTime @updatedAt
  deletedAt     DateTime?

  // インデックス
  @@index([userId])
  @@index([hash])
  @@index([status])
  @@index([createdAt])
}

model FileVariant {
  id         String @id @default(cuid())
  fileId     String
  file       File   @relation(fields: [fileId], references: [id])

  name       String  // "thumbnail", "small", "medium", "large"
  format     String  // "jpeg", "webp", "avif"
  width      Int
  height     Int
  size       Int
  storageKey String @unique
  url        String

  @@unique([fileId, name, format])
}

enum FileStatus {
  PENDING      // アップロード直後
  PROCESSING   // 画像処理中
  ACTIVE       // 利用可能
  QUARANTINED  // ウイルス検出
  DELETED      // 論理削除
}

enum ScanResult {
  CLEAN
  INFECTED
  ERROR
}
*/
```

---

## まとめ

### 実装パターン早見表

| パターン | 用途 | 難易度 | 推奨場面 |
|---------|------|--------|---------|
| `<input type="file">` + FormData | 基本的なアップロード | 低 | 小規模フォーム |
| react-dropzone | ドラッグ&ドロップ | 低〜中 | React アプリ全般 |
| XMLHttpRequest + progress | プログレス表示 | 中 | UX重視のアプリ |
| S3 Presigned URL | サーバー負荷軽減 | 中 | 中〜大規模アプリ |
| Canvas API | クライアント側リサイズ | 中 | 画像アップロード |
| チャンクアップロード | 大ファイル対応 | 高 | 動画・大容量ファイル |
| tus プロトコル | レジュームアブル | 中 | 不安定なネットワーク |
| Uppy | フル機能UI | 低 | 高機能アップロード |
| Sharp | サーバー側画像処理 | 中 | 画像最適化 |
| Lambda + S3 Event | 非同期画像処理 | 高 | 大規模サービス |

### ベストプラクティスまとめ

1. **クライアントサイド**: accept属性での入力制限、プレビュー表示、クライアントリサイズ
2. **ネットワーク**: プログレス表示、キャンセル機能、リトライ機構
3. **サーバーサイド**: マジックバイト検証、ファイル名サニタイズ、容量制限
4. **セキュリティ**: MIME検証、ウイルススキャン、CSPヘッダー、別ドメイン配信
5. **パフォーマンス**: CDN配信、オンデマンド変換、Web Worker活用

### アンチパターン

| アンチパターン | 問題点 | 正しい実装 |
|---------------|--------|-----------|
| Content-Type を手動設定 | boundary が欠落する | ブラウザに自動設定させる |
| ファイル名をそのまま使用 | パストラバーサルリスク | UUID でリネーム |
| 拡張子のみで検証 | MIME偽装が可能 | マジックバイトで実際の内容を検証 |
| Object URL を解放しない | メモリリーク | revokeObjectURL() を確実に呼ぶ |
| サーバー経由で大ファイル | サーバー帯域の浪費 | S3直接アップロードを使用 |
| 同期的な画像処理 | サーバーのレスポンス遅延 | 非同期ジョブキューに委譲 |
| エラーハンドリング不足 | ユーザーが原因不明の失敗に遭遇 | 詳細なエラーメッセージを表示 |
| 無制限のアップロード | DoS攻撃のリスク | サイズ・回数・容量の制限 |

---

## 次に読むべきガイド
→ [[03-complex-forms.md]] -- 複雑なフォーム

---

## 参考文献
1. MDN Web Docs. "File API." developer.mozilla.org, 2024.
2. MDN Web Docs. "Using FormData Objects." developer.mozilla.org, 2024.
3. MDN Web Docs. "HTML Drag and Drop API." developer.mozilla.org, 2024.
4. react-dropzone. "Simple HTML5 drag-drop zone." react-dropzone.js.org, 2024.
5. AWS. "Presigned URLs." docs.aws.amazon.com, 2024.
6. AWS. "Multipart Upload Overview." docs.aws.amazon.com, 2024.
7. tus. "tus - resumable file uploads." tus.io, 2024.
8. Sharp. "High performance Node.js image processing." sharp.pixelplumbing.com, 2024.
9. Uppy. "The next open source file uploader for web browsers." uppy.io, 2024.
10. OWASP. "File Upload Cheat Sheet." cheatsheetseries.owasp.org, 2024.
11. Multer. "Node.js middleware for handling multipart/form-data." github.com/expressjs/multer, 2024.
12. Cloudinary. "Image and Video Upload, Storage, Optimization and CDN." cloudinary.com, 2024.
