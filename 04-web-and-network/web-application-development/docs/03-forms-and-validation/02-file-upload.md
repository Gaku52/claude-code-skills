# ファイルアップロード

> ファイルアップロードはフォームの中でも特に複雑な領域。ドラッグ&ドロップ、プログレス表示、プレビュー、S3直接アップロード、画像リサイズまで、プロダクション品質のファイルアップロード実装を習得する。

## この章で学ぶこと

- [ ] ドラッグ&ドロップアップロードの実装を理解する
- [ ] S3プリサインドURLによる直接アップロードを把握する
- [ ] 画像プレビューとバリデーションを学ぶ

---

## 1. 基本的なファイルアップロード

```typescript
// React Hook Form + ファイル入力
function FileUploadForm() {
  const { register, handleSubmit } = useForm();

  const onSubmit = async (data: any) => {
    const formData = new FormData();
    formData.append('file', data.file[0]);
    formData.append('name', data.name);

    await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input
        type="file"
        accept="image/jpeg,image/png,image/webp"
        {...register('file', {
          required: 'File is required',
          validate: {
            size: (files) =>
              files[0]?.size <= 5 * 1024 * 1024 || 'Max file size is 5MB',
            type: (files) =>
              ['image/jpeg', 'image/png', 'image/webp'].includes(files[0]?.type)
              || 'Only JPEG, PNG, WebP are allowed',
          },
        })}
      />
      <button type="submit">Upload</button>
    </form>
  );
}
```

---

## 2. ドラッグ&ドロップ

```typescript
// react-dropzone を使った実装
import { useDropzone } from 'react-dropzone';

function FileDropzone({ onUpload }: { onUpload: (files: File[]) => void }) {
  const [previews, setPreviews] = useState<string[]>([]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
    maxSize: 10 * 1024 * 1024,    // 10MB
    maxFiles: 5,
    onDrop: (acceptedFiles) => {
      // プレビュー生成
      const urls = acceptedFiles.map(file => URL.createObjectURL(file));
      setPreviews(urls);
      onUpload(acceptedFiles);
    },
    onDropRejected: (rejections) => {
      rejections.forEach(({ errors }) => {
        errors.forEach(e => toast.error(e.message));
      });
    },
  });

  return (
    <div>
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}`}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Drop files here...</p>
        ) : (
          <p>Drag & drop files here, or click to select</p>
        )}
      </div>

      {previews.length > 0 && (
        <div className="flex gap-2 mt-4">
          {previews.map((url, i) => (
            <img key={i} src={url} alt="" className="w-20 h-20 object-cover rounded" />
          ))}
        </div>
      )}
    </div>
  );
}
```

---

## 3. プログレス付きアップロード

```typescript
// XMLHttpRequest でプログレス取得
function useFileUpload() {
  const [progress, setProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  const upload = async (file: File, url: string) => {
    setIsUploading(true);
    setProgress(0);

    return new Promise<string>((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          setProgress(Math.round((e.loaded / e.total) * 100));
        }
      });

      xhr.addEventListener('load', () => {
        setIsUploading(false);
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(xhr.responseText);
        } else {
          reject(new Error(`Upload failed: ${xhr.status}`));
        }
      });

      xhr.addEventListener('error', () => {
        setIsUploading(false);
        reject(new Error('Upload failed'));
      });

      const formData = new FormData();
      formData.append('file', file);

      xhr.open('POST', url);
      xhr.send(formData);
    });
  };

  return { upload, progress, isUploading };
}

// プログレスバー
function UploadProgress({ progress }: { progress: number }) {
  return (
    <div className="w-full bg-gray-200 rounded-full h-2">
      <div
        className="bg-blue-500 h-2 rounded-full transition-all"
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}
```

---

## 4. S3直接アップロード（プリサインドURL）

```typescript
// サーバー側: プリサインドURL生成
// app/api/upload/route.ts
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

const s3 = new S3Client({ region: 'ap-northeast-1' });

export async function POST(request: Request) {
  const { filename, contentType } = await request.json();

  const key = `uploads/${crypto.randomUUID()}/${filename}`;

  const command = new PutObjectCommand({
    Bucket: process.env.S3_BUCKET,
    Key: key,
    ContentType: contentType,
  });

  const presignedUrl = await getSignedUrl(s3, command, { expiresIn: 3600 });

  return Response.json({ presignedUrl, key });
}

// クライアント側: S3に直接アップロード
async function uploadToS3(file: File) {
  // 1. プリサインドURL取得
  const { presignedUrl, key } = await fetch('/api/upload', {
    method: 'POST',
    body: JSON.stringify({
      filename: file.name,
      contentType: file.type,
    }),
  }).then(r => r.json());

  // 2. S3に直接アップロード（サーバーを経由しない）
  await fetch(presignedUrl, {
    method: 'PUT',
    body: file,
    headers: { 'Content-Type': file.type },
  });

  // 3. アップロード完了後のURLを返す
  return `https://${process.env.NEXT_PUBLIC_S3_BUCKET}.s3.amazonaws.com/${key}`;
}
```

---

## 5. 画像最適化

```
アップロード時の画像処理:

  クライアント側（ブラウザ）:
  → Canvas API でリサイズ
  → 最大幅/高さの制限
  → WebP への変換
  → EXIF方向の修正

  サーバー側（Lambda / Edge Function）:
  → Sharp でリサイズ・変換
  → 複数サイズの生成（サムネイル、中、大）
  → CDNでのオンデマンド変換（Cloudinary, imgix）

バリデーション:
  → ファイルサイズ: 最大10MB
  → ファイル形式: JPEG, PNG, WebP, AVIF
  → 画像サイズ: 最大4000x4000px
  → ファイル数: 最大10ファイル
  → MIME タイプ検証（拡張子だけでなく）
```

---

## まとめ

| パターン | 用途 |
|---------|------|
| react-dropzone | ドラッグ&ドロップ |
| XMLHttpRequest | プログレス表示 |
| S3 Presigned URL | サーバーを経由しない大ファイル |
| Canvas API | クライアント側リサイズ |

---

## 次に読むべきガイド
→ [[03-complex-forms.md]] — 複雑なフォーム

---

## 参考文献
1. react-dropzone. "Simple HTML5 drag-drop zone." react-dropzone.js.org, 2024.
2. AWS. "Presigned URLs." docs.aws.amazon.com, 2024.
