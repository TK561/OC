# デプロイメントガイド

## 1. Railway バックエンドデプロイ

### 初期設定
1. Railwayアカウント作成
2. GitHubリポジトリと連携
3. 新しいプロジェクト作成

### デプロイ設定
```json
// railway.json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn app:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### 重要な設定
- **Root Directory**: `railway-backend`
- **Branch**: `master`
- **Region**: Southeast Asia (Singapore)

### サイズ制限対策
- Docker image: 最大4GB
- 軽量ライブラリのみ使用（Pillow、FastAPI）
- NumPy、PyTorch、Transformersは使用不可

## 2. Vercel フロントエンドデプロイ

### 初期設定
1. Vercelアカウント作成
2. GitHubリポジトリをインポート
3. 環境変数設定

### 環境変数
```
NEXT_PUBLIC_BACKEND_URL=https://web-production-a0df.up.railway.app
```

### ビルド設定
- **Framework Preset**: Next.js
- **Build Command**: `npm run build`
- **Output Directory**: `.next`
- **Install Command**: `npm install`

### TypeScript設定修正
```json
// tsconfig.json
{
  "paths": {
    "@/shared/*": ["./shared/*"]  // ../shared/* から修正
  }
}
```

## 3. トラブルシューティング

### Railway デプロイエラー
1. **サイズ超過**: 重いライブラリを削除
2. **ビルドエラー**: requirements.txtを最小限に
3. **API不通**: CORSミドルウェア確認

### Vercel ビルドエラー
1. **TypeScript型エラー**: 型定義の一致確認
2. **パス解決エラー**: tsconfig.jsonのpaths設定
3. **環境変数未定義**: Vercel設定で追加

### 3Dビュー問題
1. **データ未表示**: APIレスポンスにpointcloudData含むか確認
2. **画像反転**: Y軸座標計算の修正
3. **表示されない**: ブラウザコンソールでエラー確認