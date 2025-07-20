# Vercel デプロイガイド

## Vercel CLI を使ったデプロイ

### 1. Vercel CLI でログイン
```bash
cd frontend
npx vercel login
```

### 2. 初回デプロイ
```bash
npx vercel
```

以下の質問に答える：
- Set up and deploy "frontend"? [Y/n] → **Y**
- Which scope? → **あなたのアカウント**
- Link to existing project? [y/N] → **N**
- What's your project's name? → **depth-estimation-app**
- In which directory is your code located? → **./（Enter）**

### 3. 本番デプロイ
```bash
npx vercel --prod
```

## 環境変数の設定

### Vercel ダッシュボードで設定
1. https://vercel.com/dashboard でプロジェクトを選択
2. Settings → Environment Variables
3. 以下を追加：

```
Name: NEXT_PUBLIC_BACKEND_URL
Value: https://your-backend-url.onrender.com
Environment: Production, Preview, Development
```

### CLI で環境変数を設定
```bash
npx vercel env add NEXT_PUBLIC_BACKEND_URL
# 値を入力: https://your-backend-url.onrender.com
# 環境を選択: Production, Preview, Development
```

## GitHub 連携デプロイ

### 1. GitHub からインポート
1. https://vercel.com/new
2. "Import Git Repository" → GitHub リポジトリを選択
3. Framework Preset: **Next.js**
4. Root Directory: **frontend**
5. Environment Variables を設定
6. Deploy をクリック

### 2. 自動デプロイ設定
- mainブランチへのpushで自動デプロイ
- Pull Requestで Preview デプロイ

## トラブルシューティング

### 404 エラーの解決

#### 原因1: ルートディレクトリの設定ミス
```bash
# vercel.json で確認
{
  "framework": "nextjs"
}
```

#### 原因2: バックエンドURLの設定ミス
```bash
# .env.local を確認
NEXT_PUBLIC_BACKEND_URL=https://your-backend-url.onrender.com
```

#### 原因3: Rewritesの設定
```json
{
  "rewrites": [
    {
      "source": "/api/backend/:path*",
      "destination": "https://your-backend-url.onrender.com/:path*"
    }
  ]
}
```

### ビルドエラーの解決

#### TypeScript エラー
```bash
npm run lint
npm run build
```

#### 依存関係の問題
```bash
npm install
npm audit fix
```

### デプロイメント設定の確認

```bash
# プロジェクト情報確認
npx vercel ls

# ログ確認
npx vercel logs

# ドメイン確認
npx vercel domains
```

## パフォーマンス最適化

### 1. 画像最適化
```javascript
// next.config.js
module.exports = {
  images: {
    domains: ['your-backend-domain.com'],
    formats: ['image/webp', 'image/avif'],
  },
}
```

### 2. バンドルサイズ分析
```bash
npm install @next/bundle-analyzer
```

### 3. Edge Functions活用
```javascript
// pages/api/proxy.js で軽い処理をEdge側で実行
export const config = {
  runtime: 'edge',
}
```

## セキュリティ設定

### 1. CORS設定
```json
{
  "headers": [
    {
      "source": "/api/(.*)",
      "headers": [
        { "key": "Access-Control-Allow-Origin", "value": "your-domain.com" },
        { "key": "Access-Control-Allow-Methods", "value": "GET, POST, OPTIONS" }
      ]
    }
  ]
}
```

### 2. セキュリティヘッダー
```json
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "X-Frame-Options", "value": "DENY" },
        { "key": "X-Content-Type-Options", "value": "nosniff" },
        { "key": "Referrer-Policy", "value": "origin-when-cross-origin" }
      ]
    }
  ]
}
```