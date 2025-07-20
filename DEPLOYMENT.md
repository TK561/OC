# デプロイメントガイド

## 開発環境での起動

### 自動起動（推奨）
```bash
./start_dev.sh
```

### 手動起動

#### バックエンド
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### フロントエンド
```bash
cd frontend
npm install
npm run dev
```

## テスト実行

### バックエンドテスト
```bash
# バックエンドサーバーが起動している状態で
python test_backend.py
```

### アクセス確認
- **フロントエンド**: http://localhost:3000
- **バックエンドAPI**: http://localhost:8000
- **API ドキュメント**: http://localhost:8000/docs

## 本番デプロイ

### Vercel (フロントエンド)

1. GitHubリポジトリにプッシュ
2. Vercelでプロジェクトをインポート
3. ルートディレクトリを `frontend` に設定
4. 環境変数を設定:
   ```
   NEXT_PUBLIC_BACKEND_URL=https://your-backend-url.com
   ```
5. デプロイ実行

### Railway/Render (バックエンド)

#### Railway
1. GitHubリポジトリを接続
2. `backend` フォルダを指定
3. 環境変数を設定:
   ```
   ENVIRONMENT=production
   MODEL_CACHE_DIR=/app/models
   TEMP_DIR=/app/temp
   ```
4. 自動デプロイ

#### Render
1. GitHubリポジトリを接続
2. Dockerfileを使用してデプロイ
3. 環境変数を設定
4. Webサービスとして起動

## 環境変数

### バックエンド (.env)
```bash
ENVIRONMENT=development|production
MODEL_CACHE_DIR=./models
TEMP_DIR=./temp
HUGGINGFACE_TOKEN=optional_hf_token
PORT=8000
```

### フロントエンド (.env.local)
```bash
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

## トラブルシューティング

### よくある問題

1. **ポート衝突**
   - バックエンド: PORT環境変数で変更
   - フロントエンド: `npm run dev -- -p 3001`

2. **CORS エラー**
   - `backend/app/config.py` のALLOWED_ORIGINSを確認

3. **モデル読み込み失敗**
   - インターネット接続を確認
   - HUGGINGFACE_TOKENを設定（必要に応じて）

4. **依存関係エラー**
   - Python: `pip install -r requirements.txt`
   - Node.js: `npm install`

### パフォーマンス最適化

1. **GPU使用**
   - CUDA対応環境でPyTorchをGPU版に変更

2. **モデルキャッシュ**
   - MODEL_CACHE_DIRを永続化ストレージに設定

3. **メモリ最適化**
   - 画像サイズを512px以下に制限
   - バッチサイズを調整

## セキュリティ考慮事項

1. **ファイルアップロード**
   - ファイルサイズ制限（50MB）
   - ファイル形式検証
   - 一時ファイルの自動削除

2. **API制限**
   - レート制限（必要に応じて追加）
   - 認証機能（将来の拡張）

3. **HTTPS**
   - 本番環境では必須
   - SSL証明書の設定