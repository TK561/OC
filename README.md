# 深度推定・3D可視化アプリ

GitHub実装に基づいたDPT-Large、MiDaS、Depth Anything V2による深度推定と3D可視化アプリケーション。

## ✨ 特徴

- **複数の深度推定モデル**
  - DPT-Large (Intel) - 高精度深度推定
  - MiDaS v3.1 (Intel) - 高速でバランスの良い性能
  - Depth Anything V2 (Small/Base/Large) - 最新のTransformerベース

- **リアルタイム3D可視化**
  - ポイントクラウド生成
  - インタラクティブな3D表示
  - 回転・ズーム操作

- **正確な深度表現**
  - 白=近い、黒=遠い（GitHubの実装に準拠）
  - 段階的なグラデーション変化
  - 滑らかな深度マップ

## 🏗️ アーキテクチャ

```
├── backend/           # FastAPI バックエンド
│   ├── app/
│   │   ├── models/    # 深度推定モデル
│   │   ├── routers/   # API エンドポイント
│   │   └── utils/     # ユーティリティ
│   └── requirements_railway.txt
├── frontend/          # Next.js フロントエンド
│   ├── components/    # React コンポーネント
│   ├── pages/         # ページ
│   └── lib/           # ライブラリ
├── railway-backend/   # Railway軽量版
└── docs/             # ドキュメント
```

## 技術スタック

### フロントエンド
- Next.js 14
- TypeScript
- Three.js + React Three Fiber
- Tailwind CSS
- Vercel (デプロイ)

### バックエンド
- FastAPI
- PyTorch + Hugging Face Transformers
- OpenCV
- Railway/Render (デプロイ)

## 開発環境セットアップ

### 前提条件
- Node.js 18+
- Python 3.9+
- Git

### フロントエンド

```bash
cd frontend
npm install
npm run dev
```

### バックエンド

```bash
cd backend

# 仮想環境作成（推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 依存関係インストール
pip install -r requirements.txt

# 開発サーバー起動
uvicorn app.main:app --reload
```

## 環境変数

### フロントエンド (.env.local)
```
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

### バックエンド (.env)
```
ENVIRONMENT=development
HUGGINGFACE_TOKEN=your_token_here
MODEL_CACHE_DIR=./models
TEMP_DIR=./temp
```

## デプロイ

### Vercel (フロントエンド)
1. Vercelアカウントでリポジトリを接続
2. `frontend`フォルダをルートディレクトリに設定
3. 環境変数`NEXT_PUBLIC_BACKEND_URL`を設定

### Railway/Render (バックエンド)
1. `backend`フォルダをデプロイ
2. 必要な環境変数を設定
3. Dockerfileを使用して自動デプロイ

## 使用方法

1. 画像をアップロード
2. 深度推定モデルを選択
3. 深度マップと3D可視化を生成
4. 結果をエクスポート

## 開発ステータス

- [x] プロジェクト基盤構築
- [x] フロントエンド環境設定
- [x] バックエンド環境設定
- [x] 深度推定モデル統合
- [x] 3D可視化実装
- [x] UI/UX実装
- [x] 基本機能完成
- [ ] デプロイ・最適化
- [ ] 本番環境テスト

## クイックスタート

```bash
# 開発環境を一括起動
./start_dev.sh

# または手動で起動
cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && uvicorn app.main:app --reload &
cd frontend && npm install && npm run dev
```

## テスト

```bash
# バックエンドのテスト実行
python test_backend.py
```

## デプロイ

詳細な手順は [DEPLOYMENT.md](./DEPLOYMENT.md) を参照してください。

## ライセンス

MIT License