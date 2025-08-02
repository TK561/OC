# 深度推定・エッジ検出画像処理アプリ

DPT-Large、MiDaS、Depth Anythingによる深度推定とCannyエッジ検出を組み合わせた高度な画像処理アプリケーション。

## ✨ 新機能

- **エッジ検出と深度推定の融合**
  - Cannyエッジ検出器による精密なエッジ抽出
  - 深度情報とエッジ情報の合成
  - 純粋に深度に基づいた白黒グラデーション生成

- **複数の深度推定モデル**
  - Intel/dpt-large - 最高精度の深度推定
  - Intel/dpt-hybrid-midas - バランス重視
  - LiheYoung/depth-anything-small-hf - 軽量高速（推奨）

- **高度な画像処理**
  - 深度マップの反転・調整
  - マスク合成（乗算・条件付き・アルファブレンド）
  - ガンマ補正・ガウシアンブラー後処理

- **リアルタイム3D可視化**
  - ポイントクラウド生成
  - インタラクティブな3D表示
  - 回転・ズーム操作

## 🏗️ アーキテクチャ

```
├── backend/              # FastAPI バックエンド
│   ├── app.py           # メインアプリケーション
│   ├── requirements.txt # Python依存関係
│   ├── nixpacks.toml    # Railway設定
│   ├── Depth-Anything-V2/  # 深度推定モデル実装
│   └── ml-depth-pro/    # Apple DepthPro実装
├── frontend/            # Next.js フロントエンド
│   ├── components/      # React コンポーネント
│   ├── pages/          # ページ
│   ├── lib/            # ライブラリ
│   └── public/         # 静的ファイル
├── docs/               # ドキュメント
└── scripts/            # 分析・テストスクリプト
```

## 🔌 APIエンドポイント

### 基本深度推定
- `POST /api/predict` - 標準の深度推定
- `GET /health` - ヘルスチェック

### 新機能：エッジ検出+深度処理
- `POST /api/depth-edge-processing` - エッジ検出と深度推定の融合処理

#### パラメータ
- `model`: 深度推定モデル選択
- `edge_low_threshold`: Cannyエッジ検出低閾値 (デフォルト: 50)
- `edge_high_threshold`: Cannyエッジ検出高閾値 (デフォルト: 150)
- `invert_depth`: 深度マップ反転 (デフォルト: true)
- `depth_gamma`: 深度ガンマ補正 (デフォルト: 1.0)
- `composition_mode`: 合成方法 ("multiply", "conditional", "alpha_blend")

## 技術スタック

### フロントエンド
- Next.js 14
- TypeScript
- Three.js + React Three Fiber
- Tailwind CSS
- Vercel (デプロイ)

### バックエンド
- FastAPI (Python Web API)
- OpenCV / PIL (画像処理)
- NumPy / SciPy (数値計算)
- Pillow (画像操作)
- Railway (デプロイ)

### 画像処理アルゴリズム
- Cannyエッジ検出
- 深度推定（DPT、MiDaS、DepthAnything）
- ガンマ補正・コントラスト調整
- マスク合成・ブレンディング

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
python app.py
# または
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
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