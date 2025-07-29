# プロジェクト構成整理

## 🎯 メイン実装（本番用）

### `/frontend/` - Next.js フロントエンド
- **目的**: 本番用Webアプリケーション
- **技術**: Next.js + TypeScript + Three.js + Tailwind CSS
- **デプロイ**: Vercel
- **機能**: 
  - 画像アップロード
  - 深度推定表示
  - 3Dポイントクラウド可視化
  - ズーム・回転操作

### `/backend/` - FastAPI バックエンド（フル機能版）
- **目的**: 本番用APIサーバー
- **技術**: FastAPI + PyTorch + Open3D + Transformers
- **デプロイ**: Render/Railway
- **機能**:
  - 複数深度推定モデル対応
  - 3Dポイントクラウド生成
  - 高品質画像処理

### `/railway-backend/` - Railway専用軽量版
- **目的**: Railway専用の軽量APIサーバー
- **技術**: FastAPI + Pillow（NumPy不要）
- **デプロイ**: Railway
- **機能**:
  - Pillowベース深度推定
  - 軽量3Dポイントクラウド生成
  - 高速処理

---

## 🧪 実験・開発用

### `/colab-backend/` - Google Colab用
- **目的**: Google Colab環境での実験
- **技術**: Jupyter Notebook + 深度推定API
- **用途**: プロトタイピング・研究

### `/local-backend/` - ローカル開発用
- **目的**: ローカル環境での開発・テスト
- **技術**: 軽量実装版
- **用途**: 開発時のクイックテスト

---

## 🚀 各種デプロイ版

### `/huggingface/` - Hugging Face Spaces用
- **目的**: Hugging Face Spacesでのデモ
- **技術**: Gradio + 深度推定
- **用途**: 公開デモ

### `/huggingface-space-fix/` - HF Spaces修正版
- **目的**: Hugging Face Spaces用の修正版
- **用途**: デプロイ問題の修正

### `/huggingface-space-update/` - HF Spaces更新版
- **目的**: Hugging Face Spaces用の最新版
- **用途**: 機能更新版

### `/depth-estimation-api/` - 汎用API版
- **目的**: 汎用的な深度推定API
- **用途**: 他プロジェクトでの利用

### `/depth-estimation-clean/` - クリーン版API
- **目的**: シンプルな深度推定API
- **用途**: 最小限の実装

### `/depth-estimation-space/` - Space版API
- **目的**: Space環境用API
- **用途**: クラウド環境での利用

---

## 📚 ドキュメント・設定

### `/docs/` - プロジェクトドキュメント
- **内容**: 
  - 技術仕様書
  - 実装詳細
  - トラブルシューティング
  - アルゴリズム比較

### `/auto-execute/` - 自動実行スクリプト
- **内容**: デプロイ・実行の自動化

### `/exhibition-runner/` - 展示会用
- **内容**: 展示会での自動実行設定

### `/quick-start/` - クイックスタート
- **内容**: 簡単セットアップ用

### `/security/` - セキュリティ設定
- **内容**: 認証情報・セキュリティ設定

---

## 🔧 ユーティリティ・テスト

### `/shared/` - 共通型定義
- **内容**: TypeScript型定義

### テストファイル群
- `test-*.html`, `test-*.py`, `test-*.ps1`
- **用途**: API動作確認・テスト

### 設定ファイル群
- `*.json`, `*.yaml`, `*.md`
- **用途**: 各種デプロイ設定

---

## 🎯 推奨使用パターン

### 1. 本番運用
- **フロントエンド**: `/frontend/` → Vercel
- **バックエンド**: `/railway-backend/` → Railway

### 2. 開発・実験
- **ローカル開発**: `/local-backend/`
- **実験**: `/colab-backend/`

### 3. デモ・公開
- **Webデモ**: `/huggingface/`
- **API提供**: `/depth-estimation-api/`

---

## 🧹 整理提案

### 削除候補（重複・古いファイル）
- 古いバージョンのhuggingface-space-*
- 使われていないtest-*ファイル
- 重複する設定ファイル

### 統合候補
- depth-estimation-* 系を統合
- テストファイルを `/tests/` に集約
- 設定ファイルを `/config/` に集約