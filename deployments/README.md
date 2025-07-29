# デプロイメント用ディレクトリ

## 概要
このディレクトリには、各種プラットフォームへのデプロイ用コードが含まれています。

## 構成

### `/huggingface/` - Hugging Face Spaces用
- **プラットフォーム**: Hugging Face Spaces
- **技術**: Gradio + 深度推定
- **用途**: 公開デモ・API提供

### `/api-variants/` - API各種バリアント
#### `/standard/` - 標準API版
- **特徴**: フル機能の深度推定API
- **用途**: 汎用的な深度推定サービス

#### `/clean/` - クリーン版API
- **特徴**: シンプル・軽量な実装
- **用途**: 最小限の機能が必要な場合

#### `/space/` - Space環境用API
- **特徴**: クラウド環境最適化
- **用途**: Space系プラットフォームでの利用

### `/demos/` - デモ・展示用
#### `/exhibition/` - 展示会用
- **用途**: 展示会・イベントでの自動実行
- **特徴**: 自動デプロイ・チェックリスト機能

## デプロイ手順

### Hugging Face Spaces
```bash
cd deployments/huggingface
# ファイルをHF Spacesにアップロード
```

### 各種API
```bash
cd deployments/api-variants/[variant]
# 各プラットフォームの手順に従ってデプロイ
```

## 注意事項
- 各ディレクトリは独立したデプロイメント単位です
- 環境変数・設定は各プラットフォームに合わせて調整してください