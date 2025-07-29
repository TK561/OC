# スクリプト・ユーティリティディレクトリ

## 概要
このディレクトリには、プロジェクトの実行・デプロイ・セットアップ用スクリプトが含まれています。

## 構成

### `/auto/` - 自動実行スクリプト
- **2day_schedule.md**: 2日間のスケジュール
- **colab_one_click.txt**: Colab用ワンクリック設定
- **用途**: 自動化された実行・デプロイ

### `/setup/` - セットアップスクリプト
- **colab_setup.py**: Google Colab環境のセットアップ
- **用途**: 開発環境の初期設定

### ルート階層スクリプト
- **deploy.sh**: デプロイスクリプト
- **quick_deploy.sh**: クイックデプロイ
- **start_dev.sh**: 開発環境起動
- **test-api.sh**: API テスト実行

## 使用方法

### 開発環境の起動
```bash
./scripts/start_dev.sh
```

### デプロイ実行
```bash
./scripts/deploy.sh
```

### クイックデプロイ
```bash
./scripts/quick_deploy.sh
```

### APIテスト
```bash
./scripts/test-api.sh
```

## 自動化セットアップ

### Google Colab
```bash
cd scripts/setup
python colab_setup.py
```

### 2日間スケジュール実行
```bash
cd scripts/auto
# 2day_schedule.mdの手順に従って実行
```

## 注意事項
- スクリプト実行前に権限を確認してください
- 環境変数が適切に設定されていることを確認してください
- テストスクリプトは開発環境で実行してください