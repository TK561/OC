# テストディレクトリ

## 概要
このディレクトリには、プロジェクトの各種テストファイルが含まれています。

## 構成

### `/html/` - HTMLテスト
- **test-3d-api.html**: 3D API機能のHTMLテスト
- **用途**: ブラウザでの動作確認・UIテスト

### `/python/` - Pythonテスト
- **test_backend.py**: バックエンドAPI全般のテスト
- **test_railway_api.py**: Railway用APIのテスト
- **test-railway-api.py**: Railway API追加テスト
- **用途**: APIの機能テスト・パフォーマンステスト

### `/powershell/` - PowerShellテスト
- **test-api.ps1**: Windows環境でのAPIテスト
- **test_api.ps1**: API追加テスト
- **用途**: Windows環境での動作確認

## 使用方法

### HTMLテスト実行
```bash
# ブラウザでHTMLファイルを開く
open tests/html/test-3d-api.html
```

### Pythonテスト実行
```bash
cd tests/python
python test_backend.py
python test_railway_api.py
```

### PowerShellテスト実行
```powershell
cd tests/powershell
./test-api.ps1
./test_api.ps1
```

## テスト種類

### 機能テスト
- API エンドポイントの動作確認
- レスポンス形式の検証
- エラーハンドリングの確認

### 統合テスト
- フロントエンド・バックエンド間の連携
- 3D表示機能の動作確認
- ファイルアップロード・処理の確認

### パフォーマンステスト
- API応答時間の測定
- 大容量画像の処理確認
- 並列処理の動作確認

## 注意事項
- テスト実行前にサーバーが起動していることを確認してください
- テスト用の画像ファイルが必要な場合があります
- 環境変数（API URL等）を適切に設定してください