# プロジェクト整理計画

## 🎯 整理の目的
- 重複ファイルの削除
- 目的別ディレクトリ構成の明確化
- メンテナンス性の向上

## 📁 新しいディレクトリ構成

```
/
├── 🎯 メイン実装
│   ├── frontend/                    # Next.js本番アプリ
│   ├── backend/                     # フル機能APIサーバー
│   └── railway-backend/             # Railway専用軽量版
│
├── 🧪 開発・実験
│   ├── experiments/
│   │   ├── colab/                   # Google Colab用
│   │   ├── local/                   # ローカル開発用
│   │   └── prototypes/              # プロトタイプ
│   │
├── 🚀 デプロイ版
│   ├── deployments/
│   │   ├── huggingface/             # HF Spaces用
│   │   ├── api-variants/            # 各種API版
│   │   └── demos/                   # デモ版
│   │
├── 📚 ドキュメント・設定
│   ├── docs/                        # プロジェクトドキュメント
│   ├── config/                      # 設定ファイル集約
│   ├── scripts/                     # 実行スクリプト
│   └── tests/                       # テストファイル集約
│
└── 🔧 共通・ユーティリティ
    ├── shared/                      # 共通型定義・ユーティリティ
    └── security/                    # セキュリティ設定
```

## 🗂️ 移動・統合計画

### 1. experiments/ に移動
- `colab-backend/` → `experiments/colab/`
- `local-backend/` → `experiments/local/`

### 2. deployments/ に統合
```
deployments/
├── huggingface/                     # 統合版
├── api-variants/
│   ├── clean/                       # depth-estimation-clean
│   ├── standard/                    # depth-estimation-api
│   └── space/                       # depth-estimation-space
└── demos/
    └── exhibition/                  # exhibition-runner
```

### 3. config/ に集約
- `*.json` → `config/deploy/`
- `*.yaml` → `config/deploy/`
- `Procfile` → `config/deploy/`

### 4. scripts/ に集約
- `*.sh` → `scripts/`
- `auto-execute/` → `scripts/auto/`
- `quick-start/` → `scripts/setup/`

### 5. tests/ に集約
- `test-*.html` → `tests/html/`
- `test-*.py` → `tests/python/`
- `test-*.ps1` → `tests/powershell/`

## 🗑️ 削除対象

### 重複ファイル
- `huggingface-space-fix/` （huggingface/に統合）
- `huggingface-space-update/` （huggingface/に統合）
- 古いtest-api.*ファイル
- 重複する設定ファイル

### 未使用ファイル
- `CDesktopdemobackendminimal_server.py`
- `create_test_image.py`
- 古いrailway_response.json

## 📋 実行手順

### Phase 1: バックアップ作成
```bash
git tag v1.0-before-cleanup
git push origin v1.0-before-cleanup
```

### Phase 2: 新ディレクトリ作成
```bash
mkdir -p experiments/{colab,local,prototypes}
mkdir -p deployments/{huggingface,api-variants/{clean,standard,space},demos/exhibition}
mkdir -p config/{deploy,env}
mkdir -p scripts/{auto,setup}
mkdir -p tests/{html,python,powershell}
```

### Phase 3: ファイル移動・統合
- 段階的に移動
- 各段階でテスト実行
- 問題があれば即座にロールバック

### Phase 4: 不要ファイル削除
- 重複ファイルの削除
- 未使用ファイルの削除

### Phase 5: ドキュメント更新
- README.md更新
- 各ディレクトリにREADME追加
- デプロイ手順書更新

## ⚠️ 注意事項

1. **デプロイ設定の確認**
   - Vercel, Railway, Renderの設定確認
   - パス変更に伴う設定更新

2. **import文の更新**
   - 相対パスの修正
   - 型定義のパス更新

3. **CI/CDの更新**
   - GitHub Actionsの設定更新
   - テストパスの更新

4. **段階的実行**
   - 一度に全て変更しない
   - 各段階でテスト実行
   - 問題があれば即座に戻す

## 🎯 期待効果

1. **可読性向上**
   - 目的が明確なディレクトリ構成
   - 重複ファイルの削除

2. **メンテナンス性向上**
   - 設定ファイルの集約
   - テストファイルの整理

3. **開発効率向上**
   - 必要なファイルを素早く発見
   - 明確な役割分担