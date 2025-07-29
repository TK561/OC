# 実験・開発用ディレクトリ

## 概要
このディレクトリには、実験的な実装や開発用のコードが含まれています。

## 構成

### `/colab/` - Google Colab用
- **目的**: Google Colab環境での実験・プロトタイピング
- **技術**: Jupyter Notebook + 深度推定API
- **用途**: 研究・新機能の検証

### `/local/` - ローカル開発用
- **目的**: ローカル環境での開発・テスト
- **技術**: 軽量実装版
- **用途**: 開発時のクイックテスト・デバッグ

## 使用方法

### Colabでの実験
```bash
cd experiments/colab
# Jupyter Notebookを開いて実験
```

### ローカル開発
```bash
cd experiments/local
pip install -r requirements.txt
python main.py
```

## 注意事項
- このディレクトリのコードは実験的なものです
- 本番環境では使用しないでください
- 定期的にメイン実装に反映してください