# 📅 2日間展示運用スケジュール

## 🔄 URL更新タイミング（8時間毎）

### Day 1
- **09:00**: 初回起動 ✅
- **17:00**: URL更新 #1
- **01:00**: URL更新 #2

### Day 2  
- **09:00**: URL更新 #3 + Colabセッション再起動
- **17:00**: URL更新 #4
- **展示終了**: システム停止

## ⚡ URL更新手順（3分で完了）

### 方法A: 新ポートで再起動
```python
# Google Colabの新しいセルで実行
import random
new_port = random.randint(7000, 8000)
public_url = ngrok.connect(new_port)
print(f"🆕 新しいURL: {public_url}")
```

### 方法B: 既存トンネル切断→再接続
```python
# 既存接続を切断
ngrok.disconnect(7860)
# 新しい接続
public_url = ngrok.connect(7860)
print(f"🆕 新しいURL: {public_url}")
```

### Vercel更新
1. 新しいURLをコピー
2. Vercel環境変数を更新
3. Redeployクリック

## 🚨 緊急対応

### Colab切断時
1. ランタイム → すべてのランタイムをリセット
2. 最初のコードを再実行
3. 新URLでVercel更新

### ngrokエラー時
1. 別のポート番号で再試行
2. それでもダメなら新しいColabノートブック作成

## 📱 監視ポイント
- Colabタブを開いたままにする
- 8時間毎のアラームセット推奨
- スマホからでもVercel更新可能

## ✅ 展示成功のコツ
1. **事前テスト**: 前日に全手順確認
2. **バックアップ**: 予備のGoogleアカウント準備
3. **簡易マニュアル**: URL更新手順を印刷
4. **緊急連絡先**: 技術担当者の連絡先確保