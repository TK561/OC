# DepthAnything V2 深度推定API - Google Colab Backend

無料でDepthAnything V2モデルを使用した本格的な深度推定APIをGoogle Colab上で実行します。

## 🚀 クイックスタート

### 1. Google Colabでの設定

1. **Google Colabを開く**
   - [Google Colab](https://colab.research.google.com/)にアクセス
   - GPU ランタイムを選択（ランタイム > ランタイムのタイプを変更 > GPU）

2. **ノートブックのアップロード**
   - `setup_colab.ipynb` をGoogle Colabにアップロード
   - または、新しいノートブックを作成してセルをコピー

3. **ngrokトークンの取得**
   - [ngrok.com](https://ngrok.com) でアカウント作成（無料）
   - Dashboard > Your Authtoken からトークンをコピー

4. **セルの実行**
   - セルを順番に実行
   - ngrokトークンを設定
   - 最終的にパブリックURLが表示される

### 2. フロントエンドの設定

生成されたngrok URLをフロントエンドに設定:

```bash
# frontend/.env.local
NEXT_PUBLIC_BACKEND_URL=https://xxxxxxxx.ngrok-free.app
```

## 🔧 技術仕様

### 使用モデル
- **DepthAnything V2 Small**: 高速で軽量な深度推定
- **処理速度**: GPU使用時 2-5秒/画像
- **メモリ使用量**: 約2-3GB

### API仕様
- **エンドポイント**: `/call/predict`
- **メソッド**: POST
- **入力**: Gradio形式のJSON
- **出力**: 元画像 + 深度マップ

### 例: API リクエスト
```javascript
const response = await fetch(`${API_URL}/call/predict`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    data: [imageDataUrl],
    session_hash: Math.random().toString(36).substring(7)
  })
});
```

## 💡 利点

### 🆓 完全無料
- Google Colab: 無料GPU使用
- ngrok: 無料プラン（8時間/URL）
- DepthAnything V2: オープンソース

### 🚀 高性能
- T4 GPU使用で高速推論
- 最新のDepthAnything V2モデル
- リアルタイム深度推定

### 🔄 安定性
- 自動フォールバック機能
- エラーハンドリング
- セッション管理

## ⚠️ 制限事項

### 時間制限
- **Colab セッション**: 12時間で切断
- **ngrok URL**: 8時間で期限切れ（無料版）
- **アイドル状態**: 90分で自動切断

### 解決策
1. **定期的な実行**: セルを定期実行してセッション維持
2. **ngrok Pro**: 永続URL（月$8）
3. **複数アカウント**: ローテーション使用

## 🛠️ トラブルシューティング

### よくある問題

#### 1. モデル読み込みエラー
```
OutOfMemoryError: CUDA out of memory
```
**解決策**:
- ランタイムを再起動
- 他のプロセスを終了
- CPUモードで実行

#### 2. ngrok接続エラー
```
NgrokError: The authtoken you specified is invalid
```
**解決策**:
- ngrokトークンを再確認
- 新しいトークンを生成
- 特殊文字をエスケープ

#### 3. Gradio起動エラー
```
OSError: [Errno 98] Address already in use
```
**解決策**:
- ランタイムを再起動
- ポート番号を変更
- プロセスを手動終了

### デバッグ方法

```python
# GPU確認
!nvidia-smi

# メモリ使用量確認
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

# ngrok接続確認
import requests
response = requests.get(f"{public_url}/")
print(response.status_code)
```

## 🔄 代替案

### 1. Hugging Face Spaces
- **利点**: 永続URL、メンテナンス不要
- **欠点**: メモリ制限、待機時間

### 2. Replicate API
- **利点**: 安定したAPI、課金制
- **欠点**: 無料クレジット制限

### 3. ローカル実行
- **利点**: 制限なし、プライベート
- **欠点**: GPU必要、設定複雑

## 📊 パフォーマンス比較

| 方法 | 速度 | コスト | 安定性 | 設定難易度 |
|------|------|--------|--------|------------|
| Google Colab | ⭐⭐⭐⭐ | 🆓 | ⭐⭐ | ⭐⭐⭐ |
| HF Spaces | ⭐⭐ | 🆓 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Replicate | ⭐⭐⭐⭐⭐ | 💰 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| ローカル | ⭐⭐⭐⭐⭐ | 💰💰 | ⭐⭐⭐⭐⭐ | ⭐ |

## 🤝 サポート

問題が発生した場合:
1. このREADMEのトラブルシューティングを確認
2. Colab上でデバッグコードを実行
3. 必要に応じてランタイムを再起動