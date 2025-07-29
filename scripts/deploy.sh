#!/bin/bash

# 統合デプロイスクリプト
echo "🚀 深度推定アプリのデプロイを開始します"

# 現在のディレクトリを確認
if [ ! -d "frontend" ] || [ ! -d "backend" ]; then
    echo "❌ エラー: frontend と backend ディレクトリが見つかりません"
    exit 1
fi

# フロントエンドのビルドテスト
echo "🔧 フロントエンドのビルドテスト中..."
cd frontend
npm install
npm run build
if [ $? -ne 0 ]; then
    echo "❌ フロントエンドのビルドに失敗しました"
    exit 1
fi
cd ..

# バックエンドの依存関係チェック
echo "🔧 バックエンドの依存関係をチェック中..."
cd backend
python3 -m pip install -r requirements.txt --dry-run
if [ $? -ne 0 ]; then
    echo "❌ バックエンドの依存関係に問題があります"
    exit 1
fi
cd ..

echo "✅ ビルドテスト完了"

# Vercel デプロイ
echo "🌐 Vercel にフロントエンドをデプロイ中..."
cd frontend

# Vercel にログインしているかチェック
npx vercel whoami > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "📝 Vercel にログインしてください"
    npx vercel login
fi

# デプロイ実行
echo "デプロイモードを選択してください:"
echo "1) Preview デプロイ (テスト用)"
echo "2) Production デプロイ (本番用)"
read -p "選択 (1 or 2): " deploy_mode

case $deploy_mode in
    1)
        echo "📦 Preview デプロイを実行中..."
        npx vercel
        ;;
    2)
        echo "🚀 Production デプロイを実行中..."
        npx vercel --prod
        ;;
    *)
        echo "❌ 無効な選択です"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo "✅ Vercel デプロイ完了"
else
    echo "❌ Vercel デプロイに失敗しました"
    exit 1
fi

cd ..

# 環境変数の確認
echo "🔍 デプロイ後の設定確認"
echo ""
echo "📋 確認項目:"
echo "1. Vercel の環境変数 NEXT_PUBLIC_BACKEND_URL が設定されているか"
echo "2. Render でバックエンドサービスが起動しているか"
echo "3. CORS設定が正しく設定されているか"
echo ""

# デプロイ完了メッセージ
echo "🎉 デプロイプロセス完了！"
echo ""
echo "📌 次のステップ:"
echo "1. Render でバックエンドをデプロイ (RENDER_DEPLOY.md 参照)"
echo "2. 環境変数の設定確認"
echo "3. 本番環境での動作テスト"
echo ""
echo "📚 詳細なガイド:"
echo "- Vercel: VERCEL_DEPLOY.md"
echo "- Render: RENDER_DEPLOY.md"
echo "- 全般: DEPLOYMENT.md"