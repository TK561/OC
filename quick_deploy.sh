#!/bin/bash

# クイックデプロイスクリプト
echo "🚀 深度推定アプリのクイックデプロイ"

# 現在のディレクトリ確認
if [ ! -f "render.yaml" ]; then
    echo "❌ エラー: render.yaml が見つかりません"
    exit 1
fi

echo "📋 デプロイオプション:"
echo "1) GitHub経由の自動デプロイ (推奨)"
echo "2) Render Dashboard 手動セットアップガイド"
echo "3) Vercel フロントエンドのみデプロイ"
echo "4) ローカル開発環境起動"

read -p "選択してください (1-4): " choice

case $choice in
    1)
        echo "🔄 GitHub経由の自動デプロイを実行..."
        
        # 変更をコミット
        git add .
        echo "変更をコミットします..."
        read -p "コミットメッセージを入力してください: " commit_msg
        
        if [ -z "$commit_msg" ]; then
            commit_msg="Update deployment configuration"
        fi
        
        git commit -m "$commit_msg"
        git push
        
        echo "✅ GitHubにプッシュ完了"
        echo "📋 次のステップ:"
        echo "1. https://render.com/dashboard でアカウント作成/ログイン"
        echo "2. 'New +' → 'Web Service'"
        echo "3. GitHub リポジトリ 'kanalia7355/OC_display' を選択"
        echo "4. render.yaml の設定が自動で読み込まれます"
        echo "5. 'Create Web Service' をクリック"
        echo ""
        echo "🔗 Render Dashboard: https://render.com/dashboard"
        ;;
        
    2)
        echo "📖 Render Dashboard 手動セットアップガイド"
        echo ""
        echo "1. https://render.com/dashboard にアクセス"
        echo "2. 'New +' → 'Web Service' をクリック"
        echo "3. 'Build and deploy from a Git repository'"
        echo "4. GitHub リポジトリ 'kanalia7355/OC_display' を選択"
        echo ""
        echo "設定内容:"
        echo "----------------------------------------"
        echo "Name: depth-estimation-backend"
        echo "Environment: Python 3"
        echo "Region: Frankfurt (EU Central)"
        echo "Branch: master"
        echo "Root Directory: backend"
        echo "Build Command: pip install -r requirements.txt"
        echo "Start Command: uvicorn app.main:app --host 0.0.0.0 --port \$PORT"
        echo "----------------------------------------"
        echo ""
        echo "環境変数:"
        echo "ENVIRONMENT=production"
        echo "MODEL_CACHE_DIR=/opt/render/project/src/models"
        echo "TEMP_DIR=/opt/render/project/src/temp"
        echo "PYTHONPATH=/opt/render/project/src"
        echo ""
        echo "🔗 詳細ガイド: RENDER_DEPLOY.md"
        ;;
        
    3)
        echo "🌐 Vercel フロントエンドデプロイ..."
        cd frontend
        
        # 依存関係インストール
        echo "📦 依存関係をインストール中..."
        npm install
        
        # ビルドテスト
        echo "🔧 ビルドテスト中..."
        npm run build
        
        if [ $? -ne 0 ]; then
            echo "❌ ビルドに失敗しました"
            exit 1
        fi
        
        # Vercel ログイン確認
        npx vercel whoami > /dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "📝 Vercel にログインしてください"
            npx vercel login
        fi
        
        echo "🚀 Vercel にデプロイ中..."
        npx vercel --prod
        
        if [ $? -eq 0 ]; then
            echo "✅ Vercel デプロイ完了"
            echo "⚠️  注意: バックエンドもデプロイして環境変数を設定してください"
        else
            echo "❌ Vercel デプロイに失敗しました"
        fi
        
        cd ..
        ;;
        
    4)
        echo "🛠️  ローカル開発環境を起動..."
        ./start_dev.sh
        ;;
        
    *)
        echo "❌ 無効な選択です"
        exit 1
        ;;
esac

echo ""
echo "📚 関連ドキュメント:"
echo "- 全般: DEPLOYMENT.md"
echo "- Vercel: VERCEL_DEPLOY.md" 
echo "- Render: RENDER_DEPLOY.md"
echo "- CLI代替: RENDER_CLI_ALTERNATIVE.md"