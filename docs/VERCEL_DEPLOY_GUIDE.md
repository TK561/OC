# Vercelデプロイ設定ガイド

## 現在の問題
GitHub ActionsでのVercelデプロイが、認証トークンが設定されていないため失敗しています。

## 解決方法

### 方法1: GitHub Secretsを設定（推奨）

1. **Vercelトークンの取得**
   - https://vercel.com/account/tokens にアクセス
   - 「Create」をクリック
   - トークン名: `github-actions`
   - スコープ: `Full Account`
   - トークンをコピー

2. **Project IDとOrg IDの取得**
   - Vercelダッシュボード → プロジェクト → Settings → General
   - Project IDをコピー
   - Organization IDはVercelのURLから確認

3. **GitHub Secretsの設定**
   - リポジトリ → Settings → Secrets and variables → Actions
   - 以下を追加:
     ```
     VERCEL_TOKEN: (Vercelトークン)
     VERCEL_ORG_ID: (Organization ID)
     VERCEL_PROJECT_ID: (Project ID)
     ```

### 方法2: Vercel GitHubインテグレーション（簡単）

1. **Vercelダッシュボードで設定**
   - プロジェクト → Settings → Git
   - GitHubリポジトリを接続
   - Production Branchを`master`に設定

2. **自動デプロイの有効化**
   - 「Deploy Hooks」で自動デプロイを有効化
   - pushすると自動的にデプロイされる

3. **GitHub Actionsファイルの削除**
   ```bash
   rm .github/workflows/deploy.yml
   git add -A
   git commit -m "Remove GitHub Actions deploy workflow"
   git push origin master
   ```

## 即座にデプロイする方法

Vercel CLIを使用:
```bash
cd frontend
npx vercel --prod
```

または、Vercelダッシュボードから手動でデプロイ:
1. Vercelダッシュボード → プロジェクト
2. 「Redeploy」ボタンをクリック
3. 「Use existing Build Cache」のチェックを外す
4. 「Redeploy」をクリック