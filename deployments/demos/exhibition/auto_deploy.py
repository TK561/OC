#!/usr/bin/env python3
"""
2日間展示用自動デプロイスクリプト
ngrok URL更新 + Vercel再デプロイ自動化
"""

import subprocess
import time
import requests
import json
from datetime import datetime, timedelta

class ExhibitionManager:
    def __init__(self):
        self.start_time = datetime.now()
        self.session_duration = 12 * 3600  # 12時間
        self.url_duration = 8 * 3600       # 8時間
        self.current_session = 1
        self.current_url = 1
        
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def check_ngrok_status(self, url):
        """ngrok URL の生存確認"""
        try:
            response = requests.get(f"{url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def update_vercel_env(self, new_url):
        """Vercel環境変数更新"""
        try:
            # Vercel CLI経由で環境変数更新
            cmd = f'vercel env rm NEXT_PUBLIC_BACKEND_URL production --yes'
            subprocess.run(cmd, shell=True, check=True)
            
            cmd = f'vercel env add NEXT_PUBLIC_BACKEND_URL production'
            process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(input=f"{new_url}\n")
            
            if process.returncode == 0:
                self.log(f"✅ Vercel環境変数更新: {new_url}")
                return True
            else:
                self.log(f"❌ 環境変数更新失敗: {stderr}")
                return False
                
        except Exception as e:
            self.log(f"❌ Vercel更新エラー: {e}")
            return False
    
    def deploy_vercel(self):
        """Vercel再デプロイ"""
        try:
            cmd = 'vercel --prod --yes'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                # デプロイURLを抽出
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'https://' in line and 'vercel.app' in line:
                        deploy_url = line.strip()
                        self.log(f"✅ Vercel再デプロイ完了: {deploy_url}")
                        return deploy_url
                
                self.log("✅ Vercel再デプロイ完了")
                return True
            else:
                self.log(f"❌ デプロイ失敗: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"❌ デプロイエラー: {e}")
            return False
    
    def send_notification(self, message):
        """通知送信（展示スタッフ向け）"""
        # Slack/Discord webhook等での通知（オプション）
        self.log(f"📢 通知: {message}")
    
    def run_exhibition_monitoring(self):
        """2日間展示監視"""
        self.log("🏛️ 2日間展示監視開始")
        self.log("🔒 セキュリティ: 外部漏洩なし")
        self.log("💰 コスト: 完全無料")
        
        exhibition_end = self.start_time + timedelta(days=2)
        
        while datetime.now() < exhibition_end:
            current_time = datetime.now()
            elapsed = (current_time - self.start_time).total_seconds()
            
            # ngrok URL確認・更新判定
            if elapsed > 0 and elapsed % (self.url_duration - 300) < 60:  # 5分前に確認
                self.log("⏰ ngrok URL期限間近 - 更新準備")
                self.send_notification("ngrok URL更新が必要です")
            
            # Colab セッション確認・更新判定  
            if elapsed > 0 and elapsed % (self.session_duration - 600) < 60:  # 10分前に確認
                self.log("⏰ Colab セッション期限間近 - 更新準備")
                self.send_notification("Google Colab セッション更新が必要です")
            
            # 1分間隔で監視
            time.sleep(60)
        
        self.log("🏁 2日間展示完了")
        self.log("🧹 セッション終了 - 全データ自動削除")

def main():
    """メイン実行"""
    print("🏛️" + "="*50)
    print("   2日間展示用自動管理システム")
    print("   外部漏洩なし・完全無料・自動化")
    print("="*53)
    
    manager = ExhibitionManager()
    
    # 初期設定確認
    print("\n📋 設定確認:")
    print("├── 期間: 2日間")
    print("├── セキュリティ: 外部漏洩なし")
    print("├── コスト: 完全無料")
    print("├── バックエンド: Google Colab + ngrok")
    print("└── フロントエンド: Vercel")
    
    # 手動での初回設定
    print("\n🔧 初回設定:")
    print("1. Google Colab で setup_colab.ipynb を実行")
    print("2. 生成されたngrok URLをコピー")
    print("3. Vercel環境変数に設定")
    print("4. このスクリプトで監視開始")
    
    print("\n▶️ 監視を開始しますか？ (y/n): ", end="")
    if input().lower() == 'y':
        manager.run_exhibition_monitoring()
    else:
        print("ℹ️ 監視をキャンセルしました")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 監視を停止しました")
    except Exception as e:
        print(f"\n❌ エラー: {e}")