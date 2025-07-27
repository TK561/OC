// モックAPI: バックエンドが利用できない場合の代替
export async function createMockDepthMap(imageDataUrl: string): Promise<string> {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Canvas でモック深度マップを生成
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      // 元画像を読み込み
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        
        // グラデーション深度マップを生成
        const gradient = ctx!.createLinearGradient(0, 0, 0, img.height);
        gradient.addColorStop(0, 'rgb(0, 0, 255)');    // 青 (遠い)
        gradient.addColorStop(0.5, 'rgb(128, 0, 128)'); // 紫 (中間)
        gradient.addColorStop(1, 'rgb(255, 0, 0)');     // 赤 (近い)
        
        ctx!.fillStyle = gradient;
        ctx!.fillRect(0, 0, canvas.width, canvas.height);
        
        // データURLとして返す
        resolve(canvas.toDataURL('image/png'));
      };
      img.src = imageDataUrl;
    }, 1000); // 1秒の処理時間をシミュレート
  });
}

export const MOCK_API_ENABLED = true;