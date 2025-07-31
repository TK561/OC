import { useState } from 'react'
import { DepthEstimationResponse } from '@/shared/types'

interface DepthViewerProps {
  depthResult: DepthEstimationResponse | null
  isProcessing: boolean
}

export default function DepthViewer({ depthResult, isProcessing }: DepthViewerProps) {
  const [showComparison, setShowComparison] = useState(false)

  const handleDownload = async (url: string, filename: string) => {
    try {
      const response = await fetch(url)
      const blob = await response.blob()
      const downloadUrl = window.URL.createObjectURL(blob)
      
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      window.URL.revokeObjectURL(downloadUrl)
    } catch (error) {
      console.error('Download failed:', error)
      alert('ダウンロードに失敗しました')
    }
  }

  if (isProcessing) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-depth-600 mx-auto mb-4"></div>
          <p className="text-lg font-medium text-gray-900">深度推定処理中...</p>
          <p className="text-sm text-gray-600 mt-2">
            画像サイズによって数秒〜数十秒かかる場合があります
          </p>
        </div>
      </div>
    )
  }

  if (!depthResult) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center text-gray-500">
          <svg
            className="mx-auto h-12 w-12 text-gray-400 mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          <p className="text-lg">深度マップはまだ生成されていません</p>
          <p className="text-sm mt-2">
            画像をアップロードして「深度推定実行」ボタンを押してください
          </p>
        </div>
      </div>
    )
  }

  // Check if URL is already a data URL or full URL
  const getImageUrl = (url: string) => {
    if (!url) {
      return ''
    }
    // Return as-is for data URLs, HTTP URLs, and blob URLs
    if (url.startsWith('data:') || url.startsWith('http') || url.startsWith('blob:') || url.startsWith('/samples/')) {
      return url
    }
    const baseUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
    // Ensure URL starts with /
    const path = url.startsWith('/') ? url : `/${url}`
    const fullUrl = `${baseUrl}${path}`
    return fullUrl
  }

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <label htmlFor="comparison-toggle" className="flex items-center space-x-2">
            <input
              id="comparison-toggle"
              name="comparison-toggle"
              type="checkbox"
              checked={showComparison}
              onChange={(e) => setShowComparison(e.target.checked)}
              className="rounded border-gray-300 text-depth-600 focus:ring-depth-500"
            />
            <span className="text-sm font-medium">比較表示</span>
          </label>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => handleDownload(getImageUrl(depthResult.depthMapUrl || ''), 'depth_map.png')}
            className="btn-secondary text-sm"
          >
            💾 深度マップ保存
          </button>
        </div>
      </div>

      {/* Image Display */}
      <div className="bg-gray-100 rounded-lg p-4">
        {showComparison ? (
          <div className="grid grid-cols-2 gap-4">
            {/* Original Image */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-gray-700">元画像</h3>
              <div className="bg-white rounded border overflow-hidden flex items-center justify-center min-h-48">
                {depthResult.originalUrl ? (
                  <img
                    key={`original-${depthResult.originalUrl}`}
                    src={getImageUrl(depthResult.originalUrl)}
                    alt="Original"
                    className="max-w-full max-h-full object-contain"
                    onError={(e) => {
                      console.error('Failed to load original image:', depthResult.originalUrl)
                      const errorUrl = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjNmNGY2Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiBmaWxsPSIjNjc3NDg5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+Loading Error</text></svg>'
                      if (e.currentTarget.src !== errorUrl) {
                        e.currentTarget.src = errorUrl
                      }
                    }}
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-400">
                    No Image
                  </div>
                )}
              </div>
            </div>
            
            {/* Depth Map */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-gray-700">深度マップ</h3>
              <div className="bg-white rounded border overflow-hidden flex items-center justify-center min-h-48">
                {depthResult.depthMapUrl ? (
                  <img
                    key={`depth-${depthResult.depthMapUrl}`}
                    src={getImageUrl(depthResult.depthMapUrl)}
                    alt="Depth Map"
                    className="max-w-full max-h-full object-contain"
                    onError={(e) => {
                      console.error('Failed to load depth map:', depthResult.depthMapUrl)
                      const errorUrl = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjNmNGY2Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiBmaWxsPSIjNjc3NDg5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+Loading Error</text></svg>'
                      if (e.currentTarget.src !== errorUrl) {
                        e.currentTarget.src = errorUrl
                      }
                    }}
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-400">
                    No Depth Map
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-white rounded border overflow-hidden flex items-center justify-center min-h-96">
            {depthResult.depthMapUrl ? (
              <img
                key={`single-depth-${depthResult.depthMapUrl}`}
                src={getImageUrl(depthResult.depthMapUrl)}
                alt="Depth Map"
                className="max-w-full max-h-full object-contain"
                onError={(e) => {
                  console.error('Failed to load depth map (single view):', depthResult.depthMapUrl)
                  const errorUrl = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjNmNGY2Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiBmaWxsPSIjNjc3NDg5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+Loading Error</text></svg>'
                  if (e.currentTarget.src !== errorUrl) {
                    e.currentTarget.src = errorUrl
                  }
                }}
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-gray-400">
                No Depth Map
              </div>
            )}
          </div>
        )}
      </div>

      {/* Information */}
      <div className="bg-gray-50 rounded-lg p-4 space-y-2">
        <h3 className="text-sm font-medium text-gray-900">処理情報</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">使用モデル:</span>
            <span className="ml-2 font-medium">{depthResult.model || 'Unknown'}</span>
          </div>
          <div>
            <span className="text-gray-600">解像度:</span>
            <span className="ml-2 font-medium">{depthResult.resolution || 'Unknown'}</span>
          </div>
          {depthResult.note && (
            <div className="col-span-2">
              <span className="text-gray-600">備考:</span>
              <span className="ml-2 font-medium text-blue-600">{depthResult.note}</span>
            </div>
          )}
        </div>
      </div>

      {/* Technical Explanation */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-sm font-medium text-blue-900 mb-3">🔬 深度推定技術の理論</h3>
        <div className="space-y-3 text-sm">
          <div className="bg-white rounded p-3 border border-blue-100">
            <div className="flex items-center mb-2">
              <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium mr-3">高精度</span>
              <span className="font-medium text-gray-900">DPT (Dense Prediction Transformer)</span>
            </div>
            <p className="text-gray-700 leading-relaxed">
              <strong>Vision Transformerアーキテクチャを使用。</strong>画像をパッチに分割し、
              セルフアテンションメカニズムでグローバルな文脈を理解。高解像度の特徴抽出と精密な深度推定が可能です。
            </p>
          </div>
          
          <div className="bg-white rounded p-3 border border-blue-100">
            <div className="flex items-center mb-2">
              <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-medium mr-3">高速</span>
              <span className="font-medium text-gray-900">MiDaS (Mixed Dataset Training)</span>
            </div>
            <p className="text-gray-700 leading-relaxed">
              <strong>複数データセットでの混合学習。</strong>異なるスケールの深度データを統一し、
              効率的なCNNベースのエンコーダーで高速処理。相対的な深度関係を学習し、リアルタイム処理を実現。
            </p>
          </div>
          
          <div className="bg-white rounded p-3 border border-blue-100">
            <div className="flex items-center mb-2">
              <span className="bg-purple-100 text-purple-800 px-2 py-1 rounded-full text-xs font-medium mr-3">汎用</span>
              <span className="font-medium text-gray-900">Depth Anything (Foundation Model)</span>
            </div>
            <p className="text-gray-700 leading-relaxed">
              <strong>基盤モデルアプローチ。</strong>1400万枚の大規模データセットで事前学習。
              ラベル付きデータと未ラベルデータの組み合わせで、多様なシーンに対する汎化性能を獲得。
            </p>
          </div>
        </div>
      </div>

      {/* Color Map Legend */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-900 mb-3">🎨 深度カラーマップ</h3>
        <div className="flex items-center space-x-4">
          <div className="flex-1">
            <div className="h-4 bg-gradient-to-r from-white via-gray-500 to-black rounded"></div>
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>近い</span>
              <span>遠い</span>
            </div>
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          白色が最も近く、黒色が最も遠い距離を表します。グレーの濃さで距離の段階を表現しています。
        </p>
      </div>

      {/* Technical Details */}
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <h3 className="text-sm font-medium text-green-900 mb-3">⚙️ 技術的特徴</h3>
        <div className="space-y-2 text-sm text-green-800">
          <div className="flex items-start">
            <span className="text-green-600 mr-2">•</span>
            <span><strong>単眼深度推定</strong>：1枚の画像からシーンの3D構造を推定</span>
          </div>
          <div className="flex items-start">
            <span className="text-green-600 mr-2">•</span>
            <span><strong>相対深度</strong>：絶対距離ではなく、相対的な近さを表現</span>
          </div>
          <div className="flex items-start">
            <span className="text-green-600 mr-2">•</span>
            <span><strong>ニューラルネットワーク</strong>：深層学習で訓練されたモデルを使用</span>
          </div>
          <div className="flex items-start">
            <span className="text-green-600 mr-2">•</span>
            <span><strong>ピクセル単位</strong>：各ピクセルに深度値を割り当ててマップを生成</span>
          </div>
        </div>
      </div>
    </div>
  )
}
