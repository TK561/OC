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

      {/* Model Details */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-sm font-medium text-blue-900 mb-3">🔍 それぞれの特徴</h3>
        <div className="space-y-3 text-sm">
          <div className="bg-white rounded p-3 border border-blue-100">
            <div className="flex items-center mb-2">
              <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium mr-3">高精度</span>
              <div>
                <span className="font-medium text-gray-900">最新の高性能モデル</span>
                <span className="text-xs text-gray-500 ml-2">(Intel/dpt-large)</span>
              </div>
            </div>
            <p className="text-gray-700 leading-relaxed">
              <strong>DPT (Dense Prediction Transformer)モデルを使用。</strong>
              このモデルはIntelが開発した最新の深度推定技術で、人の髪の毛、葉っぱの縁、細かい物体の境界なども非常に精密に判別します。
              画像を細かいパーツに分けて解析し、各部分の関係性を理解することで、非常に正確な深度情報を提供します。
            </p>
          </div>
          
          <div className="bg-white rounded p-3 border border-blue-100">
            <div className="flex items-center mb-2">
              <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-medium mr-3">高速</span>
              <div>
                <span className="font-medium text-gray-900">結果をすぐに確認</span>
                <span className="text-xs text-gray-500 ml-2">(Intel/dpt-hybrid-midas)</span>
              </div>
            </div>
            <p className="text-gray-700 leading-relaxed">
              <strong>MiDaS (Mixed Dataset Training)モデルを使用。</strong>
              このモデルはさまざまな種類の画像データで学習されており、効率的な処理で高速に結果を出します。
              大きな画像でも数秒で処理が完了し、「とりあえず結果を見てみたい」という方に最適です。品質と速度のバランスがとれた実用的なモデルです。
            </p>
          </div>
          
          <div className="bg-white rounded p-3 border border-blue-100">
            <div className="flex items-center mb-2">
              <span className="bg-purple-100 text-purple-800 px-2 py-1 rounded-full text-xs font-medium mr-3">汎用</span>
              <div>
                <span className="font-medium text-gray-900">どんな写真でも安心</span>
                <span className="text-xs text-gray-500 ml-2">(LiheYoung/depth-anything-small)</span>
              </div>
            </div>
            <p className="text-gray-700 leading-relaxed">
              <strong>Depth Anythingモデルを使用。</strong>
              このモデルは1400万枚もの大量の写真で学習された「万能型」のモデルです。
              人物写真、風景写真、室内・室外、日中・夜間など、あらゆるシーンで安定した結果を提供します。
              「どのモデルを選べばいいかわからない」方は、まずこのモデルからお試しください。
            </p>
          </div>
        </div>
      </div>

    </div>
  )
}
