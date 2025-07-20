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

  const baseUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <label className="flex items-center space-x-2">
            <input
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
            onClick={() => handleDownload(`${baseUrl}${depthResult.depthMapUrl}`, 'depth_map.png')}
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
              <div className="aspect-square bg-white rounded border overflow-hidden">
                <img
                  src={`${baseUrl}${depthResult.originalUrl}`}
                  alt="Original"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
            
            {/* Depth Map */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-gray-700">深度マップ</h3>
              <div className="aspect-square bg-white rounded border overflow-hidden">
                <img
                  src={`${baseUrl}${depthResult.depthMapUrl}`}
                  alt="Depth Map"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="aspect-video bg-white rounded border overflow-hidden">
            <img
              src={`${baseUrl}${depthResult.depthMapUrl}`}
              alt="Depth Map"
              className="w-full h-full object-contain"
            />
          </div>
        )}
      </div>

      {/* Information */}
      <div className="bg-gray-50 rounded-lg p-4 space-y-2">
        <h3 className="text-sm font-medium text-gray-900">処理情報</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">使用モデル:</span>
            <span className="ml-2 font-medium">{depthResult.modelUsed}</span>
          </div>
          <div>
            <span className="text-gray-600">解像度:</span>
            <span className="ml-2 font-medium">{depthResult.resolution}</span>
          </div>
        </div>
      </div>

      {/* Color Map Legend */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-900 mb-3">深度カラーマップ</h3>
        <div className="flex items-center space-x-4">
          <div className="flex-1">
            <div className="h-4 bg-gradient-to-r from-purple-600 via-blue-500 via-green-500 via-yellow-500 to-red-500 rounded"></div>
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>近い</span>
              <span>遠い</span>
            </div>
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          紫色が最も近く、赤色が最も遠い距離を表します
        </p>
      </div>
    </div>
  )
}