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

      {/* Feature Overview */}
      <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4">
        <h3 className="text-sm font-medium text-emerald-900 mb-3">🎆 深度推定機能でできること</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="space-y-2">
            <div className="flex items-start">
              <span className="text-emerald-600 mr-2">📷</span>
              <div>
                <strong className="text-emerald-800">単一画像から深度情報を抽出</strong>
                <p className="text-emerald-700 text-xs mt-1">特別な機材や複数のカメラがなくても、1枚の写真だけで立体構造を解析</p>
              </div>
            </div>
            <div className="flex items-start">
              <span className="text-emerald-600 mr-2">🎨</span>
              <div>
                <strong className="text-emerald-800">視覚的な深度マップを生成</strong>
                <p className="text-emerald-700 text-xs mt-1">白（近）から黒（遠）のグラデーションで、直感的に距離関係を理解</p>
              </div>
            </div>
            <div className="flex items-start">
              <span className="text-emerald-600 mr-2">🔄</span>
              <div>
                <strong className="text-emerald-800">元画像との比較表示</strong>
                <p className="text-emerald-700 text-xs mt-1">チェックボックで元画像と深度マップを並べて表示、結果を簡単に検証</p>
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex items-start">
              <span className="text-emerald-600 mr-2">💾</span>
              <div>
                <strong className="text-emerald-800">高品質な結果を保存</strong>
                <p className="text-emerald-700 text-xs mt-1">生成された深度マップをPNG形式でダウンロード、他のアプリで活用可能</p>
              </div>
            </div>
            <div className="flex items-start">
              <span className="text-emerald-600 mr-2">⚙️</span>
              <div>
                <strong className="text-emerald-800">3種類の高性能モデル</strong>
                <p className="text-emerald-700 text-xs mt-1">精度・速度・汎用性の違うモデルから、目的に合わせて選択</p>
              </div>
            </div>
            <div className="flex items-start">
              <span className="text-emerald-600 mr-2">🚀</span>
              <div>
                <strong className="text-emerald-800">クラウドベースで高速処理</strong>
                <p className="text-emerald-700 text-xs mt-1">強力なGPUサーバーで処理するため、個人パソコンでは難しい高品質解析を実現</p>
              </div>
            </div>
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

      {/* Model Details */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-sm font-medium text-blue-900 mb-3">🔍 それぞれの特徴</h3>
        <div className="space-y-3 text-sm">
          <div className="bg-white rounded p-3 border border-blue-100">
            <div className="flex items-center mb-2">
              <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium mr-3">高精度</span>
              <div>
                <span className="text-xs text-gray-500">(Intel/dpt-large)</span>
              </div>
            </div>
            <p className="text-gray-700 leading-relaxed">
              <strong>DPT (Dense Prediction Transformer)モデルを使用。</strong>
              このモデルはIntelが開発した最新の深度推定技術で、今までの技術では難しかった細かいディテールを正確に捉えます。
            </p>
            <div className="mt-3 space-y-2 text-sm text-gray-600">
              <div className="flex items-start">
                <span className="text-blue-500 mr-2">✓</span>
                <span><strong>精密な境界検出:</strong> 人の髪の毛、メガネのフレーム、葉っぱの縁など、微細な部分も正確に認識</span>
              </div>
              <div className="flex items-start">
                <span className="text-blue-500 mr-2">✓</span>
                <span><strong>高解像度処理:</strong> 大きな画像でも細かい部分までしっかりと解析し、プロ品質の結果を出力</span>
              </div>
              <div className="flex items-start">
                <span className="text-blue-500 mr-2">✓</span>
                <span><strong>複雑なシーンに強い:</strong> 入り組んだ物体、透明な素材、影のあるシーンでも正確に処理</span>
              </div>
              <div className="flex items-start">
                <span className="text-orange-500 mr-2">⚠</span>
                <span><strong>処理時間:</strong> 精密な解析のため、他のモデルよりも数十秒多くかかる場合があります</span>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded p-3 border border-blue-100">
            <div className="flex items-center mb-2">
              <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-medium mr-3">高速</span>
              <div>
                <span className="text-xs text-gray-500">(Intel/dpt-hybrid-midas)</span>
              </div>
            </div>
            <p className="text-gray-700 leading-relaxed">
              <strong>MiDaS (Mixed Dataset Training)モデルを使用。</strong>
              このモデルは複数の異なるデータセットで同時に学習された独特の技術で、速度と品質のバランスを重視した実用モデルです。
            </p>
            <div className="mt-3 space-y-2 text-sm text-gray-600">
              <div className="flex items-start">
                <span className="text-yellow-500 mr-2">⚡</span>
                <span><strong>高速処理:</strong> 3MPの画像なら約5-10秒、大きな画像でも数十秒で処理完了</span>
              </div>
              <div className="flex items-start">
                <span className="text-yellow-500 mr-2">✓</span>
                <span><strong>リアルタイム体験:</strong> 待ち時間が短く、すぐに結果を確認できるためストレスフリー</span>
              </div>
              <div className="flex items-start">
                <span className="text-yellow-500 mr-2">✓</span>
                <span><strong>バランス型:</strong> 品質を犠牲にしすぎず、日常的な用途に十分な精度を維持</span>
              </div>
              <div className="flex items-start">
                <span className="text-yellow-500 mr-2">✓</span>
                <span><strong>メモリ効率:</strong> サーバーリソースを節約し、同時に複数のユーザーが使用しても安定</span>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded p-3 border border-blue-100">
            <div className="flex items-center mb-2">
              <span className="bg-purple-100 text-purple-800 px-2 py-1 rounded-full text-xs font-medium mr-3">汎用</span>
              <div>
                <span className="text-xs text-gray-500">(LiheYoung/depth-anything-small)</span>
              </div>
            </div>
            <p className="text-gray-700 leading-relaxed">
              <strong>Depth Anythingモデルを使用。</strong>
              このモデルは1400万枚もの大量の写真で学習された「基盤モデル」で、あらゆるシチュエーションに対応できる汎用性が最大の特徴です。
            </p>
            <div className="mt-3 space-y-2 text-sm text-gray-600">
              <div className="flex items-start">
                <span className="text-purple-500 mr-2">🌍</span>
                <span><strong>圧倒的なデータ量:</strong> 1400万枚の多様な写真で学習し、見たことのないシーンでも安定して動作</span>
              </div>
              <div className="flex items-start">
                <span className="text-purple-500 mr-2">✓</span>
                <span><strong>シーンを選ばない:</strong> 人物・風景・建物・動物・食べ物など、どんな被写体でも適切に処理</span>
              </div>
              <div className="flex items-start">
                <span className="text-purple-500 mr-2">✓</span>
                <span><strong>環境に左右されない:</strong> 室内・屋外、明るい・暗い、晴れ・曇り・雨などの条件変化に強い</span>
              </div>
              <div className="flex items-start">
                <span className="text-purple-500 mr-2">✓</span>
                <span><strong>初心者に優しい:</strong> モデル選択に迷ったら、まずこちらでテストして結果を確認</span>
              </div>
              <div className="flex items-start">
                <span className="text-purple-500 mr-2">✓</span>
                <span><strong>安定性重視:</strong> 予想外の結果やエラーが起きにくい、信頼性の高いモデル</span>
              </div>
            </div>
          </div>
        </div>
      </div>

    </div>
  )
}
