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
            <div className="h-4 bg-gradient-to-r from-purple-600 via-blue-500 via-green-500 via-yellow-500 to-red-500 rounded"></div>
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
        <h3 className="text-sm font-medium text-blue-900 mb-3">📋 モデル一覧</h3>
        <div className="space-y-3 text-sm">
          <div className="bg-white rounded p-3 border border-blue-100">
            <div className="flex items-center mb-2">
              <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium mr-3">高精度</span>
              <div>
                <span className="font-medium text-gray-900 mr-2">DPT-Large</span>
                <span className="text-xs text-gray-500 mr-2">(Intel/dpt-large)</span>
                <a 
                  href="https://huggingface.co/Intel/dpt-large" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-xs text-blue-500 hover:text-blue-700 underline"
                >
                  🔗 Hugging Face
                </a>
              </div>
            </div>
            <p className="text-gray-700 leading-relaxed">
              <strong>DPT (Dense Prediction Transformer)モデルを使用。</strong>
              このモデルはIntelが開発した最新の深度推定技術で、今までの技術では難しかった細かいディテールを正確に捉えます。
            </p>
            <div className="mt-3 p-3 bg-gray-50 rounded text-sm">
              <p className="font-medium text-gray-800 mb-2">🔬 使用技術</p>
              <p className="text-gray-600 mb-3">
                <strong>Vision Transformer (ViT)</strong> - 自然言語処理のTransformerを画像解析に応用した革新技術。従来のCNN（畳み込みニューラルネットワーク）とは異なり、画像を16×16ピクセルの小さなパッチに分割し、各パッチを「単語」として扱います。
              </p>
              <p className="text-gray-600 mb-3">
                <strong>Self-Attention機構</strong> - 各パッチが画像全体の他のすべてのパッチとの関係を同時に計算。例えば、人の顔のパッチが髪の毛や背景のパッチとどう関連するかを理解し、文脈に基づいた深度推定を実現。
              </p>
              <p className="text-gray-600 mb-3">
                <strong>Dense Prediction構造</strong> - 画像の各ピクセルに対して高精度な深度値を予測。多層のTransformerエンコーダーで特徴を抽出し、デコーダーで深度マップを生成します。
              </p>
              <p className="text-gray-600 mb-2">
                <strong>📊 学習データ</strong> - 約130万枚の画像で学習。NYU Depth V2、KITTI、Cityscapesなど高品質な深度データセットを使用し、高精度な境界検出能力を獲得。
              </p>
              <p className="font-medium text-gray-800 mb-2">🎯 検出方法</p>
              <p className="text-gray-600">
                <strong>マルチスケール境界線検出</strong> - 異なる解像度レベルで境界を検出し、細かな髪の毛から建物の大きな輪郭まで階層的に処理。グローバルコンテキストと局所的ディテールを統合した高精度な境界線推定を実現。
              </p>
            </div>
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
                <span className="font-medium text-gray-900 mr-2">MiDaS</span>
                <span className="text-xs text-gray-500 mr-2">(Intel/dpt-hybrid-midas)</span>
                <a 
                  href="https://huggingface.co/Intel/dpt-hybrid-midas" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-xs text-blue-500 hover:text-blue-700 underline"
                >
                  🔗 Hugging Face
                </a>
              </div>
            </div>
            <p className="text-gray-700 leading-relaxed">
              <strong>MiDaS (Mixed Dataset Training)モデルを使用。</strong>
              このモデルは複数の異なるデータセットで同時に学習された独特の技術で、速度と品質のバランスを重視した実用モデルです。
            </p>
            <div className="mt-3 p-3 bg-gray-50 rounded text-sm">
              <p className="font-medium text-gray-800 mb-2">🔬 使用技術</p>
              <p className="text-gray-600 mb-3">
                <strong>CNN + Transformer ハイブリッドアーキテクチャ</strong> - ResNetやEfficientNetなどのCNNバックボーンで局所的な特徴（エッジ、テクスチャ）を抽出し、Transformerで長距離依存関係を捉える2段構成。CNNの計算効率とTransformerの表現力を両立。
              </p>
              <p className="text-gray-600 mb-3">
                <strong>混合データセット学習</strong> - 屋内・屋外・映画など異なる特性を持つ12種類のデータセットで同時学習。各データセットの深度範囲や分布の違いを正規化し、統一的な深度表現を獲得。
              </p>
              <p className="text-gray-600 mb-3">
                <strong>逆深度パラメータ化</strong> - 通常の深度値ではなく逆深度（1/深度）を予測することで、遠距離の深度推定精度を向上。無限遠での数値安定性を確保し、より自然な深度勾配を実現。
              </p>
              <p className="text-gray-600 mb-2">
                <strong>📊 学習データ</strong> - 12種類のデータセット、約500万枚の画像で学習。屋内・屋外・映画・ゲームなど多様なシーンから滑らかな深度変化パターンを学習。
              </p>
              <p className="font-medium text-gray-800 mb-2">🎯 検出方法</p>
              <p className="text-gray-600">
                <strong>適応的スケール融合</strong> - 複数解像度のピラミッド構造で特徴を抽出し、注意機構により各スケールの重要度を動的に調整。近景の細部と遠景の滑らかさを適応的にバランス調整。
              </p>
            </div>
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
                <span className="font-medium text-gray-900 mr-2">Depth Anything</span>
                <span className="text-xs text-gray-500 mr-2">(LiheYoung/depth-anything-small)</span>
                <a 
                  href="https://huggingface.co/LiheYoung/depth-anything-small-hf" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-xs text-blue-500 hover:text-blue-700 underline"
                >
                  🔗 Hugging Face
                </a>
              </div>
            </div>
            <p className="text-gray-700 leading-relaxed">
              <strong>Depth Anythingモデルを使用。</strong>
              このモデルは1400万枚もの大量の写真で学習された「基盤モデル」で、あらゆるシチュエーションに対応できる汎用性が最大の特徴です。
            </p>
            <div className="mt-3 p-3 bg-gray-50 rounded text-sm">
              <p className="font-medium text-gray-800 mb-2">🔬 使用技術</p>
              <p className="text-gray-600 mb-3">
                <strong>Foundation Model アーキテクチャ</strong> - GPTやBERTと同様の大規模Transformerベース。DINOv2やCLIPなどの事前学習済み視覚エンコーダーを活用し、1400万枚の未ラベル画像から自己教師あり学習で汎用的な視覚表現を獲得。
              </p>
              <p className="text-gray-600 mb-3">
                <strong>スケール不変深度学習</strong> - 絶対深度ではなく相対深度関係を学習。アフィン不変損失関数により、カメラパラメータに依存しない汎用的な深度推定を実現。任意のスケールの画像に対応可能。
              </p>
              <p className="text-gray-600 mb-3">
                <strong>マルチドメイン適応</strong> - 実写、CG、絵画、スケッチなど多様な画像ドメインで学習。ドメイン敵対的学習により、画風や撮影条件の違いに頑健な特徴表現を獲得し、未知ドメインへの汎化性能を向上。
              </p>
              <p className="text-gray-600 mb-2">
                <strong>📊 学習データ</strong> - 1400万枚の未ラベル画像で自己教師学習。実写・CG・絵画・スケッチなど多様なドメイン、Hypersim・Virtual KITTI・NYU・KITTIなど62のデータセットを統合。
              </p>
              <p className="font-medium text-gray-800 mb-2">🎯 検出方法</p>
              <p className="text-gray-600">
                <strong>コンテキスト認識深度推定</strong> - 画像全体の意味的コンテキスト（室内/屋外、昼/夜など）を理解し、シーンタイプに応じた適応的な深度推定戦略を選択。局所的なテクスチャ情報と大域的な構造情報を統合。
              </p>
            </div>
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
