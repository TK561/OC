import { useState, useRef, useEffect } from 'react'
import ImageUpload from '@/components/ImageUpload'
import DepthViewer from '@/components/DepthViewer'
import ThreeScene from '@/components/ThreeScene'
import ControlPanel from '@/components/ControlPanel'
import { DepthEstimationResponse, ViewerSettings, EdgeDepthProcessingResponse } from '@/shared/types'

export default function Home() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [depthResult, setDepthResult] = useState<DepthEstimationResponse | null>(null)
  const [depthResults, setDepthResults] = useState<{[key: string]: DepthEstimationResponse}>({})
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0)
  const [processingStatus, setProcessingStatus] = useState('')
  const [selectedModel, setSelectedModel] = useState('Intel/dpt-hybrid-midas')
  const [compareMode, setCompareMode] = useState(false)
  const [showModelInfo, setShowModelInfo] = useState(false)
  const [viewerSettings, setViewerSettings] = useState<ViewerSettings>({
    colorMap: 'viridis',
    pointSize: 0.50,
    backgroundColor: '#000000',
    showAxes: true
  })
  const [activeTab, setActiveTab] = useState<'original' | 'depth' | '3d'>('original')

  const handleImageUpload = (imageUrl: string) => {
    setUploadedImage(imageUrl)
    setDepthResult(null)
    setDepthResults({})
    setCompareMode(false)
    setProcessingProgress(0)
    setProcessingStatus('')
  }

  const processEdgeDepthEnhancement = async (originalImageUrl: string, model: string): Promise<DepthEstimationResponse | null> => {
    try {
      console.log(`Starting edge-depth processing for model: ${model}`)
      
      // 画像をBlobに変換
      const response = await fetch(originalImageUrl)
      const blob = await response.blob()
      
      // FormData作成
      const formData = new FormData()
      formData.append('file', blob, 'image.jpg')
      formData.append('model', model)
      formData.append('edge_low_threshold', '50')
      formData.append('edge_high_threshold', '150')
      formData.append('invert_depth', 'true')
      formData.append('depth_gamma', '1.0')
      formData.append('depth_contrast', '1.0')
      formData.append('composition_mode', 'multiply')
      formData.append('post_gamma', '1.0')
      formData.append('post_blur', '0.0')
      
      const edgeResponse = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/depth-edge-processing`, {
        method: 'POST',
        body: formData,
      })
      
      if (!edgeResponse.ok) {
        console.error(`Edge processing failed for ${model}:`, edgeResponse.status)
        return null
      }
      
      const edgeResult: EdgeDepthProcessingResponse = await edgeResponse.json()
      console.log(`Edge processing completed for ${model}:`, edgeResult)
      
      if (!edgeResult.success) {
        console.error(`Edge processing returned failure for ${model}`)
        return null
      }
      
      // DepthEstimationResponse形式に変換（最終処理画像を使用）
      const enhancedResult: DepthEstimationResponse = {
        depthMapUrl: edgeResult.finalImageUrl, // エッジ処理済みの最終画像
        originalUrl: edgeResult.originalUrl,
        success: true,
        model: edgeResult.processing_info.model,
        resolution: edgeResult.resolution,
        note: 'Enhanced with edge detection and depth processing',
        algorithms: ['Edge Detection', 'Depth Enhancement', 'Gradient Processing'],
        implementation: 'Custom Edge-Depth Pipeline',
        features: ['Canny Edge Detection', 'Depth Inversion', 'Mask Composition']
      }
      
      return enhancedResult
      
    } catch (error) {
      console.error(`Edge processing error for ${model}:`, error)
      return null
    }
  }

  const handleDepthEstimation = async () => {
    if (!uploadedImage) return

    setIsProcessing(true)
    setProcessingProgress(0)
    setProcessingStatus('画像を準備中...')
    try {
      // Convert image to base64 data URL if needed
      let imageDataUrl = uploadedImage
      if (uploadedImage.startsWith('blob:') || !uploadedImage.startsWith('data:')) {
        const response = await fetch(uploadedImage)
        const blob = await response.blob()
        const reader = new FileReader()
        imageDataUrl = await new Promise<string>((resolve) => {
          reader.onload = () => resolve(reader.result as string)
          reader.readAsDataURL(blob)
        })
      }
      
      console.log('Original image data URL type:', imageDataUrl.substring(0, 50))

      console.log('Backend URL:', process.env.NEXT_PUBLIC_BACKEND_URL)
      
      setProcessingProgress(25)
      setProcessingStatus('深度推定を実行中...')
      
      // Use Railway API only - no fallback
      console.log('Using Railway API exclusively...')
      console.log('Backend URL:', process.env.NEXT_PUBLIC_BACKEND_URL)
      
      if (!process.env.NEXT_PUBLIC_BACKEND_URL) {
        throw new Error('Backend URL is not configured')
      }
      
      // ファイルアップロード用のFormData作成
      const formData = new FormData()
      
      // Base64をBlobに変換（元のMIMEタイプを保持）
      const mimeMatch = imageDataUrl.match(/data:([^;]+);/)
      const mimeType = mimeMatch ? mimeMatch[1] : 'image/jpeg'
      console.log('Original image MIME type:', mimeType)
      
      const byteCharacters = atob(imageDataUrl.split(',')[1])
      const byteNumbers = new Array(byteCharacters.length)
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i)
      }
      const byteArray = new Uint8Array(byteNumbers)
      const blob = new Blob([byteArray], { type: mimeType })
      
      formData.append('file', blob, 'image.jpg')
      formData.append('model', selectedModel)
      
      // 30秒タイムアウト設定（深度推定は時間がかかる場合がある）
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 30000)
      
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/predict`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      })
      
      clearTimeout(timeoutId)
      setProcessingProgress(50)
      setProcessingStatus('深度推定完了、エッジ処理中...')

      if (!response.ok) {
        const errorText = await response.text()
        const errorMessage = `Backend API failed: ${response.status} ${response.statusText}. Details: ${errorText}`
        console.error('Railway API failed:', errorMessage)
        throw new Error(errorMessage)
      }
      
      const result = await response.json()
      console.log('Railway API Response:', result)
      
      if (!result.success || !result.depthMapUrl) {
        throw new Error(`Backend API returned invalid result: ${JSON.stringify(result)}`)
      }
      
      // エッジ処理を実行
      setProcessingProgress(75)
      setProcessingStatus('エッジ検出+深度処理実行中...')
      
      const enhancedResult = await processEdgeDepthEnhancement(uploadedImage, selectedModel)
      
      if (enhancedResult) {
        // エッジ処理済み結果を使用
        const finalResult = {
          ...enhancedResult,
          pointcloudData: result.pointcloudData // 元の3Dデータは保持
        }
        
        console.log('Enhanced result with edge processing:', finalResult)
        setDepthResult(finalResult)
        setDepthResults(prev => ({...prev, [selectedModel]: finalResult}))
        setProcessingProgress(100)
        setProcessingStatus('エッジ処理完了！')
      } else {
        // エッジ処理に失敗した場合は元の結果を使用
        console.warn('Edge processing failed, using original depth result')
        const newResult = {
          depthMapUrl: result.depthMapUrl,
          originalUrl: result.originalUrl || uploadedImage,
          success: true,
          model: result.model || selectedModel || 'Railway-API',
          resolution: result.resolution || 'unknown',
          note: result.note,
          algorithms: result.algorithms,
          implementation: result.implementation,
          features: result.features,
          pointcloudData: result.pointcloudData
        }
        
        setDepthResult(newResult)
        setDepthResults(prev => ({...prev, [selectedModel]: newResult}))
        setProcessingProgress(100)
        setProcessingStatus('完了！')
      }
      
      setActiveTab('depth')
      console.log('✅ Depth estimation with edge enhancement completed!')
    } catch (error) {
      console.error('Depth estimation failed:', error)
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      alert(`深度推定に失敗しました。\n\nエラー詳細: ${errorMessage}\n\n画像形式やサイズを確認してもう一度お試しください。`)
    } finally {
      setTimeout(() => {
        setIsProcessing(false)
        setProcessingProgress(0)
        setProcessingStatus('')
      }, 1500) // 完了メッセージを1.5秒表示
    }
  }

  const handleCompareAllModels = async () => {
    if (!uploadedImage) return

    const models = [
      'Intel/dpt-hybrid-midas',
      'Intel/dpt-large', 
      'LiheYoung/depth-anything-small-hf'
    ]

    setIsProcessing(true)
    setProcessingProgress(0)
    setProcessingStatus('全モデル比較を開始...')
    const newResults: {[key: string]: DepthEstimationResponse} = {}

    try {
      for (let i = 0; i < models.length; i++) {
        const model = models[i]
        const modelName = model === 'Intel/dpt-hybrid-midas' ? 'MiDaS' :
                         model === 'Intel/dpt-large' ? 'DPT-Large' :
                         'DepthAnything'
        
        setProcessingProgress(Math.round((i / models.length) * 90))
        setProcessingStatus(`${modelName} で処理中... (${i + 1}/${models.length})`)
        
        console.log(`Processing with ${model}...`)
        
        // Convert image to base64 data URL if needed
        let imageDataUrl = uploadedImage
        if (uploadedImage.startsWith('blob:') || !uploadedImage.startsWith('data:')) {
          const response = await fetch(uploadedImage)
          const blob = await response.blob()
          const reader = new FileReader()
          imageDataUrl = await new Promise<string>((resolve) => {
            reader.onload = () => resolve(reader.result as string)
            reader.readAsDataURL(blob)
          })
        }

        try {
          // ファイルアップロード用のFormData作成
          const formData = new FormData()
          
          // Base64をBlobに変換（元のMIMEタイプを保持）
          const mimeMatch = imageDataUrl.match(/data:([^;]+);/)
          const mimeType = mimeMatch ? mimeMatch[1] : 'image/jpeg'
          console.log(`Model ${model} - Original image MIME type:`, mimeType)
          
          const byteCharacters = atob(imageDataUrl.split(',')[1])
          const byteNumbers = new Array(byteCharacters.length)
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i)
          }
          const byteArray = new Uint8Array(byteNumbers)
          const blob = new Blob([byteArray], { type: mimeType })
          
          formData.append('file', blob, 'image.jpg')
          formData.append('model', model)
          
          const controller = new AbortController()
          const timeoutId = setTimeout(() => controller.abort(), 45000)  // 比較モードはさらに長く
          
          const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/predict`, {
            method: 'POST',
            body: formData,
            signal: controller.signal
          })
          
          clearTimeout(timeoutId)

          if (response.ok) {
            const result = await response.json()
            console.log(`Compare mode - ${model} result:`, result)
            
            if (result.success && result.depthMapUrl) {
              // エッジ処理を実行
              const enhancedResult = await processEdgeDepthEnhancement(uploadedImage, model)
              
              if (enhancedResult) {
                // エッジ処理済み結果を使用
                const modelResult = {
                  ...enhancedResult,
                  pointcloudData: result.pointcloudData // 元の3Dデータは保持
                }
                console.log(`Compare mode - ${model} enhanced result:`, modelResult)
                newResults[model] = modelResult
              } else {
                // エッジ処理に失敗した場合は元の結果を使用
                const modelResult = {
                  depthMapUrl: result.depthMapUrl,
                  originalUrl: result.originalUrl || uploadedImage,
                  success: true,
                  model: result.model || model,
                  resolution: result.resolution || 'unknown',
                  note: result.note,
                  algorithms: result.algorithms,
                  implementation: result.implementation,
                  features: result.features,
                  pointcloudData: result.pointcloudData
                }
                console.log(`Compare mode - ${model} original result:`, modelResult)
                newResults[model] = modelResult
              }
            } else {
              console.error(`Compare mode - ${model} returned invalid result:`, result)
            }
          } else {
            const errorText = await response.text()
            console.error(`Compare mode - ${model} failed:`, response.status, errorText)
          }
        } catch (error) {
          console.error(`Failed to process with ${model}:`, error)
        }
      }

      // 結果を更新
      setDepthResults(newResults)
      setProcessingProgress(100)
      setProcessingStatus(`完了！ ${Object.keys(newResults).length}個のモデルで処理`)
      
      // 最初の成功した結果を表示
      const firstResult = Object.values(newResults)[0]
      if (firstResult) {
        setDepthResult(firstResult)
        setActiveTab('depth')
        setCompareMode(true)
      }

      console.log(`✅ Processed with ${Object.keys(newResults).length} models`)
      
    } catch (error) {
      console.error('Compare all models failed:', error)
      alert('一部のモデルで処理に失敗しました。')
    } finally {
      setTimeout(() => {
        setIsProcessing(false)
        setProcessingProgress(0)
        setProcessingStatus('')
      }, 1500) // 完了メッセージを1.5秒表示
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <h1 className="text-2xl font-bold text-gray-900">
              深度推定・エッジ検出・3D可視化アプリ
            </h1>
            <div className="flex items-center space-x-4">
              <select
                id="model-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="input-field text-sm"
                title="深度推定モデル選択"
                aria-label="深度推定に使用するモデルを選択してください"
              >
                <option value="Intel/dpt-hybrid-midas">MiDaS v3.1 - エッジ検出と構造理解</option>
                <option value="Intel/dpt-large">DPT-Large - Vision Transformer深度推定</option>
                <option value="LiheYoung/depth-anything-small-hf">DepthAnything v1 - Foundation Model深度推定</option>
              </select>
              
              {/* 情報ボタン */}
              <button
                onClick={() => setShowModelInfo(!showModelInfo)}
                className="w-6 h-6 bg-depth-600 text-white rounded-full flex items-center justify-center text-xs hover:bg-depth-700 transition-colors"
                title="モデル詳細情報"
              >
                ?
              </button>
            </div>
            
            {/* モデル説明パネル（別の位置に移動） */}
            {showModelInfo && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowModelInfo(false)}>
                <div className="bg-white rounded-lg shadow-xl p-6 max-w-md mx-4" onClick={(e) => e.stopPropagation()}>
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">モデル詳細</h3>
                    <button
                      onClick={() => setShowModelInfo(false)}
                      className="text-gray-400 hover:text-gray-600 text-xl"
                    >
                      ×
                    </button>
                  </div>
                  
                  {selectedModel === 'Intel/dpt-hybrid-midas' && (
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3">MiDaS v3.1 (Mixed Data Sampling)</h4>
                      <ul className="space-y-2 text-gray-600">
                        <li>• <strong>技術:</strong> CNN + Transformer ハイブリッド</li>
                        <li>• <strong>特徴:</strong> 滑らかで自然な深度変化</li>
                        <li>• <strong>得意:</strong> 風景、人物、多様なシーン</li>
                        <li>• <strong>応用:</strong> 写真編集、映像制作、バーチャル背景</li>
                      </ul>
                    </div>
                  )}
                  {selectedModel === 'Intel/dpt-large' && (
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3">DPT-Large (Dense Prediction Transformer)</h4>
                      <ul className="space-y-2 text-gray-600">
                        <li>• <strong>技術:</strong> Vision Transformer (ViT) ベース</li>
                        <li>• <strong>特徴:</strong> 細かい境界線と物体の輪郭を正確に検出</li>
                        <li>• <strong>得意:</strong> 建築物、家具、複雑な構造物</li>
                        <li>• <strong>応用:</strong> ロボット視覚、AR/VR、自動運転</li>
                      </ul>
                    </div>
                  )}
                  {selectedModel === 'LiheYoung/depth-anything-small-hf' && (
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3">Depth Anything V1 (汎用深度推定)</h4>
                      <ul className="space-y-2 text-gray-600">
                        <li>• <strong>技術:</strong> 大規模データセット学習 Transformer</li>
                        <li>• <strong>特徴:</strong> あらゆる画像タイプに対応</li>
                        <li>• <strong>得意:</strong> 未知のシーン、多様な物体</li>
                        <li>• <strong>応用:</strong> 汎用AI、研究開発、プロトタイピング</li>
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Panel - Upload and Controls */}
          <div className="lg:col-span-1 space-y-6">
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">画像アップロード</h2>
              <ImageUpload onImageUpload={handleImageUpload} />
              
              {uploadedImage && (
                <div className="mt-4 space-y-2">
                  <button
                    onClick={handleDepthEstimation}
                    disabled={isProcessing}
                    className="btn-primary w-full"
                  >
                    深度推定実行
                  </button>
                  <button
                    onClick={handleCompareAllModels}
                    disabled={isProcessing}
                    className="btn-secondary w-full text-sm"
                  >
                    全モデルで比較実行
                  </button>
                  
                  {/* プログレスバー */}
                  {isProcessing && (
                    <div className="mt-3 space-y-2">
                      <div className="flex justify-between text-sm text-gray-600">
                        <span>{processingStatus}</span>
                        <span>{processingProgress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-depth-600 h-2 rounded-full transition-all duration-300 ease-out"
                          style={{ width: `${processingProgress}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {depthResult && (
              <ControlPanel
                settings={viewerSettings}
                onSettingsChange={setViewerSettings}
                depthResult={depthResult}
              />
            )}
          </div>

          {/* Main Content Area */}
          <div className="lg:col-span-3">
            {/* Tab Navigation */}
            <div className="flex justify-between items-center mb-6">
              <div className="flex space-x-1">
                <button
                  onClick={() => setActiveTab('original')}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    activeTab === 'original'
                      ? 'bg-depth-600 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  元画像
                </button>
                <button
                  onClick={() => setActiveTab('depth')}
                  disabled={!depthResult}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    activeTab === 'depth'
                      ? 'bg-depth-600 text-white'
                      : depthResult
                      ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                      : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  }`}
                >
                  深度マップ
                </button>
                <button
                  onClick={() => setActiveTab('3d')}
                  disabled={!depthResult}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    activeTab === '3d'
                      ? 'bg-depth-600 text-white'
                      : depthResult
                      ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                      : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  }`}
                >
                  3Dビュー
                </button>
              </div>
              
              {/* 比較モードトグル */}
              {Object.keys(depthResults).length > 1 && (
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600">比較表示</span>
                  <button
                    onClick={() => setCompareMode(!compareMode)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      compareMode ? 'bg-depth-600' : 'bg-gray-300'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        compareMode ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
              )}
            </div>

            {/* Content Display */}
            <div className="card min-h-96">
              {activeTab === 'original' && (
                <div className="flex items-center justify-center h-96">
                  {uploadedImage ? (
                    <img
                      src={uploadedImage}
                      alt="Uploaded"
                      className="max-w-full max-h-full object-contain rounded-lg"
                    />
                  ) : (
                    <div className="text-center text-gray-500">
                      <p className="text-lg">画像をアップロードしてください</p>
                      <p className="text-sm mt-2">
                        JPEG、PNG、WebP、HEIC、RAW等の画像形式に対応しています
                      </p>
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'depth' && (
                compareMode && Object.keys(depthResults).length > 1 ? (
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {Object.entries(depthResults).map(([modelName, result]) => (
                        <div key={modelName} className="border border-gray-200 rounded-lg p-3">
                          <h4 className="text-sm font-medium text-gray-900 mb-2">
                            {modelName === 'Intel/dpt-hybrid-midas' ? 'MiDaS v3.1' :
                             modelName === 'Intel/dpt-large' ? 'DPT-Large' :
                             modelName === 'LiheYoung/depth-anything-small-hf' ? 'DepthAnything' :
                             modelName}
                          </h4>
                          <img
                            src={result.depthMapUrl}
                            alt={`Depth map - ${modelName}`}
                            className="w-full h-48 object-contain rounded border"
                          />
                          <div className="mt-2 text-xs text-gray-500">
                            {result.resolution}
                          </div>
                        </div>
                      ))}
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
                ) : (
                  <DepthViewer
                    depthResult={depthResult}
                    isProcessing={isProcessing}
                  />
                )
              )}

              {activeTab === '3d' && (
                <ThreeScene
                  originalImage={uploadedImage}
                  depthResult={depthResult}
                  settings={viewerSettings}
                />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-gray-600">
            <p>&copy; 深度推定・3D可視化アプリ<br />2025 オープンキャンパス 塚本吉川研究室展示物</p>
          </div>
        </div>
      </footer>
    </div>
  )
}