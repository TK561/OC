import { useState } from 'react'
import ImageUpload from '@/components/ImageUpload'
import DepthViewer from '@/components/DepthViewer'
import ThreeScene from '@/components/ThreeScene'
import ControlPanel from '@/components/ControlPanel'
import { DepthEstimationResponse, ViewerSettings } from '@/shared/types'

export default function Home() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [depthResult, setDepthResult] = useState<DepthEstimationResponse | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [selectedModel, setSelectedModel] = useState('depth-anything/Depth-Anything-V2-Base-hf')
  const [viewerSettings, setViewerSettings] = useState<ViewerSettings>({
    colorMap: 'viridis',
    pointSize: 0.1,
    backgroundColor: '#000000',
    showAxes: true
  })
  const [activeTab, setActiveTab] = useState<'original' | 'depth' | '3d'>('original')

  const handleImageUpload = (imageUrl: string) => {
    setUploadedImage(imageUrl)
    setDepthResult(null)
  }

  const handleDepthEstimation = async () => {
    if (!uploadedImage) return

    setIsProcessing(true)
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

      console.log('Backend URL:', process.env.NEXT_PUBLIC_BACKEND_URL)
      
      // Try Railway API first, fallback to mock
      try {
        console.log('Trying Railway API...')
        console.log('Backend URL:', process.env.NEXT_PUBLIC_BACKEND_URL)
        
        // ファイルアップロード用のFormData作成
        const formData = new FormData()
        
        // Base64をBlobに変換
        const byteCharacters = atob(imageDataUrl.split(',')[1])
        const byteNumbers = new Array(byteCharacters.length)
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i)
        }
        const byteArray = new Uint8Array(byteNumbers)
        const blob = new Blob([byteArray], { type: 'image/jpeg' })
        
        formData.append('file', blob, 'image.jpg')
        formData.append('model', selectedModel)
        
        // 10秒タイムアウト設定
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 10000)
        
        const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/predict`, {
          method: 'POST',
          body: formData,
          signal: controller.signal
        })
        
        clearTimeout(timeoutId)

        if (response.ok) {
          const result = await response.json()
          console.log('Railway API Response:', result)
          console.log('PointCloud Data:', result.pointcloudData)
          
          if (result.success && result.depthMapUrl) {
            // Railway APIからの深度マップ
            setDepthResult({
              depthMapUrl: result.depthMapUrl,
              originalUrl: result.originalUrl || uploadedImage,
              success: true,
              model: result.model || 'Railway-API',
              resolution: result.resolution || 'unknown',
              note: result.note,
              algorithms: result.algorithms,
              implementation: result.implementation,
              features: result.features,
              pointcloudData: result.pointcloudData
            })
            setActiveTab('depth')
            console.log('✅ Railway API depth estimation successful!')
            return
          }
        }
        throw new Error(`Railway API failed: ${response.status} ${response.statusText}`)
      } catch (apiError) {
        console.log('Railway API failed:', apiError)
        console.log('Falling back to mock API...')
        
        // フォールバック: モックAPI
        const { createMockDepthMap } = await import('../lib/mockApi')
        const mockDepthMap = await createMockDepthMap(imageDataUrl)
        
        setDepthResult({
          depthMapUrl: mockDepthMap,
          originalUrl: uploadedImage,
          success: true,
          model: 'mock-gradient',
          resolution: 'original'
        })
        setActiveTab('depth')
        
        console.log('⚠️ Using mock depth estimation (Railway API unavailable)')
      }
    } catch (error) {
      console.error('All depth estimation methods failed:', error)
      alert('深度推定に失敗しました。画像を確認してもう一度お試しください。')
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <h1 className="text-2xl font-bold text-gray-900">
              2025 深度推定・3D可視化アプリ
            </h1>
            <div className="flex items-center space-x-4">
              <label htmlFor="model-select" className="sr-only">深度推定モデル選択</label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="input-field text-sm"
                title="深度推定モデル選択"
                aria-label="深度推定に使用するモデルを選択してください"
              >
                <optgroup label="従来モデル">
                  <option value="Intel/dpt-hybrid-midas">MiDaS v3.1 (高速・470MB)</option>
                  <option value="Intel/dpt-large">DPT-Large (高精度・1.3GB)</option>
                  <option value="LiheYoung/depth-anything-large-hf">DepthAnything v1 (汎用・1.4GB)</option>
                </optgroup>
                <optgroup label="最新モデル (2025)">
                  <option value="depth-anything/Depth-Anything-V2-Small-hf">DepthAnything V2 Small (軽量・99MB)</option>
                  <option value="depth-anything/Depth-Anything-V2-Base-hf">DepthAnything V2 Base (バランス・390MB)</option>
                  <option value="depth-anything/Depth-Anything-V2-Large-hf">DepthAnything V2 Large (最高精度・1.3GB)</option>
                </optgroup>
                <optgroup label="特殊用途">
                  <option value="apple/DepthPro">Apple DepthPro (メトリック深度・1.9GB)</option>
                  <option value="Intel/zoedepth-nyu-kitti">ZoeDepth (絶対深度・1.4GB)</option>
                </optgroup>
              </select>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Panel - Upload and Controls */}
          <div className="lg:col-span-1 space-y-6">
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">画像アップロード</h2>
              <ImageUpload onImageUpload={handleImageUpload} />
              
              {uploadedImage && (
                <div className="mt-4">
                  <button
                    onClick={handleDepthEstimation}
                    disabled={isProcessing}
                    className="btn-primary w-full"
                  >
                    {isProcessing ? '処理中...' : '深度推定実行'}
                  </button>
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
            <div className="flex space-x-1 mb-6">
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
                        JPEG、PNG、WebP形式に対応しています
                      </p>
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'depth' && (
                <DepthViewer
                  depthResult={depthResult}
                  isProcessing={isProcessing}
                />
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
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-gray-600">
            <p>&copy; 深度推定・3D可視化アプリ 2025 オープンキャンパス 塚本吉川研究室展示物</p>
          </div>
        </div>
      </footer>
    </div>
  )
}