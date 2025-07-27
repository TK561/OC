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
  const [selectedModel, setSelectedModel] = useState('Intel/dpt-hybrid-midas')
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
      // Convert image to base64 data URL if it's a blob URL
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
      console.log('Making request to Gradio API')
      
      // Gradio Client-style API call
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/call/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data: [imageDataUrl],
          session_hash: Math.random().toString(36).substring(7)
        })
      })

      console.log('Response status:', response.status)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('API Error Response:', errorText)
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`)
      }

      const result = await response.json()
      console.log('Gradio response:', result)
      
      // Process Gradio response format
      if (result.data && result.data.length >= 2) {
        const originalImageUrl = result.data[0]
        const depthImageUrl = result.data[1]
        
        setDepthResult({
          depthMapUrl: depthImageUrl,
          originalUrl: originalImageUrl || uploadedImage,
          success: true,
          modelUsed: 'gradio-depth',
          resolution: 'auto'
        })
        setActiveTab('depth')
      } else {
        throw new Error('Invalid response format from Gradio API')
      }
    } catch (error) {
      console.error('Depth estimation failed:', error)
      alert('深度推定に失敗しました。バックエンドの状態を確認してください。')
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
              深度推定・3D可視化アプリ
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
                <option value="Intel/dpt-hybrid-midas">MiDaS (高速)</option>
                <option value="Intel/dpt-large">DPT-Large (高精度)</option>
                <option value="LiheYoung/depth-anything-large-hf">DepthAnything (汎用)</option>
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
            <p>&copy; 2024 深度推定・3D可視化アプリ</p>
            <p className="text-sm mt-1">
              DPT, MiDaS, DepthAnything モデルを使用した展示アプリケーション
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}