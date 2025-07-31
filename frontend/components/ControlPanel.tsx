import { useState } from 'react'
import { ViewerSettings, DepthEstimationResponse } from '@/shared/types'

interface ControlPanelProps {
  settings: ViewerSettings
  onSettingsChange: (settings: ViewerSettings) => void
  depthResult: DepthEstimationResponse
}

export default function ControlPanel({ settings, onSettingsChange, depthResult }: ControlPanelProps) {
  const [isGenerating3D, setIsGenerating3D] = useState(false)

  const handleSettingChange = (key: keyof ViewerSettings, value: any) => {
    onSettingsChange({
      ...settings,
      [key]: value
    })
  }

  const handleGenerate3D = async (format: 'ply' | 'obj') => {
    setIsGenerating3D(true)
    try {
      // Get the original image from the result
      if (!depthResult.originalUrl) {
        throw new Error('元画像データが見つかりません')
      }

      // Convert data URL to blob if needed
      let imageBlob: Blob
      if (depthResult.originalUrl.startsWith('data:')) {
        // Data URL - convert to blob
        const response = await fetch(depthResult.originalUrl)
        imageBlob = await response.blob()
      } else {
        // Regular URL - fetch from backend
        const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}${depthResult.originalUrl}`)
        if (!response.ok) {
          throw new Error('元画像の取得に失敗しました')
        }
        imageBlob = await response.blob()
      }

      const formData = new FormData()
      formData.append('file', imageBlob, 'image.jpg')
      formData.append('model', depthResult.model || 'Intel/dpt-large')
      formData.append('format', format)

      const apiResponse = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/export-3d`, {
        method: 'POST',
        body: formData,
      })

      if (!apiResponse.ok) {
        const errorText = await apiResponse.text()
        throw new Error(`API Error: ${apiResponse.status} - ${errorText}`)
      }

      // Download the generated file
      const resultBlob = await apiResponse.blob()
      const downloadUrl = window.URL.createObjectURL(resultBlob)
      
      // Generate filename with model and timestamp
      const modelName = depthResult.model?.replace('/', '_') || 'unknown'
      const timestamp = new Date().toISOString().slice(0, 19).replace(/[:]/g, '-')
      const filename = `depth_${modelName}_${timestamp}.${format}`
      
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      window.URL.revokeObjectURL(downloadUrl)
      
      console.log(`✅ ${format.toUpperCase()} file downloaded: ${filename}`)
    } catch (error) {
      console.error('3D export failed:', error)
      const errorMessage = error instanceof Error ? error.message : '不明なエラー'
      alert(`3Dファイルの生成に失敗しました:\n${errorMessage}`)
    } finally {
      setIsGenerating3D(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* 3D Viewer Settings */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">3D表示設定</h3>
        
        <div className="space-y-4">
          {/* Color Map */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              カラーマップ
            </label>
            <select
              value={settings.colorMap}
              onChange={(e) => handleSettingChange('colorMap', e.target.value as ViewerSettings['colorMap'])}
              className="input-field w-full"
              title="カラーマップ選択"
              aria-label="深度マップのカラーマップを選択"
            >
              <option value="viridis">Viridis</option>
              <option value="plasma">Plasma</option>
              <option value="hot">Hot</option>
              <option value="cool">Cool</option>
            </select>
          </div>

          {/* Point Size */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              ポイントサイズ: {settings.pointSize.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={settings.pointSize}
              onChange={(e) => handleSettingChange('pointSize', parseFloat(e.target.value))}
              className="w-full"
              title={`ポイントサイズ: ${settings.pointSize.toFixed(2)}`}
              aria-label={`ポイントサイズを${settings.pointSize.toFixed(2)}に設定`}
            />
          </div>

          {/* Background Color */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              背景色
            </label>
            <div className="flex space-x-2">
              {['#000000', '#ffffff', '#1f2937', '#0f172a'].map((color) => (
                <button
                  key={color}
                  onClick={() => handleSettingChange('backgroundColor', color)}
                  className={`w-8 h-8 rounded border-2 ${
                    settings.backgroundColor === color ? 'border-depth-500' : 'border-gray-300'
                  }`}
                  style={{ backgroundColor: color }}
                  title={`背景色を${color}に設定`}
                  aria-label={`背景色を${color}に設定`}
                />
              ))}
              <input
                type="color"
                value={settings.backgroundColor}
                onChange={(e) => handleSettingChange('backgroundColor', e.target.value)}
                className="w-8 h-8 rounded border border-gray-300"
                title="カスタム背景色選択"
                aria-label="カスタム背景色を選択"
              />
            </div>
          </div>

          {/* Show Axes */}
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="showAxes"
              checked={settings.showAxes}
              onChange={(e) => handleSettingChange('showAxes', e.target.checked)}
              className="rounded border-gray-300 text-depth-600 focus:ring-depth-500"
            />
            <label htmlFor="showAxes" className="text-sm font-medium text-gray-700">
              座標軸を表示
            </label>
          </div>
        </div>
      </div>

      {/* Export Options */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">エクスポート</h3>
        
        <div className="space-y-3">
          <button
            onClick={() => handleGenerate3D('ply')}
            disabled={isGenerating3D}
            className="btn-primary w-full"
          >
            {isGenerating3D ? '生成中...' : '📦 PLYファイル出力'}
          </button>
          
          <button
            onClick={() => handleGenerate3D('obj')}
            disabled={isGenerating3D}
            className="btn-secondary w-full"
          >
            {isGenerating3D ? '生成中...' : '📐 OBJファイル出力'}
          </button>
        </div>
        
        <div className="mt-4 text-xs text-gray-500">
          <p>• PLY: ポイントクラウド形式（推奨）</p>
          <p>• OBJ: メッシュ形式（処理時間長）</p>
        </div>
      </div>

      {/* Processing Info */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">処理情報</h3>
        
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">モデル:</span>
            <span className={`font-medium ${
              depthResult.model?.includes('mock') 
                ? 'text-orange-600' 
                : 'text-green-600'
            }`}>
              {depthResult.model}
              {depthResult.model?.includes('mock') && ' (デモ)'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">解像度:</span>
            <span className="font-medium">{depthResult.resolution}</span>
          </div>
        </div>
      </div>

      {/* Tips */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="text-sm font-medium text-blue-900 mb-2">💡 使い方のヒント</h4>
        <ul className="text-xs text-blue-800 space-y-1">
          <li>• マウスで3Dビューを回転・ズーム可能</li>
          <li>• カラーマップで深度の可視化を調整</li>
          <li>• PLY形式は軽量で推奨</li>
          <li>• OBJ形式はメッシュ生成で時間がかかります</li>
        </ul>
      </div>
    </div>
  )
}