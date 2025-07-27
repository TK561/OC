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
      const formData = new FormData()
      
      // Get original image and add to form data
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}${depthResult.originalUrl}`)
      const blob = await response.blob()
      formData.append('file', blob, 'image.jpg')
      formData.append('export_format', format)
      formData.append('point_density', '1.0')

      const apiResponse = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/depth/generate-3d`, {
        method: 'POST',
        body: formData,
      })

      if (!apiResponse.ok) {
        throw new Error(`HTTP error! status: ${apiResponse.status}`)
      }

      // Download the generated file
      const resultBlob = await apiResponse.blob()
      const downloadUrl = window.URL.createObjectURL(resultBlob)
      
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = `pointcloud.${format}`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      
      window.URL.revokeObjectURL(downloadUrl)
    } catch (error) {
      console.error('3D generation failed:', error)
      alert('3Dデータ生成に失敗しました')
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
                />
              ))}
              <input
                type="color"
                value={settings.backgroundColor}
                onChange={(e) => handleSettingChange('backgroundColor', e.target.value)}
                className="w-8 h-8 rounded border border-gray-300"
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
              depthResult.modelUsed.includes('mock') 
                ? 'text-orange-600' 
                : 'text-green-600'
            }`}>
              {depthResult.modelUsed}
              {depthResult.modelUsed.includes('mock') && ' (デモ)'}
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