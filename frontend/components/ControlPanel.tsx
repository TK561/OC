import { useState } from 'react'
import { ViewerSettings, DepthEstimationResponse } from '@/shared/types'

interface ControlPanelProps {
  settings: ViewerSettings
  onSettingsChange: (settings: ViewerSettings) => void
  depthResult: DepthEstimationResponse
}

export default function ControlPanel({ settings, onSettingsChange, depthResult }: ControlPanelProps) {
  const [showColorMapInfo, setShowColorMapInfo] = useState(false)

  const handleSettingChange = (key: keyof ViewerSettings, value: any) => {
    onSettingsChange({
      ...settings,
      [key]: value
    })
  }


  return (
    <div className="space-y-6">
      {/* 3D Viewer Settings */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">3D表示設定</h3>
        
        <div className="space-y-4">
          {/* Color Map */}
          <div>
            <div className="flex items-center mb-2">
              <label className="block text-sm font-medium text-gray-700">
                カラーマップ
              </label>
              <button
                onClick={() => setShowColorMapInfo(!showColorMapInfo)}
                className="ml-2 w-5 h-5 bg-gray-200 hover:bg-gray-300 text-gray-600 text-xs font-bold rounded-full flex items-center justify-center transition-colors"
                title="カラーマップの説明を表示"
              >
                ?
              </button>
            </div>
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
            
            {/* カラーマップ説明 */}
            {showColorMapInfo && (
              <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded text-xs text-blue-800">
                <div className="space-y-2">
                  <div><strong>Viridis:</strong> 紫から青、緑、黄へと変化。科学的可視化に最適です。</div>
                  <div><strong>Plasma:</strong> 紫から赤、黄へと変化。高コントラストで細かい変化を強調します。</div>
                  <div><strong>Hot:</strong> 黒から赤、黄、白へと変化。熱画像風の色合いで温度を連想させます。</div>
                  <div><strong>Cool:</strong> 青から緑、赤へと変化。涼しい色合いで落ち着いた表示です。</div>
                </div>
              </div>
            )}
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
              {['#000000', '#2d3748', '#4a5568', '#ffffff'].map((color) => (
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
                className="w-8 h-8 rounded border-2 border-gray-300 cursor-pointer"
                title="カスタム背景色"
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
          <li>• ポイントサイズで点の大きさを調節</li>
          <li>• 背景色は自由に変更できます</li>
        </ul>
      </div>
    </div>
  )
}