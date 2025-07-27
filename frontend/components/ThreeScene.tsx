import { DepthEstimationResponse, ViewerSettings } from '@/shared/types'

interface ThreeSceneProps {
  originalImage: string | null
  depthResult: DepthEstimationResponse | null
  settings: ViewerSettings
}

// Temporarily disabled 3D viewer due to Three.js version conflicts
export default function ThreeScene({ originalImage, depthResult, settings }: ThreeSceneProps) {
  return (
    <div className="flex items-center justify-center h-96 bg-gray-100 rounded-lg">
      <div className="text-center text-gray-600">
        <div className="text-6xl mb-4">🔧</div>
        <h3 className="text-lg font-semibold mb-2">3Dビューアー</h3>
        <p className="text-sm">現在メンテナンス中です</p>
        <p className="text-xs mt-2">深度マップは「深度マップ」タブで確認できます</p>
      </div>
    </div>
  )
}