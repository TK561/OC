import { useRef, useEffect, useState } from 'react'
import { DepthEstimationResponse, ViewerSettings } from '@/shared/types'

interface ThreeSceneProps {
  originalImage: string | null
  depthResult: DepthEstimationResponse | null
  settings: ViewerSettings
}

export default function ThreeScene({ originalImage, depthResult, settings }: ThreeSceneProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [rotation, setRotation] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 })

  useEffect(() => {
    if (!depthResult?.pointcloudData || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    setIsLoading(true)

    // キャンバスサイズ設定
    canvas.width = 600
    canvas.height = 400

    const renderPointCloud = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      
      // 背景
      ctx.fillStyle = settings.backgroundColor
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      if (!depthResult.pointcloudData) return

      const { points, colors } = depthResult.pointcloudData
      const centerX = canvas.width / 2
      const centerY = canvas.height / 2
      const scale = 150

      // 3D → 2D投影
      points.forEach((point: number[], index: number) => {
        const [x, y, z] = point
        
        // 回転適用
        const cosY = Math.cos(rotation.y)
        const sinY = Math.sin(rotation.y)
        const cosX = Math.cos(rotation.x)
        const sinX = Math.sin(rotation.x)
        
        // Y軸回転
        const rotatedX = x * cosY - z * sinY
        const rotatedZ = x * sinY + z * cosY
        
        // X軸回転
        const rotatedY = y * cosX - rotatedZ * sinX
        const finalZ = y * sinX + rotatedZ * cosX
        
        // 透視投影 - より強いパースペクティブ効果
        const perspective = 3.5  // より強い透視効果
        const depth = Math.max(0.1, perspective - finalZ)  // ゼロ除算防止
        const projectedX = centerX + (rotatedX * scale) / depth
        const projectedY = centerY + (rotatedY * scale) / depth
        
        // 深度による点サイズ - より自然なサイズ変化
        const pointSize = Math.max(0.5, settings.pointSize * 8 / depth)
        
        // 色設定
        const color = colors[index]
        ctx.fillStyle = `rgb(${Math.floor(color[0] * 255)}, ${Math.floor(color[1] * 255)}, ${Math.floor(color[2] * 255)})`
        
        // 点描画
        ctx.beginPath()
        ctx.arc(projectedX, projectedY, pointSize, 0, Math.PI * 2)
        ctx.fill()
      })

      // 軸表示
      if (settings.showAxes) {
        drawAxes(ctx, centerX, centerY, scale)
      }
    }

    renderPointCloud()
    setIsLoading(false)
  }, [depthResult, settings, rotation])

  const drawAxes = (ctx: CanvasRenderingContext2D, centerX: number, centerY: number, scale: number) => {
    ctx.strokeStyle = '#ff0000'
    ctx.lineWidth = 2
    
    // X軸 (赤)
    ctx.beginPath()
    ctx.moveTo(centerX, centerY)
    ctx.lineTo(centerX + scale * 0.3, centerY)
    ctx.stroke()
    
    // Y軸 (緑)
    ctx.strokeStyle = '#00ff00'
    ctx.beginPath()
    ctx.moveTo(centerX, centerY)
    ctx.lineTo(centerX, centerY - scale * 0.3)
    ctx.stroke()
    
    // Z軸 (青)
    ctx.strokeStyle = '#0000ff'
    ctx.beginPath()
    ctx.moveTo(centerX, centerY)
    ctx.lineTo(centerX - scale * 0.2, centerY + scale * 0.2)
    ctx.stroke()
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true)
    setLastMouse({ x: e.clientX, y: e.clientY })
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return
    
    const deltaX = e.clientX - lastMouse.x
    const deltaY = e.clientY - lastMouse.y
    
    setRotation(prev => ({
      x: prev.x + deltaY * 0.01,
      y: prev.y + deltaX * 0.01
    }))
    
    setLastMouse({ x: e.clientX, y: e.clientY })
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  if (!depthResult) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-100 rounded-lg">
        <div className="text-center text-gray-600">
          <div className="text-6xl mb-4">📸</div>
          <p className="text-lg">画像をアップロードして深度推定を実行してください</p>
        </div>
      </div>
    )
  }

  if (!depthResult.pointcloudData) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-100 rounded-lg">
        <div className="text-center text-gray-600">
          <div className="text-6xl mb-4">⚠️</div>
          <p className="text-lg">3Dデータが生成されていません</p>
          <p className="text-sm">新しいバックエンドAPIが必要です</p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative bg-white rounded-lg overflow-hidden">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
          <div className="text-center">
            <div className="animate-spin text-4xl mb-2">🔄</div>
            <p>3D描画中...</p>
          </div>
        </div>
      )}
      
      <canvas
        ref={canvasRef}
        className="w-full h-96 cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
      
      <div className="absolute bottom-4 left-4 bg-black bg-opacity-70 text-white px-3 py-2 rounded text-sm">
        <div>🎮 ドラッグで回転</div>
        <div>📊 ポイント数: {depthResult.pointcloudData.count}</div>
        <div>🔧 モデル: {depthResult.model}</div>
      </div>
    </div>
  )
}