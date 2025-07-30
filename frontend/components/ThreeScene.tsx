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
  const [rotation, setRotation] = useState({ x: 0, y: 0 })  // 初期位置を0°に設定
  const [isDragging, setIsDragging] = useState(false)
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1.8)  // 初期ズームを大きく設定

  // グローバルマウスイベントの処理
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      if (isDragging) {
        setIsDragging(false)
      }
    }

    const handleGlobalMouseMove = (e: MouseEvent) => {
      if (!isDragging) return
      
      const deltaX = e.clientX - lastMouse.x
      const deltaY = e.clientY - lastMouse.y
      
      setRotation(prev => ({
        x: prev.x + deltaY * 0.01,
        y: prev.y + deltaX * 0.01
      }))
      
      setLastMouse({ x: e.clientX, y: e.clientY })
    }

    if (isDragging) {
      document.addEventListener('mouseup', handleGlobalMouseUp)
      document.addEventListener('mousemove', handleGlobalMouseMove)
    }

    return () => {
      document.removeEventListener('mouseup', handleGlobalMouseUp)
      document.removeEventListener('mousemove', handleGlobalMouseMove)
    }
  }, [isDragging, lastMouse])

  // キャンバス内でのホイール操作を完全に制御
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const handleCanvasWheel = (e: WheelEvent) => {
      e.preventDefault()
      e.stopPropagation()
      e.stopImmediatePropagation()
      
      const delta = e.deltaY > 0 ? 0.9 : 1.1
      setZoom(prev => Math.max(0.1, Math.min(5.0, prev * delta)))
      
      return false
    }

    // passive: falseを明示的に設定してpreventDefaultを有効にする
    canvas.addEventListener('wheel', handleCanvasWheel, { passive: false })

    return () => {
      canvas.removeEventListener('wheel', handleCanvasWheel)
    }
  }, [depthResult])

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
      const scale = 200 * zoom  // ベーススケールをさらに大きく（180→200）

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
        
        // 透視投影 - 初期表示を見やすく調整
        const perspective = 4.0  // 透視効果をやや強化
        const depth = Math.max(0.1, perspective - finalZ)  // ゼロ除算防止
        const projectedX = centerX + (rotatedX * scale) / depth
        const projectedY = centerY + (rotatedY * scale) / depth
        
        // 深度による点サイズ - より自然なサイズ変化
        const pointSize = Math.max(0.8, settings.pointSize * 10 / depth)  // 最小サイズを0.8に
        
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
  }, [depthResult, settings, rotation, zoom])

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
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
    setLastMouse({ x: e.clientX, y: e.clientY })
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!isDragging) return
    
    const deltaX = e.clientX - lastMouse.x
    const deltaY = e.clientY - lastMouse.y
    
    setRotation(prev => ({
      x: prev.x + deltaY * 0.01,
      y: prev.y + deltaX * 0.01
    }))
    
    setLastMouse({ x: e.clientX, y: e.clientY })
  }

  const handleMouseUp = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleMouseLeave = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleWheel = (e: React.WheelEvent) => {
    // より強力な画面スクロール防止
    e.preventDefault()
    e.stopPropagation()
    e.nativeEvent.preventDefault()
    e.nativeEvent.stopPropagation()
    e.nativeEvent.stopImmediatePropagation()
    
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setZoom(prev => Math.max(0.1, Math.min(5.0, prev * delta)))
    
    // 確実にスクロールを防ぐため、戻り値をfalseに
    return false
  }

  const resetToInitialView = () => {
    setRotation({ x: 0, y: 0 })  // 初期角度に戻す
    setZoom(1.8)  // 初期ズームに戻す
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
        className="w-full h-96 cursor-grab active:cursor-grabbing select-none"
        style={{ 
          touchAction: 'none',
          userSelect: 'none',
          WebkitUserSelect: 'none',
          overscrollBehavior: 'none',
          scrollBehavior: 'auto'
        } as React.CSSProperties}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onWheel={handleWheel}
        onContextMenu={(e) => e.preventDefault()} // 右クリックメニュー無効化
        onDragStart={(e) => e.preventDefault()} // ドラッグ開始を無効化
      />
      
      {/* 操作説明パネル */}
      <div className="absolute top-4 right-4 bg-black bg-opacity-80 text-white px-4 py-3 rounded-lg text-sm">
        <div className="font-semibold mb-2 text-center border-b border-gray-400 pb-1">
          🎮 3D操作ガイド
        </div>
        <div className="space-y-1 mb-3">
          <div className="flex items-center">
            <span className="w-4 text-center">🖱️</span>
            <span className="ml-2">ドラッグ: 3D回転</span>
          </div>
          <div className="flex items-center">
            <span className="w-4 text-center">🔍</span>
            <span className="ml-2">ホイール: ズーム (×{zoom.toFixed(1)})</span>
          </div>
          <div className="flex items-center">
            <span className="w-4 text-center">↻</span>
            <span className="ml-2">角度: X:{(rotation.x * 57.3).toFixed(0)}° Y:{(rotation.y * 57.3).toFixed(0)}°</span>
          </div>
        </div>
        <button
          onClick={resetToInitialView}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium py-2 px-3 rounded transition-colors duration-200"
        >
          🏠 初期配置に戻す
        </button>
      </div>

      {/* 情報パネル */}
      <div className="absolute bottom-4 left-4 bg-black bg-opacity-80 text-white px-4 py-2 rounded-lg text-sm">
        <div className="space-y-1">
          <div>📊 ポイント数: {depthResult.pointcloudData.count.toLocaleString()}</div>
          <div>🔧 モデル: {depthResult.model}</div>
          <div>📐 3D解像度: {
            depthResult.pointcloudData.sampled_size 
              ? `${depthResult.pointcloudData.sampled_size.width}×${depthResult.pointcloudData.sampled_size.height}px`
              : depthResult.pointcloudData.downsample_factor 
                ? `標準画質（軽量化済み）`
                : '高画質'
          }</div>
        </div>
      </div>
    </div>
  )
}