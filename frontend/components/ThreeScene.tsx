import { useRef, useEffect, useState, useMemo } from 'react'
import { Canvas, useFrame, useLoader } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Grid } from '@react-three/drei'
import * as THREE from 'three'
import { ViewerSettings, DepthEstimationResponse } from '@/shared/types'

interface ThreeSceneProps {
  originalImage: string | null
  depthResult: DepthEstimationResponse | null
  settings: ViewerSettings
}

interface PointCloudProps {
  originalImage: string
  depthResult: DepthEstimationResponse
  settings: ViewerSettings
}

function PointCloud({ originalImage, depthResult, settings }: PointCloudProps) {
  const meshRef = useRef<THREE.Points>(null)
  const [pointsData, setPointsData] = useState<{
    positions: Float32Array
    colors: Float32Array
  } | null>(null)

  // Create colormap function
  const getColorFromDepth = useMemo(() => {
    return (depth: number) => {
      const normalized = Math.max(0, Math.min(1, depth))
      
      switch (settings.colorMap) {
        case 'viridis':
          return new THREE.Color().setHSL(0.75 - normalized * 0.75, 1, 0.5)
        case 'plasma':
          return new THREE.Color().setHSL(0.85 - normalized * 0.85, 1, 0.6)
        case 'hot':
          if (normalized < 0.33) {
            return new THREE.Color(normalized * 3, 0, 0)
          } else if (normalized < 0.66) {
            return new THREE.Color(1, (normalized - 0.33) * 3, 0)
          } else {
            return new THREE.Color(1, 1, (normalized - 0.66) * 3)
          }
        case 'cool':
          return new THREE.Color(normalized, 1 - normalized, 1)
        default:
          return new THREE.Color().setHSL(0.75 - normalized * 0.75, 1, 0.5)
      }
    }
  }, [settings.colorMap])

  // Generate point cloud data
  useEffect(() => {
    const generatePointCloud = async () => {
      try {
        if (!originalImage || !depthResult) return

        // Load original image
        const img = new Image()
        img.crossOrigin = 'anonymous'
        
        await new Promise((resolve, reject) => {
          img.onload = resolve
          img.onerror = reject
          img.src = originalImage.startsWith('data:') 
            ? originalImage 
            : `${process.env.NEXT_PUBLIC_BACKEND_URL}${depthResult.originalUrl}`
        })

        // Load depth map
        const depthImg = new Image()
        depthImg.crossOrigin = 'anonymous'
        
        await new Promise((resolve, reject) => {
          depthImg.onload = resolve
          depthImg.onerror = reject
          depthImg.src = `${process.env.NEXT_PUBLIC_BACKEND_URL}${depthResult.depthMapUrl}`
        })

        // Create canvas to extract pixel data
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')!
        
        const width = Math.min(img.width, 400) // Limit resolution for performance
        const height = Math.min(img.height, 400)
        
        canvas.width = width
        canvas.height = height

        // Draw and extract original image data
        ctx.drawImage(img, 0, 0, width, height)
        const imageData = ctx.getImageData(0, 0, width, height)

        // Draw and extract depth data  
        ctx.drawImage(depthImg, 0, 0, width, height)
        const depthData = ctx.getImageData(0, 0, width, height)

        // Generate 3D points
        const positions: number[] = []
        const colors: number[] = []
        
        const step = 2 // Skip pixels for performance
        const scale = 0.01 // Scale factor for visualization
        
        for (let y = 0; y < height; y += step) {
          for (let x = 0; x < width; x += step) {
            const idx = (y * width + x) * 4
            
            // Get RGB color
            const r = imageData.data[idx] / 255
            const g = imageData.data[idx + 1] / 255
            const b = imageData.data[idx + 2] / 255
            
            // Get depth (use red channel of depth map)
            const depth = depthData.data[idx] / 255
            
            // Skip transparent or very dark pixels
            if (imageData.data[idx + 3] < 128 || depth < 0.01) continue
            
            // Convert to 3D coordinates
            const centerX = width / 2
            const centerY = height / 2
            const focal = Math.max(width, height)
            
            const z = depth * scale * 10
            const worldX = (x - centerX) * z * scale
            const worldY = -(y - centerY) * z * scale // Flip Y
            
            positions.push(worldX, worldY, z)
            
            // Use depth-based coloring or original colors
            const depthColor = getColorFromDepth(depth)
            colors.push(depthColor.r, depthColor.g, depthColor.b)
          }
        }

        setPointsData({
          positions: new Float32Array(positions),
          colors: new Float32Array(colors)
        })

      } catch (error) {
        console.error('Failed to generate point cloud:', error)
      }
    }

    generatePointCloud()
  }, [originalImage, depthResult, getColorFromDepth])

  // Update colors when colormap changes
  useEffect(() => {
    if (!pointsData || !meshRef.current) return

    const geometry = meshRef.current.geometry
    const positionAttribute = geometry.getAttribute('position')
    const colorAttribute = geometry.getAttribute('color')
    
    if (!positionAttribute || !colorAttribute) return

    const positions = positionAttribute.array as Float32Array
    const colors = new Float32Array(positions.length)

    for (let i = 0; i < positions.length; i += 3) {
      const z = positions[i + 2]
      const normalizedDepth = Math.max(0, Math.min(1, z / 0.1)) // Normalize based on max z
      const color = getColorFromDepth(normalizedDepth)
      
      colors[i] = color.r
      colors[i + 1] = color.g  
      colors[i + 2] = color.b
    }

    colorAttribute.array = colors
    colorAttribute.needsUpdate = true
  }, [settings.colorMap, pointsData, getColorFromDepth])

  if (!pointsData) {
    return null
  }

  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={pointsData.positions.length / 3}
          array={pointsData.positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={pointsData.colors.length / 3}
          array={pointsData.colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={settings.pointSize}
        vertexColors
        sizeAttenuation={true}
      />
    </points>
  )
}

function SceneContent({ originalImage, depthResult, settings }: ThreeSceneProps) {
  return (
    <>
      {/* Camera */}
      <PerspectiveCamera
        makeDefault
        position={[0, 0, 0.5]}
        fov={75}
        near={0.001}
        far={1000}
      />
      
      {/* Controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        dampingFactor={0.05}
        enableDamping={true}
      />
      
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 10, 5]} intensity={0.8} />
      
      {/* Point Cloud */}
      {originalImage && depthResult && (
        <PointCloud
          originalImage={originalImage}
          depthResult={depthResult}
          settings={settings}
        />
      )}
      
      {/* Axes Helper */}
      {settings.showAxes && <axesHelper args={[0.1]} />}
      
      {/* Grid */}
      <Grid
        args={[0.2, 0.2]}
        position={[0, -0.05, 0]}
        fadeDistance={0.3}
        fadeStrength={1}
        cellSize={0.01}
        cellThickness={0.5}
        cellColor="#6366f1"
        sectionSize={0.05}
        sectionThickness={1}
        sectionColor="#4f46e5"
        infiniteGrid
      />
    </>
  )
}

export default function ThreeScene({ originalImage, depthResult, settings }: ThreeSceneProps) {
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    if (originalImage && depthResult) {
      setIsLoading(true)
      // Simulate loading time
      const timer = setTimeout(() => setIsLoading(false), 1000)
      return () => clearTimeout(timer)
    }
  }, [originalImage, depthResult])

  if (!originalImage || !depthResult) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center text-gray-500">
          <div className="w-16 h-16 mx-auto mb-4 text-gray-400">
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1}
                d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
              />
            </svg>
          </div>
          <p className="text-lg">3Dビューはまだ利用できません</p>
          <p className="text-sm mt-2">
            深度推定を実行すると3D可視化が表示されます
          </p>
        </div>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-pulse w-16 h-16 bg-depth-200 rounded-lg mx-auto mb-4"></div>
          <p className="text-lg font-medium text-gray-900">3Dビュー生成中...</p>
          <p className="text-sm text-gray-600 mt-2">
            ポイントクラウドを生成しています
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative w-full h-96 bg-gray-900 rounded-lg overflow-hidden">
      <Canvas
        style={{ background: settings.backgroundColor }}
        gl={{ antialias: true, alpha: false }}
        camera={{ position: [0, 0, 0.5], fov: 75 }}
      >
        <SceneContent
          originalImage={originalImage}
          depthResult={depthResult}
          settings={settings}
        />
      </Canvas>
      
      {/* Controls Hint */}
      <div className="absolute bottom-4 left-4 bg-black bg-opacity-50 text-white text-xs px-3 py-2 rounded">
        <p>🖱️ ドラッグ: 回転 | ホイール: ズーム | 右クリック: パン</p>
      </div>
      
      {/* Info Panel */}
      <div className="absolute top-4 right-4 bg-black bg-opacity-50 text-white text-xs px-3 py-2 rounded">
        <p>カラーマップ: {settings.colorMap}</p>
        <p>ポイントサイズ: {settings.pointSize.toFixed(2)}</p>
      </div>
    </div>
  )
}