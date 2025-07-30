import { useState, useRef, DragEvent } from 'react'

interface ImageUploadProps {
  onImageUpload: (imageUrl: string) => void
}

export default function ImageUpload({ onImageUpload }: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const sampleImages = [
    { name: '風景', url: '/samples/landscape.jpeg' },
    { name: '建物', url: '/samples/building.jpeg' },
    { name: '人物', url: '/samples/animal.jpg' }
  ]

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
    
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileUpload(files[0])
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileUpload(files[0])
    }
  }

  const handleFileUpload = (file: File) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('画像ファイルを選択してください')
      return
    }

    // Validate file size (50MB max)
    if (file.size > 50 * 1024 * 1024) {
      alert('ファイルサイズが大きすぎます（50MB以下）')
      return
    }

    // Create file reader
    const reader = new FileReader()
    
    reader.onloadstart = () => setUploadProgress(0)
    reader.onprogress = (e) => {
      if (e.lengthComputable) {
        setUploadProgress((e.loaded / e.total) * 100)
      }
    }
    
    reader.onload = (e) => {
      setUploadProgress(100)
      const result = e.target?.result as string
      onImageUpload(result)
      setTimeout(() => setUploadProgress(0), 1000)
    }
    
    reader.onerror = () => {
      alert('ファイルの読み込みに失敗しました')
      setUploadProgress(0)
    }
    
    reader.readAsDataURL(file)
  }

  const handleSampleSelect = (sampleUrl: string) => {
    onImageUpload(sampleUrl)
  }

  const handleUrlUpload = () => {
    const url = prompt('画像URLを入力してください:')
    if (url) {
      // Validate URL format
      try {
        new URL(url)
        onImageUpload(url)
      } catch {
        alert('有効なURLを入力してください')
      }
    }
  }

  return (
    <div className="space-y-4">
      {/* Drag & Drop Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer ${
          isDragging
            ? 'border-depth-500 bg-depth-50'
            : 'border-gray-300 hover:border-depth-400'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="space-y-2">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            stroke="currentColor"
            fill="none"
            viewBox="0 0 48 48"
          >
            <path
              d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <div>
            <p className="text-lg font-medium text-gray-900">
              画像をドラッグ&ドロップ
            </p>
            <p className="text-sm text-gray-600">
              または <span className="text-depth-600 font-medium">クリックして選択</span>
            </p>
          </div>
          <p className="text-xs text-gray-500">
            JPEG, PNG, WebP (最大50MB)
          </p>
        </div>
      </div>

      {/* Progress Bar */}
      {uploadProgress > 0 && uploadProgress < 100 && (
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-depth-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${uploadProgress}%` }}
          />
        </div>
      )}

      {/* Hidden File Input */}
      <input
        id="file-input"
        name="file-input"
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
        title="画像ファイル選択"
        aria-label="画像ファイルを選択してください"
      />

      {/* Action Buttons */}
      <div className="flex space-x-2">
        <button
          onClick={() => fileInputRef.current?.click()}
          className="btn-secondary flex-1 text-sm"
        >
          📁 ファイル選択
        </button>
        <button
          onClick={handleUrlUpload}
          className="btn-secondary flex-1 text-sm"
        >
          🔗 URL指定
        </button>
      </div>

      {/* Sample Images */}
      <div className="border-t pt-4">
        <h3 className="text-sm font-medium text-gray-900 mb-3">
          サンプル画像
        </h3>
        <div className="grid grid-cols-3 gap-2">
          {sampleImages.map((sample, index) => (
            <button
              key={index}
              onClick={() => handleSampleSelect(sample.url)}
              className="group relative aspect-square bg-gray-100 rounded-lg overflow-hidden hover:ring-2 hover:ring-depth-500 transition-all"
            >
              <img
                src={sample.url}
                alt={sample.name}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                onError={(e) => {
                  e.currentTarget.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpemU9IjE0IiBmaWxsPSIjOTk5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+Tm8gSW1hZ2U8L3RleHQ+PC9zdmc+'
                }}
              />
              <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all" />
              <div className="absolute bottom-1 left-1 right-1">
                <p className="text-xs text-white bg-black bg-opacity-50 rounded px-1 py-0.5 truncate">
                  {sample.name}
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
