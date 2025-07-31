// 画像のEXIF情報を読み取って正しい向きで表示するためのユーティリティ

export async function getOrientedImageUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    
    reader.onload = async (e) => {
      const arrayBuffer = e.target?.result as ArrayBuffer
      const blob = new Blob([arrayBuffer], { type: file.type })
      
      // EXIF Orientation を取得
      const orientation = await getExifOrientation(arrayBuffer)
      
      // 画像を回転・反転する必要がある場合
      if (orientation > 1) {
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')
        const img = new Image()
        
        img.onload = () => {
          // キャンバスのサイズを設定
          const width = img.width
          const height = img.height
          
          // Orientationに応じてキャンバスサイズを調整
          if (orientation > 4) {
            canvas.width = height
            canvas.height = width
          } else {
            canvas.width = width
            canvas.height = height
          }
          
          // 変換を適用
          if (ctx) {
            switch (orientation) {
              case 2: // horizontal flip
                ctx.transform(-1, 0, 0, 1, width, 0)
                break
              case 3: // 180° rotate
                ctx.transform(-1, 0, 0, -1, width, height)
                break
              case 4: // vertical flip
                ctx.transform(1, 0, 0, -1, 0, height)
                break
              case 5: // vertical flip + 90° rotate clockwise
                ctx.transform(0, 1, 1, 0, 0, 0)
                break
              case 6: // 90° rotate clockwise
                ctx.transform(0, 1, -1, 0, height, 0)
                break
              case 7: // horizontal flip + 90° rotate clockwise
                ctx.transform(0, -1, -1, 0, height, width)
                break
              case 8: // 90° rotate counter-clockwise
                ctx.transform(0, -1, 1, 0, 0, width)
                break
            }
            
            // 画像を描画
            ctx.drawImage(img, 0, 0)
            
            // Data URLとして出力（EXIF情報は除去される）
            canvas.toBlob((blob) => {
              if (blob) {
                const url = URL.createObjectURL(blob)
                resolve(url)
              } else {
                reject(new Error('Canvas to blob conversion failed'))
              }
            }, 'image/jpeg', 0.95)  // JPEGで出力してEXIF情報をクリア
          }
        }
        
        img.onerror = () => reject(new Error('Image loading failed'))
        img.src = URL.createObjectURL(blob)
      } else {
        // 回転が必要ない場合は直接Data URLを返す
        const reader2 = new FileReader()
        reader2.onload = (e) => resolve(e.target?.result as string)
        reader2.onerror = () => reject(new Error('File reading failed'))
        reader2.readAsDataURL(file)
      }
    }
    
    reader.onerror = () => reject(new Error('File reading failed'))
    reader.readAsArrayBuffer(file)
  })
}

// EXIF Orientationの値を取得
async function getExifOrientation(arrayBuffer: ArrayBuffer): Promise<number> {
  const view = new DataView(arrayBuffer)
  
  // JPEGマーカーを確認
  if (view.getUint16(0) !== 0xFFD8) {
    return 1 // Not a JPEG
  }
  
  let offset = 2
  while (offset < view.byteLength) {
    const marker = view.getUint16(offset)
    
    // APPマーカーを探す
    if (marker === 0xFFE1) {
      const exifLength = view.getUint16(offset + 2)
      const exifData = new DataView(arrayBuffer, offset + 4, exifLength - 2)
      
      // "Exif\0\0"を確認
      if (exifData.getUint32(0) === 0x45786966 && exifData.getUint16(4) === 0x0000) {
        // TIFFヘッダーを解析
        const tiffOffset = 6
        const littleEndian = exifData.getUint16(tiffOffset) === 0x4949
        const orientationTag = littleEndian ? 0x0112 : 0x1201
        
        // IFDを検索
        const firstIfdOffset = exifData.getUint32(tiffOffset + 4, littleEndian)
        const tags = exifData.getUint16(tiffOffset + firstIfdOffset, littleEndian)
        
        for (let i = 0; i < tags; i++) {
          const tagOffset = tiffOffset + firstIfdOffset + 2 + (i * 12)
          const tag = exifData.getUint16(tagOffset, littleEndian)
          
          if (tag === 0x0112) { // Orientation tag
            return exifData.getUint16(tagOffset + 8, littleEndian)
          }
        }
      }
      break
    } else if ((marker & 0xFF00) !== 0xFF00) {
      break
    } else {
      offset += 2 + view.getUint16(offset + 2)
    }
  }
  
  return 1 // Default orientation
}