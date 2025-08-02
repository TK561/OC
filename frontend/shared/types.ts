export interface DepthEstimationResponse {
  depthMapUrl: string
  originalUrl: string
  success: boolean
  model: string
  resolution: string
  note?: string
  algorithms?: string[]
  implementation?: string
  features?: string[]
  pointcloudData?: {
    points: number[][]
    colors: number[][]
    count: number
    downsample_factor: number
    original_size?: {
      width: number
      height: number
    }
    sampled_size?: {
      width: number
      height: number
    }
  }
}

export interface EnhancedDepthProcessingResponse {
  success: boolean
  originalUrl: string
  rawDepthMapUrl: string
  enhancedDepthMapUrl: string
  finalImageUrl: string
  processing_info: {
    model: string
    depth_inverted: boolean
    depth_gamma: number
    depth_contrast: number
    smoothing_strength: number
    gradient_enhancement: number
    processing_type: string
  }
  resolution: string
}

export interface ViewerSettings {
  colorMap: string
  pointSize: number
  backgroundColor: string
  showAxes: boolean
}