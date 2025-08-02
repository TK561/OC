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

export interface EdgeDepthProcessingResponse {
  success: boolean
  originalUrl: string
  depthMapUrl: string
  edgeMapUrl: string
  composedImageUrl: string
  finalImageUrl: string
  processing_info: {
    model: string
    edge_thresholds: number[]
    depth_inverted: boolean
    depth_gamma: number
    depth_contrast: number
    composition_mode: string
    post_gamma: number
    post_blur: number
  }
  resolution: string
}

export interface ViewerSettings {
  colorMap: string
  pointSize: number
  backgroundColor: string
  showAxes: boolean
}