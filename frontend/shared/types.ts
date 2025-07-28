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
  }
}

export interface ViewerSettings {
  colorMap: string
  pointSize: number
  backgroundColor: string
  showAxes: boolean
}