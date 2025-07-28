export interface DepthEstimationResponse {
  depthMapUrl: string
  originalUrl: string
  success: boolean
  modelUsed: string
  resolution: string
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