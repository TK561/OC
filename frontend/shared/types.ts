export interface DepthEstimationResponse {
  depthMapUrl: string
  originalUrl: string
  success: boolean
  modelUsed: string
  resolution: string
}

export interface ViewerSettings {
  colorMap: string
  pointSize: number
  backgroundColor: string
  showAxes: boolean
}