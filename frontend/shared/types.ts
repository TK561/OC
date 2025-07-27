export interface DepthEstimationResponse {
  depthMapUrl: string
  originalUrl: string
  model_used?: string
}

export interface ViewerSettings {
  colorMap: string
  pointSize: number
  backgroundColor: string
  showAxes: boolean
}