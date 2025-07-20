export interface DepthEstimationRequest {
  image: File | string;
  modelName?: string;
  resolution?: number;
}

export interface DepthEstimationResponse {
  success: boolean;
  depthMapUrl: string;
  originalUrl: string;
  modelUsed: string;
  resolution: string;
}

export interface PointCloudRequest {
  image: File | string;
  modelName?: string;
  pointDensity?: number;
  exportFormat?: 'ply' | 'obj';
}

export interface ProcessingRequest {
  image: File | string;
  method?: string;
  parameters?: Record<string, any>;
}

export interface ProcessingResponse {
  success: boolean;
  processedUrl: string;
  method: string;
  parameters: Record<string, any>;
}

export interface ModelInfo {
  name: string;
  size: string;
  description: string;
  features: string[];
}

export interface AvailableModelsResponse {
  models: string[];
  default: string;
}

export interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
}

export interface ViewerSettings {
  colorMap: 'viridis' | 'plasma' | 'hot' | 'cool';
  pointSize: number;
  backgroundColor: string;
  showAxes: boolean;
}