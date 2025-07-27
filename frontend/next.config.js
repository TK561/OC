/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['three'],
  
  // 🖼️ 画像最適化
  images: {
    domains: [
      'localhost', 
      'depth-estimation-backend.onrender.com',
      'tk156-depth-estimation-api.hf.space',
      'huggingface.co'
    ],
    unoptimized: true,
  },
  
  // 🔄 API プロキシ
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: (process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000') + '/:path*',
      },
    ];
  },
  
  // 🌍 環境変数
  env: {
    NEXT_PUBLIC_BACKEND_URL: process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000',
  },
  
  // 🗜️ 基本設定
  compress: true,
  poweredByHeader: false,
}

module.exports = nextConfig