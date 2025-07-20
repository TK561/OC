/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['three'],
  images: {
    domains: ['localhost', 'depth-estimation-backend.onrender.com'],
    unoptimized: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: (process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000') + '/:path*',
      },
    ];
  },
  env: {
    NEXT_PUBLIC_BACKEND_URL: process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000',
  },
  // Vercel 最適化
  experimental: {
    optimizePackageImports: ['three', '@react-three/fiber', '@react-three/drei'],
  },
  // 静的ファイルの最適化
  compress: true,
}

module.exports = nextConfig