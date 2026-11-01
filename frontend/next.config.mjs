const backend = process.env.BACKEND_ORIGIN || 'http://localhost:8000'

const nextConfig = {
  async rewrites() {
    return [
      { source: '/api/:path*', destination: `${backend}/api/:path*` },
      { source: '/health', destination: `${backend}/health` },
    ]
  },
}

export default nextConfig
