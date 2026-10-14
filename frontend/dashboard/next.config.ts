import type { NextConfig } from "next";

// No rewrites. The browser talks to the backend directly at
// NEXT_PUBLIC_API_BASE_URL (see src/lib/api.ts). A catch-all `/api/*`
// rewrite to localhost:8000 used to sit here unused, and since `next dev`
// listens on every interface it let anyone on the network drive the
// backend through the dashboard.
const nextConfig: NextConfig = {};

export default nextConfig;
