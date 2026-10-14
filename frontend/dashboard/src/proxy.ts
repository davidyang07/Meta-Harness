import { NextRequest, NextResponse } from "next/server";
import { hasAuth0Config } from "./lib/auth-config";

/**
 * Optional Auth0 gate in front of every page (Next 16 `proxy` convention,
 * formerly `middleware`). With no Auth0 environment configured it is a
 * pass-through, and the dashboard behaves exactly as it does without it.
 */
export async function proxy(request: NextRequest) {
  if (!hasAuth0Config()) {
    // The dashboard asks /auth/profile whether login exists at all.
    // Answer it here rather than letting it 404 into the console on
    // every page load.
    if (request.nextUrl.pathname === "/auth/profile") {
      return NextResponse.json({ auth: "disabled" }, { headers: { "cache-control": "no-store" } });
    }
    return NextResponse.next();
  }

  const { auth0 } = await import("./lib/auth0");
  const authRes = await auth0.middleware(request);

  // Auth routes — let Auth0 handle them fully (login, callback, logout)
  if (request.nextUrl.pathname.startsWith("/auth")) {
    return authRes;
  }

  const { origin } = new URL(request.url);
  const session = await auth0.getSession(request);

  // No session → redirect to Auth0 login
  if (!session) {
    return NextResponse.redirect(`${origin}/auth/login`);
  }

  return authRes;
}

export const config = {
  matcher: [
    /*
     * Every route, including "/", except:
     * - _next/static  (static assets)
     * - _next/image   (image optimization)
     * - favicon.ico, sitemap.xml, robots.txt (metadata files)
     *
     * This used to carry a `|$` alternative, which exempted "/" itself
     * and left the landing page — and the run launcher on it — public
     * even with Auth0 configured.
     */
    "/((?!_next/static|_next/image|favicon\\.ico|sitemap\\.xml|robots\\.txt).*)",
  ],
};
