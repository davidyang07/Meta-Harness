'use client';

import { useEffect, useState } from 'react';

export type AuthProfile = {
  name?: string;
  email?: string;
};

export type AuthStatus = {
  /** Auth0 is configured on this dashboard, so /auth/* routes exist. */
  enabled: boolean;
  profile: AuthProfile | null;
};

const SIGNED_OUT: AuthStatus = { enabled: false, profile: null };

/**
 * Whether login exists here at all, and who is signed in.
 *
 * Asked of the server at runtime rather than baked in at build time:
 * with no Auth0 environment the proxy mounts no /auth routes, so the
 * profile endpoint 404s and the Login/Logout links — which would 404
 * too — are not shown.
 */
export function useAuthStatus(): AuthStatus {
  const [status, setStatus] = useState<AuthStatus>(SIGNED_OUT);

  useEffect(() => {
    let cancelled = false;
    fetch('/auth/profile', { cache: 'no-store' })
      .then(async (response): Promise<AuthStatus> => {
        if (response.status === 404) return SIGNED_OUT;
        if (response.status === 200) {
          return { enabled: true, profile: (await response.json()) as AuthProfile };
        }
        // 401 / 204: Auth0 is configured and nobody is signed in.
        return { enabled: true, profile: null };
      })
      .catch(() => SIGNED_OUT)
      .then((next) => {
        if (!cancelled) setStatus(next);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return status;
}
