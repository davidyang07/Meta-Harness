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
 * with no Auth0 environment the proxy answers the profile request with
 * `{"auth": "disabled"}`, and the Login/Logout links — which would 404 —
 * are not shown.
 */
export function useAuthStatus(): AuthStatus {
  const [status, setStatus] = useState<AuthStatus>(SIGNED_OUT);

  useEffect(() => {
    let cancelled = false;
    fetch('/auth/profile', { cache: 'no-store' })
      .then(async (response): Promise<AuthStatus> => {
        if (response.status === 404) return SIGNED_OUT;
        if (response.status === 200) {
          const body = (await response.json()) as AuthProfile & { auth?: string };
          // The proxy's own answer when Auth0 is not configured.
          if (body.auth === "disabled") return SIGNED_OUT;
          return { enabled: true, profile: body };
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
