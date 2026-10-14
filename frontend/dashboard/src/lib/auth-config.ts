/**
 * Auth0 is optional. It is on only when every variable the SDK needs is
 * present; otherwise the proxy lets every request through and the
 * dashboard hides its login controls.
 */
export function hasAuth0Config(): boolean {
  return Boolean(
    process.env.AUTH0_DOMAIN &&
      process.env.AUTH0_CLIENT_ID &&
      process.env.AUTH0_SECRET &&
      (process.env.AUTH0_CLIENT_SECRET ||
        process.env.AUTH0_CLIENT_ASSERTION_SIGNING_KEY),
  );
}
