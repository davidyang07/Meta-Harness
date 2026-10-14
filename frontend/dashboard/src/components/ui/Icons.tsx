/** Branch glyph used for fork actions and the fork dialog. */
export function ForkIcon({ className = '' }: { className?: string }) {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden className={className}>
      <circle cx="3" cy="2.5" r="1.4" stroke="currentColor" strokeWidth="1.1" />
      <circle cx="3" cy="9.5" r="1.4" stroke="currentColor" strokeWidth="1.1" />
      <circle cx="9" cy="4.5" r="1.4" stroke="currentColor" strokeWidth="1.1" />
      <path d="M3 4v4M9 6c0 1.6-1.4 2-3 2.3S3 8.5 3 8.5" stroke="currentColor" strokeWidth="1.1" />
    </svg>
  );
}
