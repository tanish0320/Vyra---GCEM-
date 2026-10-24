import { Link } from "react-router-dom";
/** Intersecting aperture arcs: observation, focus, and the same specimen seen from different angles. */
export function BrandMark({ size = 34 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      aria-hidden="true"
    >
      <circle cx="24" cy="24" r="20" stroke="currentColor" strokeWidth="1.6" />
      <ellipse
        cx="24"
        cy="24"
        rx="16"
        ry="19"
        transform="rotate(36 24 24)"
        stroke="currentColor"
        strokeWidth="1.6"
      />
      <ellipse
        cx="24"
        cy="24"
        rx="16"
        ry="19"
        transform="rotate(-36 24 24)"
        stroke="currentColor"
        strokeWidth="1.6"
      />
      <path
        d="M10 17C16 12 29 12 37 23M10 31c8 6 20 5 27-5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}
export function Brand({ light = false }: { light?: boolean }) {
  return (
    <Link
      to="/"
      className={`brand${light ? " brand-light" : ""} cursor-target`}
      data-cursor-label="VYRA Home"
      aria-label="Vyra home"
    >
      <BrandMark />
      <span>VYRA</span>
    </Link>
  );
}
