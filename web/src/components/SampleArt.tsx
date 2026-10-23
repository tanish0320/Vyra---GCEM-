import { useId } from "react";
import type { Modality } from "../types";

export function SampleArt({
  modality = "conjunctiva",
  normalized = false,
  glare = false,
  className = "",
  cover = false,
  pallor = 0,
}: {
  modality?: Modality;
  normalized?: boolean;
  glare?: boolean;
  className?: string;
  cover?: boolean;
  pallor?: number;
}) {
  const id = useId().replaceAll(":", "");
  const ref = (name: string) => `url(#${id}-${name})`;
  return (
    <svg
      className={`sample-art ${className}`}
      viewBox="0 0 800 560"
      preserveAspectRatio={cover ? "xMidYMid slice" : "xMidYMid meet"}
      role="img"
      aria-label={`Illustrative ${modality === "conjunctiva" ? "eye" : "fingernail"} sample${normalized ? ", normalized view" : ""}`}
    >
      <defs>
        <radialGradient id={`${id}-back`} cx="48%" cy="45%" r="70%">
          <stop stopColor={normalized ? "#f1e9e2" : "#f4e8dc"} />
          <stop offset="1" stopColor={normalized ? "#e5e0d9" : "#e4d5c9"} />
        </radialGradient>
        <linearGradient id={`${id}-skin`} x1="0" y1="0" x2=".85" y2="1">
          <stop stopColor={normalized ? "#d9b6a5" : "#dbaf94"} />
          <stop offset=".4" stopColor={normalized ? "#ecd4c4" : "#f1cbae"} />
          <stop offset=".76" stopColor={normalized ? "#cfa48e" : "#d39f7d"} />
          <stop offset="1" stopColor="#efd6c1" />
        </linearGradient>
        <radialGradient id={`${id}-white`} cx="50%" cy="20%" r="80%">
          <stop stopColor="#fffdf6" />
          <stop offset=".7" stopColor={normalized ? "#ece7e1" : "#ece1d4"} />
          <stop offset="1" stopColor="#d1b5a7" />
        </radialGradient>
        <radialGradient id={`${id}-iris`} cx="40%" cy="35%" r="65%">
          <stop stopColor="#b6a07b" />
          <stop offset=".42" stopColor="#8e8066" />
          <stop offset=".8" stopColor="#594f43" />
          <stop offset="1" stopColor="#302f2e" />
        </radialGradient>
        <linearGradient id={`${id}-tissue`} x1="0" y1="0" x2="0" y2="1">
          <stop
            stopColor={`color-mix(in srgb, ${normalized ? "#d7a4a0" : "#d8958b"} ${100 - pallor * 100}%, #dbccc0)`}
          />
          <stop
            offset=".5"
            stopColor={`color-mix(in srgb, ${normalized ? "#bf8588" : "#bd7473"} ${100 - pallor * 100}%, #cfc0b4)`}
          />
          <stop
            offset="1"
            stopColor={`color-mix(in srgb, ${normalized ? "#e4b8ae" : "#e6aaa0"} ${100 - pallor * 100}%, #e0d2c3)`}
          />
        </linearGradient>
        <linearGradient id={`${id}-nail`} x1="0" y1="0" x2="1" y2=".4">
          <stop stopColor={normalized ? "#edd4cb" : "#eccfc1"} />
          <stop
            offset=".35"
            stopColor={`color-mix(in srgb, ${normalized ? "#deb6b2" : "#dcb0a3"} ${100 - pallor * 100}%, #e6dacd)`}
          />
          <stop offset=".7" stopColor={normalized ? "#f0d8d1" : "#efd2c1"} />
          <stop offset="1" stopColor={normalized ? "#dec5bf" : "#d0b6a6"} />
        </linearGradient>
        <radialGradient id={`${id}-shine`}>
          <stop stopColor="white" stopOpacity=".92" />
          <stop offset="1" stopColor="white" stopOpacity="0" />
        </radialGradient>
        <filter
          id={`${id}-shadow`}
          x="-30%"
          y="-30%"
          width="160%"
          height="160%"
        >
          <feGaussianBlur stdDeviation="15" />
        </filter>
        <filter id={`${id}-grain`}>
          <feTurbulence
            type="fractalNoise"
            baseFrequency=".45"
            numOctaves="2"
            stitchTiles="stitch"
          />
          <feColorMatrix type="saturate" values="0" />
          <feComponentTransfer>
            <feFuncA type="linear" slope=".07" />
          </feComponentTransfer>
          <feBlend in="SourceGraphic" mode="multiply" />
        </filter>
        <clipPath id={`${id}-eye`}>
          <path d="M109 263C193 150 317 134 403 145C522 130 638 174 696 263C618 347 497 369 403 364C286 374 174 340 109 263Z" />
        </clipPath>
        <clipPath id={`${id}-mucosa`}>
          <path d="M183 311C305 372 510 382 626 313C601 374 525 404 407 402C291 403 214 369 183 311Z" />
        </clipPath>
      </defs>
      <rect width="800" height="560" fill={ref("back")} />
      {modality === "conjunctiva" ? (
        <>
          <ellipse
            cx="404"
            cy="310"
            rx="300"
            ry="130"
            fill="#9b6c5a"
            opacity=".13"
            filter={ref("shadow")}
          />
          <path
            d="M0 244C112 132 199 106 311 105C487 69 665 117 800 250V560H0Z"
            fill={ref("skin")}
          />
          <path
            d="M82 230C195 112 307 94 418 111C541 93 675 153 724 233"
            fill="none"
            stroke="#b58a72"
            strokeWidth="3"
            opacity=".45"
          />
          <path
            d="M99 258C178 143 301 119 404 131C522 112 661 169 709 263C636 359 511 384 402 379C279 383 162 344 99 258Z"
            fill="#966f5c"
            opacity=".36"
          />
          <path
            d="M109 263C193 150 317 134 403 145C522 130 638 174 696 263C618 347 497 369 403 364C286 374 174 340 109 263Z"
            fill={ref("white")}
          />
          <g clipPath={ref("eye")}>
            <circle cx="402" cy="248" r="117" fill="#3e3c35" />
            <circle cx="402" cy="248" r="112" fill={ref("iris")} />
            {Array.from({ length: 58 }, (_, i) => {
              const angle = (i / 58) * Math.PI * 2;
              return (
                <path
                  key={i}
                  d={`M${402 + Math.cos(angle) * 55} ${248 + Math.sin(angle) * 55}Q${402 + Math.cos(angle + 0.035) * 90} ${248 + Math.sin(angle + 0.035) * 90} ${402 + Math.cos(angle) * 108} ${248 + Math.sin(angle) * 108}`}
                  stroke={i % 3 ? "#d3b98d" : "#342f2c"}
                  strokeWidth={i % 4 ? "1.5" : "3"}
                  opacity=".38"
                  fill="none"
                />
              );
            })}
            <circle cx="402" cy="248" r="53" fill="#22272a" />
            <circle
              cx="402"
              cy="248"
              r="58"
              fill="none"
              stroke="#6a5c44"
              strokeWidth="4"
            />
            <ellipse
              cx="431"
              cy="207"
              rx="27"
              ry="19"
              fill="white"
              opacity=".84"
              transform="rotate(24 431 207)"
            />
            <circle cx="370" cy="280" r="10" fill="white" opacity=".13" />
            <path
              d="M118 273C169 246 183 273 234 246M160 306C199 288 220 312 264 290M568 228C608 245 639 216 674 247M556 308C604 285 645 308 683 274"
              fill="none"
              stroke="#bb8b89"
              strokeWidth="1.7"
              opacity=".48"
            />
            <path
              d="M158 259L167 236M211 294L220 315M634 237L643 219M611 296L623 314"
              stroke="#b58b89"
              opacity=".4"
              fill="none"
            />
          </g>
          <path
            d="M183 311C305 372 510 382 626 313C601 374 525 404 407 402C291 403 214 369 183 311Z"
            fill={ref("tissue")}
          />
          <g
            clipPath={ref("mucosa")}
            opacity={normalized ? ".34" : ".43"}
            stroke="#96565c"
            fill="none"
            strokeWidth="1.4"
          >
            {Array.from({ length: 19 }, (_, i) => (
              <path
                key={i}
                d={`M${210 + i * 22} 410Q${221 + i * 20} ${355 + (i % 3) * 7} ${204 + i * 22} 320M${221 + i * 21} 375l${i % 2 ? 13 : -12} -13`}
              />
            ))}
          </g>
          <path
            d="M109 263C193 150 317 134 403 145C522 130 638 174 696 263"
            fill="none"
            stroke="#5b4237"
            strokeWidth="8"
            strokeLinecap="round"
          />
          <path
            d="M108 264C161 333 279 372 404 367C527 373 641 333 696 263"
            fill="none"
            stroke="#b67e73"
            strokeWidth="4"
          />
          <path
            d="M182 404C303 453 501 465 628 395"
            stroke="#b88d75"
            strokeWidth="2"
            opacity=".38"
            fill="none"
          />
          <ellipse
            cx="153"
            cy="276"
            rx="23"
            ry="12"
            fill="#cf9c8f"
            transform="rotate(20 153 276)"
          />
          {!normalized && (
            <ellipse
              cx="485"
              cy="350"
              rx="35"
              ry="18"
              fill={ref("shine")}
              opacity=".8"
            />
          )}
        </>
      ) : (
        <>
          <ellipse
            cx="419"
            cy="430"
            rx="154"
            ry="115"
            fill="#a28370"
            opacity=".17"
            filter={ref("shadow")}
          />
          <path
            d="M164 560V305C164 221 247 218 253 305L270 560Z"
            fill={ref("skin")}
            opacity=".48"
          />
          <path
            d="M542 560L549 304C549 227 626 223 631 310L650 560Z"
            fill={ref("skin")}
            opacity=".48"
          />
          <path
            d="M284 560L294 215C296 122 338 87 401 87C465 87 508 132 510 215L524 560Z"
            fill={ref("skin")}
          />
          <path
            d="M305 254C291 199 314 114 374 104"
            fill="none"
            stroke="#f7dfcc"
            strokeWidth="7"
            opacity=".7"
          />
          <path
            d="M495 233C503 160 467 115 435 105"
            fill="none"
            stroke="#b88b71"
            strokeWidth="3"
            opacity=".3"
          />
          <path
            d="M319 211C317 161 349 130 400 130C452 130 485 163 482 211L475 329C472 362 443 380 400 380C359 380 328 361 326 329Z"
            fill="#a77f70"
            opacity=".22"
          />
          <path
            d="M329 202C326 160 357 137 400 137C444 137 475 164 472 204L465 318C463 347 438 364 400 364C361 364 337 347 335 318Z"
            fill={ref("nail")}
          />
          <path
            d="M330 205C329 160 358 139 400 139C440 139 471 163 471 205C433 183 367 183 330 205Z"
            fill="#f5e8d9"
            opacity=".83"
          />
          <path
            d="M354 321C377 301 423 301 446 321C434 352 368 352 354 321Z"
            fill="#eee0d7"
            opacity=".45"
          />
          <path
            d="M336 241C352 257 351 300 353 319M363 224C374 259 368 280 374 296M417 218C406 256 420 281 410 310M446 231C430 257 447 286 438 316"
            stroke="#b87c7c"
            strokeWidth="1.4"
            opacity=".22"
            fill="none"
          />
          <path
            d="M326 332C339 384 465 385 476 330"
            fill="none"
            stroke="#b28773"
            strokeWidth="2"
            opacity=".5"
          />
          <path
            d="M324 424Q401 443 482 424M326 435Q400 451 481 435"
            fill="none"
            stroke="#bd927b"
            strokeWidth="1.7"
            opacity=".4"
          />
          {!normalized && (
            <ellipse
              cx="442"
              cy="233"
              rx="15"
              ry="76"
              fill={ref("shine")}
              transform="rotate(6 442 233)"
              opacity=".65"
            />
          )}
        </>
      )}
      <rect
        width="800"
        height="560"
        fill="transparent"
        filter={ref("grain")}
        opacity=".45"
      />
      {glare && !normalized && (
        <ellipse cx="430" cy="280" rx="260" ry="195" fill={ref("shine")} />
      )}
    </svg>
  );
}
