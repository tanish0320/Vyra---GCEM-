import {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
} from "react";
import { Move, RotateCcw } from "lucide-react";
import { clampBox } from "../lib/domain";
import type { Box, Modality, Photo } from "../types";
import { SampleArt } from "./SampleArt";

interface Props {
  width: number;
  height: number;
  box: Box;
  onChange: (box: Box) => void;
  onReset: () => void;
  modality: Modality;
  photo: Photo | null;
  glare: boolean;
}
type Drag = { mode: string; x: number; y: number; box: Box };

export function ROIEditor({
  width,
  height,
  box,
  onChange,
  onReset,
  modality,
  photo,
  glare,
}: Props) {
  const frame = useRef<HTMLDivElement>(null);
  const drag = useRef<Drag | null>(null);
  const [renderedWidth, setRenderedWidth] = useState(560);
  const hintId = useId();
  useEffect(() => {
    const element = frame.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) =>
      setRenderedWidth(entries[0].contentRect.width),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const point = (event: PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(
        0,
        Math.min(width, ((event.clientX - rect.left) / rect.width) * width),
      ),
      y: Math.max(
        0,
        Math.min(height, ((event.clientY - rect.top) / rect.height) * height),
      ),
    };
  };
  const move = (original: Box, dx: number, dy: number) => {
    const w = original.x1 - original.x0,
      h = original.y1 - original.y0;
    const x0 = Math.max(0, Math.min(width - w, original.x0 + dx));
    const y0 = Math.max(0, Math.min(height - h, original.y0 + dy));
    onChange(clampBox({ x0, y0, x1: x0 + w, y1: y0 + h }, width, height));
  };
  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.focus();
    event.currentTarget.setPointerCapture(event.pointerId);
    const p = point(event);
    const target = event.target as Element;
    const handle = target.closest("[data-handle]")?.getAttribute("data-handle");
    const mode =
      handle ?? (target.closest("[data-roi-body]") ? "move" : "draw");
    drag.current = { mode, x: p.x, y: p.y, box: { ...box } };
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    const p = point(event),
      start = drag.current;
    if (start.mode === "move") {
      move(start.box, p.x - start.x, p.y - start.y);
      return;
    }
    let x0 = start.box.x0,
      y0 = start.box.y0,
      x1 = start.box.x1,
      y1 = start.box.y1;
    if (start.mode === "draw") {
      x0 = Math.min(start.x, p.x);
      y0 = Math.min(start.y, p.y);
      x1 = Math.max(start.x, p.x);
      y1 = Math.max(start.y, p.y);
    } else {
      if (start.mode.includes("l")) x0 = p.x;
      if (start.mode.includes("r")) x1 = p.x;
      if (start.mode.includes("t")) y0 = p.y;
      if (start.mode.includes("b")) y1 = p.y;
    }
    onChange(
      clampBox(
        {
          x0: Math.min(x0, x1),
          y0: Math.min(y0, y1),
          x1: Math.max(x0, x1),
          y1: Math.max(y0, y1),
        },
        width,
        height,
      ),
    );
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const shift = event.shiftKey ? 10 : 1;
    const steps: Record<string, [number, number]> = {
      ArrowLeft: [-shift, 0],
      ArrowRight: [shift, 0],
      ArrowUp: [0, -shift],
      ArrowDown: [0, shift],
    };
    if (steps[event.key]) {
      event.preventDefault();
      move(box, ...steps[event.key]);
    }
  };
  const handleSize = (width / Math.max(renderedWidth, 1)) * 10;
  const hitSize = (width / Math.max(renderedWidth, 1)) * 44;
  const corners: [string, number, number][] = [
    ["tl", box.x0, box.y0],
    ["tr", box.x1, box.y0],
    ["bl", box.x0, box.y1],
    ["br", box.x1, box.y1],
  ];
  return (
    <div className="roi-editor">
      <div className="roi-editor-top">
        <span>
          <Move size={15} /> Frame the tissue region
        </span>
        <button className="subtle-button" onClick={onReset}>
          <RotateCcw size={13} /> Reset region
        </button>
      </div>
      <div className="roi-editor-stage">
        <div
          ref={frame}
          className="roi-canvas"
          tabIndex={0}
          role="group"
          aria-label="Region of interest editor"
          aria-describedby={hintId}
          style={{
            aspectRatio: `${width}/${height}`,
            width: `min(100%, ${Math.round((560 * width) / height)}px)`,
          }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={() => {
            drag.current = null;
          }}
          onPointerCancel={() => {
            drag.current = null;
          }}
          onKeyDown={onKeyDown}
        >
          {photo ? (
            <img src={photo.url} alt="Your selected image" draggable={false} />
          ) : (
            <SampleArt modality={modality} glare={glare} />
          )}
          <svg
            className="roi-overlay"
            viewBox={`0 0 ${width} ${height}`}
            aria-hidden="true"
          >
            <path
              d={`M0 0H${width}V${height}H0Z M${box.x0} ${box.y0}V${box.y1}H${box.x1}V${box.y0}Z`}
              fill="#122e32"
              fillOpacity=".18"
              fillRule="evenodd"
            />
            <rect
              data-roi-body="true"
              x={box.x0}
              y={box.y0}
              width={box.x1 - box.x0}
              height={box.y1 - box.y0}
              fill="transparent"
              stroke="#007b74"
              strokeWidth={(width / Math.max(renderedWidth, 1)) * 2}
              className="roi-selection"
            />
            {corners.map(([name, x, y]) => (
              <g key={name} className={`roi-handle roi-handle-${name}`}>
                <rect
                  data-handle={name}
                  x={x - hitSize / 2}
                  y={y - hitSize / 2}
                  width={hitSize}
                  height={hitSize}
                  fill="transparent"
                />
                <rect
                  x={x - handleSize / 2}
                  y={y - handleSize / 2}
                  width={handleSize}
                  height={handleSize}
                  fill="#ffffff"
                  stroke="#007b74"
                  strokeWidth={width / Math.max(renderedWidth, 1)}
                  pointerEvents="none"
                />
              </g>
            ))}
          </svg>
        </div>
      </div>
      <div className="roi-controls">
        {[
          ["X", box.x0],
          ["Y", box.y0],
          ["Width", box.x1 - box.x0],
          ["Height", box.y1 - box.y0],
        ].map(([label, value]) => (
          <label key={label}>
            <span>{label}</span>
            <input
              aria-label={`ROI ${label}`}
              type="number"
              min={label === "X" || label === "Y" ? 0 : 1}
              max={label === "X" || label === "Width" ? width : height}
              value={value}
              onChange={(e) => {
                const n = Number(e.target.value);
                let next = { ...box };
                if (label === "X") {
                  const w = box.x1 - box.x0;
                  next.x0 = Math.min(width - w, Math.max(0, n));
                  next.x1 = next.x0 + w;
                }
                if (label === "Y") {
                  const h = box.y1 - box.y0;
                  next.y0 = Math.min(height - h, Math.max(0, n));
                  next.y1 = next.y0 + h;
                }
                if (label === "Width") next.x1 = box.x0 + n;
                if (label === "Height") next.y1 = box.y0 + n;
                onChange(clampBox(next, width, height));
              }}
            />
            <small>px</small>
          </label>
        ))}
      </div>
      <p className="roi-hint" id={hintId}>
        Drag to select or move the region. Use arrow keys to move it; hold Shift
        for larger steps.
      </p>
    </div>
  );
}
