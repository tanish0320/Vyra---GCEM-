import { lazy, Suspense, type ReactNode } from "react";
import { useLocation } from "react-router-dom";
import AnimatedContent from "./react-bits/AnimatedContent/AnimatedContent";
import ClickSpark from "./react-bits/ClickSpark/ClickSpark";
import { useReducedMotion } from "../hooks/useReducedMotion";

const DotGrid = lazy(() => import("./react-bits/DotGrid/DotGrid"));

export function Reveal({
  children,
  className,
  delay = 0,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
}) {
  const reduced = useReducedMotion();
  return (
    <AnimatedContent
      className={className}
      distance={reduced ? 0 : 28}
      duration={reduced ? 0 : 0.55}
      delay={reduced ? 0 : delay}
      initialOpacity={reduced ? 1 : 0}
      animateOpacity={!reduced}
      once
    >
      {children}
    </AnimatedContent>
  );
}

export function PageTransition({ children }: { children: ReactNode }) {
  const location = useLocation();
  const reduced = useReducedMotion();
  return (
    <AnimatedContent
      key={location.pathname}
      className="page-transition"
      trigger="mount"
      distance={reduced ? 0 : 10}
      duration={reduced ? 0 : 0.22}
      initialOpacity={reduced ? 1 : 0.5}
      animateOpacity={!reduced}
    >
      {children}
    </AnimatedContent>
  );
}

export function ClickFeedback({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  const reduced = useReducedMotion();
  return (
    <ClickSpark
      className={className}
      disabled={reduced}
      sparkColor="#006c67"
      sparkCount={4}
      sparkRadius={16}
      sparkSize={4}
      sparkWidth={1.5}
      duration={280}
      glow={0}
      shape="line"
    >
      {children}
    </ClickSpark>
  );
}

export function ScientificBackdrop() {
  const reduced = useReducedMotion();
  return (
    <div className="workflow-backdrop" aria-hidden="true">
      {!reduced && (
        <Suspense fallback={null}>
          <DotGrid
            baseColor="#c6d6d3"
            activeColor="#006c67"
            dotSize={2}
            gap={24}
            opacity={0.5}
            dpr={1}
            glow={0}
            intro={false}
            clickShock={false}
            mouseInteraction
          />
        </Suspense>
      )}
    </div>
  );
}
