import { ArrowRight, Scan, Image, FileText } from "lucide-react";
import { Link } from "react-router-dom";
import SplitText from "../components/react-bits/SplitText/SplitText";
import { Button } from "../components/ui/button";
import {
  ClickFeedback,
  Reveal,
  ScientificBackdrop,
} from "../components/Motion";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { AnatomyShowcase } from "../anatomy/AnatomyShowcase";

const steps = [
  {
    icon: Image,
    title: "Upload an image",
    detail: "Add a clear photo for screening.",
  },
  {
    icon: Scan,
    title: "Confirm the region",
    detail: "Review and adjust the selected area.",
  },
  {
    icon: FileText,
    title: "Review the result",
    detail: "See the screening output.",
  },
];

export function Home() {
  const reduced = useReducedMotion();
  return (
    <>
      <section
        className="atlas-hero"
        aria-label="Anatomy and screening introduction"
      >
        <div className="atlas-hero-copy">
          {reduced ? (
            <h1>Explore the anatomy behind anemia screening.</h1>
          ) : (
            <SplitText
              text="Explore the anatomy behind anemia screening."
              tag="h1"
              splitType="words"
              effect="rise"
              trigger="load"
              delay={35}
              duration={0.7}
              textAlign="left"
            />
          )}
          <p>
            Discover the layers of the eye. Explore the tissue used in
            screening, then follow your image through the process.
          </p>
          <ClickFeedback className="hero-ctas">
            <Button size="lg" asChild>
              <Link to="/anatomy">Explore anatomy</Link>
            </Button>
            <Link className="text-link" to="/studio">
              Start screening <ArrowRight size={18} />
            </Link>
          </ClickFeedback>
        </div>
        <AnatomyShowcase />
      </section>
      <section className="workflow-band">
        <ScientificBackdrop />
        <div className="workflow-content wrap">
          <Reveal>
            <h2>
              From image <br />
              to result.
            </h2>
          </Reveal>
          <div className="workflow-sequence">
            {steps.map(({ icon: Icon, title, detail }, i) => (
              <Reveal className="workflow-item" key={title} delay={i * 0.08}>
                <div>
                  <span>{String(i + 1).padStart(2, "0")}</span>
                  <Icon size={25} strokeWidth={1.5} />
                </div>
                <h3>{title}</h3>
                <p>{detail}</p>
              </Reveal>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
