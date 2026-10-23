import { Link } from "react-router-dom";
import {
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  CircleHelp,
  ScanLine,
} from "lucide-react";
import { Button } from "../components/ui/button";
import "../research.css";

export function About() {
  return (
    <div className="workspace-page wrap research-page">
      <div className="workspace-heading">
        <div>
          <span className="eyebrow">HOW TO USE VYRA</span>
          <h1>About this project</h1>
          <p>Image screening, understandable outputs, and visible evidence.</p>
        </div>
        <Button asChild variant="outline">
          <Link to="/studio">
            <ScanLine data-icon="inline-start" /> Open screening
          </Link>
        </Button>
      </div>
      <section className="research-intro">
        <CircleHelp aria-hidden="true" />
        <p>
          VYRA is a research prototype for studying visual anemia-screening
          signals in the lower-eyelid conjunctiva and fingernail. The website
          connects image capture, model outputs, explainability, and educational
          anatomy in one workflow.
        </p>
      </section>
      <section aria-labelledby="about-workflow">
        <div className="research-section-heading">
          <h2 id="about-workflow">From image to saved result</h2>
        </div>
        <ol className="research-workflow">
          <li>
            <span>01</span>
            <div>
              <h3>Choose the tissue and add visit details</h3>
              <p>
                Upload a JPEG, PNG, or WebP photo, use a camera when permitted,
                or select a supplied reference image. Enter a patient name or
                case ID, age and biological sex.
              </p>
            </div>
          </li>
          <li>
            <span>02</span>
            <div>
              <h3>Review the proposed image region</h3>
              <p>
                Confirm or adjust the tissue crop. If automatic detection cannot
                propose a usable region, deliberately select the tissue or
                retake the photo.
              </p>
            </div>
          </li>
          <li>
            <span>03</span>
            <div>
              <h3>Follow processing and review the result</h3>
              <p>
                The connected backend prepares the image, runs its model, and
                returns the available explanation. Review quality notes, model
                outputs, and follow-up information together.
              </p>
            </div>
          </li>
          <li>
            <span>04</span>
            <div>
              <h3>Return to the screening</h3>
              <p>
                History provides the saved result, its available images and
                report. Research tools such as Achilles Heel run separate
                experiments on an image.
              </p>
            </div>
          </li>
        </ol>
      </section>
      <div className="research-two-column">
        <section className="research-card">
          <span className="eyebrow">LOCAL BACKEND SETUP</span>
          <h2>Connect the working service</h2>
          <p>
            On the configured project computer, start the Integration backend
            from the project root:
          </p>
          <pre className="research-command">
            <code>python app.py</code>
          </pre>
          <p>
            Open the served website at <code>http://127.0.0.1:5000</code>.
            Camera access requires permission and a secure browser context.
            Upload remains available if the camera cannot open.
          </p>
          <a className="research-inline-link" href="/health">
            Check backend readiness <ArrowUpRight size={15} />
          </a>
          <a className="research-inline-link" href="/about">
            Legacy backend documentation <ArrowUpRight size={15} />
          </a>
        </section>
        <section className="research-card">
          <span className="eyebrow">READ THE OUTPUTS</span>
          <h2>What the numbers mean</h2>
          <p>
            The anemia probability is a model output. The eye hemoglobin value
            is an image estimate; nail hemoglobin is unavailable. Patient
            details provide record context and are not extra inputs to the
            existing model.
          </p>
          <p>
            Grad-CAM displays model attention. Colorimetry summarizes image
            pixels. The educational 3D models and perturbation experiments do
            not establish a diagnosis or prove model correctness.
          </p>
          <Link className="research-inline-link" to="/evidence">
            <BookOpen size={17} /> View evaluation and limitations{" "}
            <ArrowRight size={15} />
          </Link>
        </section>
      </div>
      <section className="research-card research-boundary">
        <h2>Clinical assessment remains necessary</h2>
        <p>
          This research prototype does not diagnose or rule out anemia. Concerns
          about symptoms require clinical assessment and, when indicated by a
          qualified professional, laboratory testing. Keep image estimates
          separate from measured laboratory hemoglobin.
        </p>
        <p>
          Optional assistant, dual-image, stress-test and research features
          appear when the connected backend declares them available. A missing
          capability is shown explicitly.
        </p>
      </section>
    </div>
  );
}
