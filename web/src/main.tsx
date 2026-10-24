import React, { lazy, Suspense } from "react";
import ReactDOM from "react-dom/client";
import { HashRouter, Route, Routes } from "react-router-dom";
import "@fontsource/space-grotesk/latin-500.css";
import "@fontsource/space-grotesk/latin-600.css";
import "@fontsource/space-grotesk/latin-700.css";
import "@fontsource/ibm-plex-sans/latin-400.css";
import "@fontsource/ibm-plex-sans/latin-500.css";
import "@fontsource/ibm-plex-sans/latin-600.css";
import { SessionProvider } from "./context";
import { Header, Footer, RouteBehavior } from "./components/Layout";
import { Studio } from "./pages/Studio";
import { PageTransition } from "./components/Motion";
import TargetCursor from "./components/TargetCursor";
import "./styles.css";
const Home = lazy(() =>
  import("./pages/Home").then((m) => ({ default: m.Home })),
);
const Anatomy = lazy(() =>
  import("./pages/Anatomy").then((m) => ({ default: m.Anatomy })),
);
const Evidence = lazy(() =>
  import("./pages/Evidence").then((m) => ({ default: m.Evidence })),
);
const History = lazy(() =>
  import("./pages/History").then((m) => ({ default: m.History })),
);
const Analytics = lazy(() =>
  import("./pages/Analytics").then((m) => ({ default: m.Analytics })),
);
const Research = lazy(() =>
  import("./pages/Research").then((m) => ({ default: m.Research })),
);
const AchillesHeel = lazy(() =>
  import("./pages/AchillesHeel").then((m) => ({ default: m.AchillesHeel })),
);
const About = lazy(() =>
  import("./pages/About").then((m) => ({ default: m.About })),
);
function App() {
  return (
    <>
      <TargetCursor
        targetSelector="button, a, input, select, textarea, [role='button'], .cursor-target, [data-cursor-label]"
        cursorColor="#026e6d"
        cursorColorOnTarget="#026e6d"
        spinDuration={2}
        hideDefaultCursor={true}
        parallaxOn={true}
        blendMode="normal"
        hoverDuration={0.2}
        showLabel={true}
      />
      <RouteBehavior />
      <Header />
      <main id="main" tabIndex={-1}>
        <PageTransition>
          <Suspense
            fallback={
              <div className="loading-state" role="status">
                Opening workspace…
              </div>
            }
          >
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/studio" element={<Studio />} />
              <Route path="/studio/jobs/:id" element={<Studio />} />
              <Route path="/anatomy" element={<Anatomy />} />
              <Route path="/history" element={<History />} />
              <Route path="/history/:id" element={<History />} />
              <Route path="/evidence" element={<Evidence />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/research" element={<Research />} />
              <Route path="/achilles-heel" element={<AchillesHeel />} />
              <Route path="/about" element={<About />} />
              <Route path="*" element={<Home />} />
            </Routes>
          </Suspense>
        </PageTransition>
      </main>
      <Footer />
    </>
  );
}
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <HashRouter>
      <SessionProvider>
        <App />
      </SessionProvider>
    </HashRouter>
  </React.StrictMode>,
);
