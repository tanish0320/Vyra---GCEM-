import { useEffect, useRef, useState } from "react";
import { Menu, X, ArrowUpRight } from "lucide-react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { Brand } from "./Brand";
import { Button } from "./ui/button";
import { cn } from "../lib/utils";
import { ClickFeedback } from "./Motion";
export function Header() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const menu = useRef<HTMLButtonElement>(null);
  const more = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    setOpen(false);
    if (more.current) more.current.open = false;
  }, [location.pathname]);
  useEffect(() => {
    if (!open) return;
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        menu.current?.focus();
      }
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [open]);
  return (
    <>
      <a
        className="skip-link"
        href="#main"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById("main")?.focus();
        }}
      >
        Skip to content
      </a>
      <header className="site-header">
        <div className="nav-shell wrap">
          <Brand />
          <nav
            id="main-navigation"
            className={cn("main-nav", open && "nav-open")}
            aria-label="Main navigation"
          >
            <NavLink to="/anatomy">Anatomy</NavLink>
            <NavLink to="/studio">Screening</NavLink>
            <NavLink to="/history">History</NavLink>
            <NavLink to="/evidence">Evidence</NavLink>
            <details
              className="more-navigation"
              ref={more}
              onKeyDown={(event) => {
                if (event.key === "Escape" && more.current) {
                  more.current.open = false;
                  more.current.querySelector("summary")?.focus();
                }
              }}
            >
              <summary>More</summary>
              <div className="more-navigation-links">
                <NavLink to="/analytics">Analytics</NavLink>
                <NavLink to="/research">Research &amp; EDA</NavLink>
                <NavLink to="/achilles-heel">Achilles Heel</NavLink>
                <NavLink to="/about">Docs &amp; help</NavLink>
              </div>
            </details>
          </nav>
          <div className="nav-actions">
            <ClickFeedback className="header-cta">
              <Button asChild>
                <Link to="/studio">Start screening</Link>
              </Button>
            </ClickFeedback>
            <button
              ref={menu}
              className="mobile-menu icon-button"
              aria-controls="main-navigation"
              aria-expanded={open}
              aria-label={open ? "Close navigation" : "Open navigation"}
              onClick={() => setOpen((v) => !v)}
            >
              {open ? <X size={22} /> : <Menu size={22} />}
            </button>
          </div>
        </div>
      </header>
    </>
  );
}
export function Footer() {
  return (
    <footer className="unified-footer">
      <div className="wrap">
        <Brand />
        <p>Research screening prototype</p>
        <Link className="text-link" to="/evidence">
          Evidence <ArrowUpRight size={16} />
        </Link>
      </div>
    </footer>
  );
}
export function RouteBehavior() {
  const location = useLocation();
  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      if (location.hash)
        document.getElementById(location.hash.slice(1))?.scrollIntoView({
          behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
            ? "instant"
            : "smooth",
        });
      else {
        window.scrollTo(0, 0);
        document.getElementById("main")?.focus({ preventScroll: true });
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [location.pathname, location.hash]);
  return null;
}
