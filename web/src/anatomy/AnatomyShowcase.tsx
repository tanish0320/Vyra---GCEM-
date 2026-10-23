import { useCallback, useEffect, useRef, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";
import { useReducedMotion } from "../hooks/useReducedMotion";
import "./eye-film.css";

/** A local film keeps the homepage light; interactive 3D lives on /anatomy. */
export function AnatomyShowcase() {
  const root = useRef<HTMLDivElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const inView = useRef(false);
  const intent = useRef<boolean | null>(null);
  const blocked = useRef(false);
  const availabilityCheck = useRef<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [failed, setFailed] = useState(false);
  const reduced = useReducedMotion();
  const base = `${import.meta.env.BASE_URL}anatomy/eye-film`;
  const play = useCallback(() => {
    const element = video.current;
    if (!element) return;
    // Some Chromium versions leave play() pending and omit error events when
    // every <source> fails. Inspect the terminal media state while loading.
    if (availabilityCheck.current === null) {
      availabilityCheck.current = window.setInterval(() => {
        const exhausted =
          element.currentSrc &&
          element.networkState === HTMLMediaElement.NETWORK_NO_SOURCE;
        if (
          element.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA ||
          exhausted
        ) {
          window.clearInterval(availabilityCheck.current!);
          availabilityCheck.current = null;
          if (exhausted) {
            blocked.current = true;
            element.pause();
            setFailed(true);
            setPlaying(false);
          }
        }
      }, 250);
    }
    element
      .play()
      .then(() => {
        if (availabilityCheck.current !== null) {
          window.clearInterval(availabilityCheck.current);
          availabilityCheck.current = null;
        }
        blocked.current = false;
      })
      .catch((error: unknown) => {
        if (availabilityCheck.current !== null) {
          window.clearInterval(availabilityCheck.current);
          availabilityCheck.current = null;
        }
        // Leaving the viewport may pause an in-flight play request.
        if (error instanceof DOMException && error.name === "AbortError")
          return;
        if (error instanceof DOMException && error.name === "NotSupportedError")
          setFailed(true);
        blocked.current = true;
        setPlaying(false);
      });
  }, []);
  useEffect(() => {
    const element = video.current;
    if (!element) return;
    const sync = () => {
      const wanted = intent.current ?? !reduced;
      if (wanted && inView.current && !document.hidden && !blocked.current)
        play();
      else element.pause();
    };
    const observer = new IntersectionObserver(
      ([entry]) => {
        inView.current =
          entry.isIntersecting && entry.intersectionRatio >= 0.15;
        sync();
      },
      { threshold: [0, 0.15] },
    );
    if (root.current) observer.observe(root.current);
    document.addEventListener("visibilitychange", sync);
    sync();
    return () => {
      observer.disconnect();
      document.removeEventListener("visibilitychange", sync);
      element.pause();
      if (availabilityCheck.current !== null) {
        window.clearInterval(availabilityCheck.current);
        availabilityCheck.current = null;
      }
    };
  }, [play, reduced]);
  const toggle = () => {
    if (playing) {
      intent.current = false;
      video.current?.pause();
    } else {
      intent.current = true;
      blocked.current = false;
      play();
    }
  };
  const replay = () => {
    if (!video.current) return;
    video.current.currentTime = 0;
    intent.current = true;
    blocked.current = false;
    play();
  };
  return (
    <figure className="eye-film" aria-label="Eye anatomy film">
      <div ref={root} className="eye-film-picture">
        <video
          ref={video}
          className="eye-film-video"
          width={1600}
          height={1200}
          poster={`${base}-poster.webp`}
          preload={reduced ? "none" : "metadata"}
          muted
          playsInline
          loop
          disablePictureInPicture
          aria-label="A detailed eye slowly separates into its anatomical layers, then reassembles."
          onPlaying={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onError={(event) => {
            // React bubbles individual <source> failures while the browser may
            // successfully select another format or responsive source.
            const element = event.currentTarget;
            const terminalError = event.target === element && element.error;
            const exhaustedSources =
              element.networkState === HTMLMediaElement.NETWORK_NO_SOURCE &&
              element.readyState === HTMLMediaElement.HAVE_NOTHING;
            if (terminalError || exhaustedSources) {
              setFailed(true);
              setPlaying(false);
            }
          }}
        >
          <source
            src={`${base}-small.webm`}
            type="video/webm"
            media="(max-width: 767px)"
          />
          <source
            src={`${base}-small.mp4`}
            type="video/mp4"
            media="(max-width: 767px)"
          />
          <source src={`${base}.webm`} type="video/webm" />
          <source
            src={`${base}.mp4`}
            type="video/mp4"
            onError={() => {
              // The final alternative has no successor. Chromium can leave
              // play() pending here instead of rejecting with NotSupportedError.
              if (video.current?.readyState === HTMLMediaElement.HAVE_NOTHING) {
                setFailed(true);
                setPlaying(false);
              }
            }}
          />
          Eye anatomy: the outer eye, iris, lens and retinal layers separate and
          come together.
        </video>
      </div>
      <figcaption className="eye-film-caption">
        <div className="eye-film-description">
          <span className="eye-film-marker" aria-hidden="true" />
          <span>Eye anatomy</span>
          <span
            className={failed ? "eye-film-error" : "eye-film-duration"}
            role={failed ? "status" : undefined}
          >
            {failed ? "Film unavailable" : "20-second film"}
          </span>
        </div>
        <div
          className="eye-film-controls"
          role="group"
          aria-label="Eye film playback"
        >
          <button
            type="button"
            onClick={toggle}
            disabled={failed}
            aria-label={playing ? "Pause eye film" : "Play eye film"}
          >
            {playing ? <Pause size={15} /> : <Play size={15} />}
            <span>{playing ? "Pause" : "Play"}</span>
          </button>
          <button
            type="button"
            onClick={replay}
            disabled={failed}
            aria-label="Replay eye film"
          >
            <RotateCcw size={15} />
            <span>Replay</span>
          </button>
        </div>
      </figcaption>
    </figure>
  );
}
