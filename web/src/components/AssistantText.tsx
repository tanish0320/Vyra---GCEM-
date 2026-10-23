import { Fragment, type ReactNode } from "react";

/** Deliberately small Markdown subset. All content stays in React text nodes. */
function inline(text: string): ReactNode[] {
  return text
    .split(/(\*\*[^*\n]+\*\*)/g)
    .map((part, index) =>
      part.startsWith("**") && part.endsWith("**") ? (
        <strong key={index}>{part.slice(2, -2)}</strong>
      ) : (
        <Fragment key={index}>{part}</Fragment>
      ),
    );
}

export function AssistantText({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  let paragraph: string[] = [];
  let bullets: string[] = [];
  function flushParagraph() {
    if (!paragraph.length) return;
    const value = paragraph.join("\n");
    const notice = /^(?:⚠(?:️)?|disclaimer\b|notice\b|अस्वीकरण|चेतावनी)/i.test(
      value.trim().replace(/^\*\*/, ""),
    );
    blocks.push(
      <p
        key={blocks.length}
        className={notice ? "assistant-disclaimer" : undefined}
      >
        {inline(value)}
      </p>,
    );
    paragraph = [];
  }
  function flushBullets() {
    if (!bullets.length) return;
    blocks.push(
      <ul key={blocks.length}>
        {bullets.map((value, index) => (
          <li key={index}>{inline(value)}</li>
        ))}
      </ul>,
    );
    bullets = [];
  }
  for (const line of text.replaceAll("\r\n", "\n").split("\n")) {
    const bullet = /^\s*(?:[-*•]|\d+[.)])\s+(.+)$/.exec(line);
    if (bullet) {
      flushParagraph();
      bullets.push(bullet[1]);
    } else if (!line.trim()) {
      flushParagraph();
      flushBullets();
    } else {
      flushBullets();
      paragraph.push(line);
    }
  }
  flushParagraph();
  flushBullets();
  return <div className="assistant-answer-text">{blocks}</div>;
}
