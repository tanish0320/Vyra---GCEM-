import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import {
  AlertTriangle,
  LoaderCircle,
  MessageSquare,
  Send,
  X,
} from "lucide-react";
import type { AssistantResponse } from "../integration";
import { api, errorMessage } from "../services/api";
import { Button } from "./ui/button";
import { AssistantText } from "./AssistantText";

type Language = "en" | "hi";
type Message = {
  id: number;
  language: Language;
  role: "user" | "assistant";
  text: string;
  result?: AssistantResponse;
};
const COPY = {
  en: {
    title: "Ask about this screening",
    intro:
      "Ask the local assistant to explain this saved result. Each answer identifies its source.",
    question: "Your question",
    placeholder: "What does this screening result mean?",
    send: "Send question",
    cancel: "Cancel request",
    clear: "Clear conversation",
    empty: "Choose a question below or write your own.",
    waiting: "Preparing an answer…",
    you: "Your question",
    answer: "Assistant answer",
    unavailable:
      "The assistant is unavailable with this service. Your screening result and follow-up guidance remain available above.",
    notice:
      "Assistant responses may contain errors. They do not diagnose anemia or replace clinical assessment and laboratory testing.",
    timeout:
      "The assistant took too long to respond. Your question is still here; please try again.",
    cancelled: "Request cancelled. You can edit your question and try again.",
    switched:
      "The previous request was cancelled. New questions will use this language.",
    emergency: "Urgent guidance from the safety service",
    hotlines: "Emergency contacts returned by the service",
    qwen: "Local model · Qwen 2.5",
    deterministic: "Rule-based response · deterministic guardrail",
    safety: "Emergency safety response",
    backend: "Backend assistant",
    chips: [
      "What does this result mean?",
      "Explain the image and attention map.",
      "What should I discuss with a clinician?",
    ],
  },
  hi: {
    title: "इस स्क्रीनिंग के बारे में पूछें",
    intro:
      "इस सहेजे गए परिणाम को समझने के लिए स्थानीय सहायक से पूछें। हर उत्तर में उसका स्रोत दिखाया जाता है।",
    question: "आपका प्रश्न",
    placeholder: "इस स्क्रीनिंग परिणाम का क्या अर्थ है?",
    send: "प्रश्न भेजें",
    cancel: "अनुरोध रद्द करें",
    clear: "बातचीत साफ़ करें",
    empty: "नीचे कोई प्रश्न चुनें या अपना प्रश्न लिखें।",
    waiting: "उत्तर तैयार हो रहा है…",
    you: "आपका प्रश्न",
    answer: "सहायक का उत्तर",
    unavailable:
      "इस सेवा में सहायक उपलब्ध नहीं है। आपका स्क्रीनिंग परिणाम और आगे की सलाह ऊपर उपलब्ध हैं।",
    notice:
      "सहायक के उत्तरों में त्रुटियाँ हो सकती हैं। ये एनीमिया का निदान नहीं करते और चिकित्सकीय मूल्यांकन या प्रयोगशाला जाँच का विकल्प नहीं हैं।",
    timeout:
      "उत्तर आने में अधिक समय लगा। आपका प्रश्न अभी यहाँ है; कृपया फिर प्रयास करें।",
    cancelled: "अनुरोध रद्द किया गया। आप प्रश्न बदलकर फिर प्रयास कर सकते हैं।",
    switched:
      "पिछला अनुरोध रद्द किया गया। नए प्रश्नों के उत्तर इस भाषा में होंगे।",
    emergency: "सुरक्षा सेवा की तत्काल सलाह",
    hotlines: "सेवा द्वारा दिए गए आपातकालीन संपर्क",
    qwen: "स्थानीय मॉडल · Qwen 2.5",
    deterministic: "नियम-आधारित उत्तर · deterministic guardrail",
    safety: "आपातकालीन सुरक्षा उत्तर",
    backend: "बैकएंड सहायक",
    chips: [
      "इस परिणाम का क्या अर्थ है?",
      "चित्र और अटेंशन मैप समझाएँ।",
      "चिकित्सक से किन बातों पर चर्चा करूँ?",
    ],
  },
};

function sourceLabel(result: AssistantResponse, language: Language) {
  const copy = COPY[language];
  if (result.is_emergency || result.model_used === "emergency_firewall")
    return copy.safety;
  if (result.model_used === "deterministic_guardrail")
    return copy.deterministic;
  if (result.model_used.startsWith("qwen2.5")) return copy.qwen;
  return `${copy.backend} · ${result.model_used}`;
}

export function AssistantCard({
  scanId,
  enabled,
}: {
  scanId: string;
  enabled: boolean;
}) {
  const [language, setLanguage] = useState<Language>("en");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const generation = useRef(0);
  const sequence = useRef(0);
  const pending = useRef<{
    controller: AbortController;
    timer: ReturnType<typeof setTimeout>;
  } | null>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const stream = useRef<HTMLDivElement>(null);
  const focusAfterClear = useRef(false);
  const location = useLocation();
  const copy = COPY[language];
  const abort = useCallback(() => {
    generation.current++;
    if (pending.current) {
      clearTimeout(pending.current.timer);
      pending.current.controller.abort();
      pending.current = null;
    }
  }, []);
  useEffect(() => {
    abort();
    setBusy(false);
    setMessages([]);
    setQuestion("");
    setError("");
    return abort;
  }, [scanId, enabled, location.key, abort]);
  useEffect(() => {
    if (stream.current) stream.current.scrollTop = stream.current.scrollHeight;
    if (focusAfterClear.current && !busy && enabled) {
      input.current?.focus();
      focusAfterClear.current = false;
    }
  }, [messages, busy, enabled]);
  function clear() {
    abort();
    focusAfterClear.current = true;
    setBusy(false);
    setMessages([]);
    setError("");
    setQuestion("");
  }
  function changeLanguage(next: Language) {
    if (next === language) return;
    const wasPending = pending.current !== null;
    abort();
    setBusy(false);
    setLanguage(next);
    setQuestion("");
    setError(wasPending ? COPY[next].switched : "");
  }
  async function send(value = question) {
    const text = value.trim();
    if (!enabled || pending.current || !text || text.length > 1000) return;
    const token = ++generation.current;
    const requestLanguage = language;
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, 50_000);
    pending.current = { controller, timer };
    setQuestion(text);
    setError("");
    setBusy(true);
    const userMessage: Message = {
      id: sequence.current++,
      language: requestLanguage,
      role: "user",
      text,
    };
    setMessages((items) => [...items, userMessage]);
    try {
      const response = await api.explain(
        scanId,
        text,
        requestLanguage,
        controller.signal,
      );
      if (token !== generation.current || controller.signal.aborted) return;
      if (!response.response.trim())
        throw new Error(
          "The assistant returned an empty answer. Please try again.",
        );
      const assistantMessage: Message = {
        id: sequence.current++,
        language: requestLanguage,
        role: "assistant",
        text: response.response,
        result: response,
      };
      setMessages((items) => [...items, assistantMessage]);
      setQuestion("");
    } catch (cause) {
      if (token !== generation.current) return;
      setError(timedOut ? COPY[requestLanguage].timeout : errorMessage(cause));
    } finally {
      clearTimeout(timer);
      if (token === generation.current) {
        pending.current = null;
        setBusy(false);
      }
    }
  }
  return (
    <section
      id="assistantSection"
      className="integration-result-section assistant-card"
      aria-labelledby="assistantHeading"
      lang={language}
    >
      <div className="assistant-heading">
        <div>
          <span className="eyebrow">
            {language === "hi"
              ? "डॉ. VYRA · स्क्रीनिंग सहायक"
              : "DR. VYRA · SCREENING ASSISTANT"}
          </span>
          <h2 id="assistantHeading">
            <MessageSquare aria-hidden="true" size={23} />
            {copy.title}
          </h2>
        </div>
        <div
          className="assistant-languages"
          role="group"
          aria-label={language === "hi" ? "उत्तर की भाषा" : "Answer language"}
        >
          <button
            type="button"
            className="lang-btn"
            data-lang="en"
            lang="en"
            aria-pressed={language === "en"}
            onClick={() => changeLanguage("en")}
          >
            English
          </button>
          <button
            type="button"
            className="lang-btn"
            data-lang="hi"
            lang="hi"
            aria-pressed={language === "hi"}
            onClick={() => changeLanguage("hi")}
          >
            हिन्दी
          </button>
        </div>
      </div>
      <p>{enabled ? copy.intro : copy.unavailable}</p>
      <p className="assistant-boundary">{copy.notice}</p>
      <div
        id="assistantChatStream"
        ref={stream}
        className="assistant-chat-stream"
        role="log"
        aria-live="polite"
        aria-relevant="additions"
        aria-label={
          language === "hi" ? "स्क्रीनिंग बातचीत" : "Screening conversation"
        }
      >
        {!messages.length && enabled && (
          <p className="assistant-empty">{copy.empty}</p>
        )}
        {messages.map((message) => {
          const labels = COPY[message.language];
          const emergency = message.result?.is_emergency === true;
          const hotlines = Array.isArray(message.result?.hotlines)
            ? message.result.hotlines.filter(
                (item): item is string => typeof item === "string",
              )
            : [];
          return (
            <article
              key={message.id}
              lang={message.language}
              className={`assistant-message assistant-message-${message.role}${emergency ? " assistant-emergency" : ""}`}
              role={emergency ? "alert" : undefined}
            >
              <div className="assistant-message-label">
                <strong>
                  {message.role === "user" ? labels.you : labels.answer}
                </strong>
                <span>{message.language === "hi" ? "हिन्दी" : "English"}</span>
              </div>
              {emergency && (
                <h3>
                  <AlertTriangle aria-hidden="true" size={21} />
                  {labels.emergency}
                </h3>
              )}
              {message.role === "assistant" ? (
                <AssistantText text={message.text} />
              ) : (
                <p className="assistant-user-text">{message.text}</p>
              )}
              {hotlines.length > 0 && emergency && (
                <div className="assistant-hotlines">
                  <strong>{labels.hotlines}</strong>
                  <ul>
                    {hotlines.map((hotline, index) => (
                      <li key={index}>{hotline}</li>
                    ))}
                  </ul>
                </div>
              )}
              {message.result && (
                <p
                  className="assistant-source"
                  data-source={message.result.model_used}
                >
                  {sourceLabel(message.result, message.language)}
                </p>
              )}
            </article>
          );
        })}
      </div>
      {busy && (
        <p className="assistant-pending" role="status">
          <LoaderCircle className="spin" aria-hidden="true" size={18} />
          {copy.waiting}
        </p>
      )}
      {error && (
        <p className="integration-error" role="alert">
          {error}
        </p>
      )}
      <div
        id="assistantChips"
        className="assistant-chips"
        aria-label={
          language === "hi" ? "सुझाए गए प्रश्न" : "Suggested questions"
        }
      >
        {copy.chips.map((chip) => (
          <button
            type="button"
            key={chip}
            disabled={!enabled || busy}
            onClick={() => void send(chip)}
          >
            {chip}
          </button>
        ))}
      </div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void send();
        }}
      >
        <label htmlFor="assistantInput">{copy.question}</label>
        <textarea
          ref={input}
          id="assistantInput"
          rows={3}
          maxLength={1000}
          required
          disabled={!enabled || busy}
          value={question}
          placeholder={copy.placeholder}
          onChange={(event) => setQuestion(event.target.value)}
        />
        <div className="assistant-form-actions">
          <Button
            type="submit"
            id="assistantSendBtn"
            disabled={!enabled || busy || !question.trim()}
          >
            <Send aria-hidden="true" size={17} />
            {copy.send}
          </Button>
          {busy && (
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                abort();
                setBusy(false);
                setError(copy.cancelled);
              }}
            >
              <X aria-hidden="true" size={17} />
              {copy.cancel}
            </Button>
          )}
          {(messages.length > 0 || error) && (
            <Button type="button" variant="ghost" onClick={clear}>
              {copy.clear}
            </Button>
          )}
        </div>
      </form>
    </section>
  );
}
