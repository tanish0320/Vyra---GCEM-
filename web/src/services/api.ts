import type {
  Analytics,
  Box,
  CaseDetails,
  Evidence,
  Job,
  Modality,
  Scan,
  SystemStatus,
} from "../types";
import type {
  AssistantResponse,
  ResearchOverview,
  StressReport,
} from "../integration";
const ROOT = "/api/v1";
export class RequestError extends Error {
  code: string;
  status: number;
  constructor(message: string, code: string, status: number) {
    super(message);
    this.code = code;
    this.status = status;
  }
}
async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${ROOT}${path}`, options);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw error;
    throw new RequestError(
      "The local service is not reachable. Start Vyra, then try again.",
      "NETWORK",
      0,
    );
  }
  if (!response.ok) {
    let error: { error?: string | { message?: string; code?: string } } = {};
    try {
      error = await response.json();
    } catch {}
    throw new RequestError(
      (typeof error.error === "string" ? error.error : error.error?.message) ||
        `Request failed (${response.status}).`,
      (typeof error.error === "object" ? error.error?.code : null) ||
        "REQUEST_FAILED",
      response.status,
    );
  }
  if (response.status === 204) return undefined as T;
  try {
    return (await response.json()) as T;
  } catch {
    throw new RequestError(
      "The service returned an unexpected response. Please restart Vyra.",
      "INVALID_RESPONSE",
      response.status,
    );
  }
}
export const api = {
  system: (signal?: AbortSignal) =>
    request<SystemStatus>("/system", { signal }),
  evidence: (signal?: AbortSignal) =>
    request<Evidence>("/evidence", { signal }),
  analytics: (signal?: AbortSignal) =>
    request<Analytics>("/analytics", { signal }),
  job: (id: string, signal?: AbortSignal) =>
    request<Job>(`/jobs/${encodeURIComponent(id)}`, { signal }),
  upload: (
    image: File,
    modality: Modality,
    details: CaseDetails,
    key: string,
    secondaryImage?: File,
  ) => {
    const body = new FormData();
    body.set("image", image);
    body.set("modality", modality);
    if (secondaryImage) body.set("secondary_image", secondaryImage);
    for (const [name, value] of Object.entries(details))
      if (value) body.set(name, value);
    return request<Job>("/jobs", {
      method: "POST",
      body,
      headers: { "Idempotency-Key": key },
    });
  },
  confirm: (id: string, roi: Box, secondaryRoi?: Box) =>
    request<Job>(`/jobs/${encodeURIComponent(id)}/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        roi,
        ...(secondaryRoi ? { secondary_roi: secondaryRoi } : {}),
      }),
    }),
  cancel: (id: string) =>
    request<Job>(`/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
  scans: (signal?: AbortSignal) =>
    request<{ scans: Scan[] }>("/scans", { signal }),
  scan: (id: string, signal?: AbortSignal) =>
    request<Scan>(`/scans/${encodeURIComponent(id)}`, { signal }),
  remove: (id: string) =>
    request<void>(`/scans/${encodeURIComponent(id)}`, { method: "DELETE" }),
  report: (id: string) => `${ROOT}/scans/${encodeURIComponent(id)}/report.pdf`,
  csv: `${ROOT}/scans.csv`,
  explain: async (
    id: string,
    question: string,
    language: "en" | "hi",
    signal?: AbortSignal,
  ) => {
    const value = await request<AssistantResponse>(
      `/explain/${encodeURIComponent(id)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, language }),
        signal,
      },
    );
    if (
      typeof value?.response !== "string" ||
      typeof value.is_emergency !== "boolean" ||
      typeof value.model_used !== "string"
    ) {
      throw new RequestError(
        "The assistant returned an incomplete response. Please retry.",
        "INVALID_RESPONSE",
        502,
      );
    }
    return value;
  },
  stressTest: (image: File, modality: Modality, signal?: AbortSignal) => {
    const body = new FormData();
    body.set("image", image);
    body.set("modality", modality);
    return request<StressReport>("/achilles-heel", {
      method: "POST",
      body,
      signal,
    });
  },
  research: (signal?: AbortSignal) =>
    request<ResearchOverview>("/research", { signal }),
  sample: async (modality: Modality) => {
    const response = await fetch(`${ROOT}/samples/${modality}`);
    if (!response.ok)
      throw new Error(
        "The reference photo is unavailable. Please upload a photo instead.",
      );
    return new File([await response.blob()], `reference-${modality}.png`, {
      type: "image/png",
    });
  },
};
export const errorMessage = (error: unknown) =>
  error instanceof Error
    ? error.message
    : "Something went wrong. Please try again.";
