import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api, errorMessage, RequestError } from "./services/api";
import type {
  Box,
  CaseDetails,
  Job,
  Modality,
  Scan,
  SystemStatus,
} from "./types";
const ACTIVE_KEY = "vyra.active-job.v1";
const terminal = new Set(["completed", "failed", "cancelled", "interrupted"]);
function savedJob() {
  try {
    return sessionStorage.getItem(ACTIVE_KEY);
  } catch {
    return null;
  }
}
function remember(id: string | null) {
  try {
    if (id) sessionStorage.setItem(ACTIVE_KEY, id);
    else sessionStorage.removeItem(ACTIVE_KEY);
  } catch {}
}
interface SessionValue {
  system: SystemStatus | null;
  serviceError: string | null;
  refreshSystem: () => Promise<void>;
  job: Job | null;
  scan: Scan | null;
  busy: boolean;
  error: string | null;
  restoring: boolean;
  upload: (
    file: File,
    modality: Modality,
    details: CaseDetails,
    secondaryImage?: File,
  ) => Promise<void>;
  confirm: (roi: Box, secondaryRoi?: Box) => Promise<void>;
  cancel: () => Promise<void>;
  reset: () => void;
  retry: () => void;
  clearError: () => void;
  openJob: (id: string) => Promise<void>;
}
const SessionContext = createContext<SessionValue | null>(null);
export function SessionProvider({ children }: { children: ReactNode }) {
  const [system, setSystem] = useState<SystemStatus | null>(null),
    [serviceError, setServiceError] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null),
    [scan, setScan] = useState<Scan | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState<string | null>(null);
  const [restoring, setRestoring] = useState(Boolean(savedJob())),
    [retryCount, setRetryCount] = useState(0);
  const [pollRevision, setPollRevision] = useState(0);
  const generation = useRef(0),
    lock = useRef(false),
    mounted = useRef(true);
  const uploadIdentity = useRef<{
    file: File;
    secondaryImage?: File;
    signature: string;
    key: string;
  } | null>(null);
  const openJob = useCallback(async (id: string) => {
    const token = ++generation.current;
    remember(id);
    setRestoring(true);
    setError(null);
    setScan(null);
    setJob(null);
    try {
      const value = await api.job(id);
      if (token === generation.current && mounted.current) setJob(value);
    } catch (e) {
      if (token === generation.current && mounted.current) {
        if (e instanceof RequestError && e.status === 404) remember(null);
        setError(errorMessage(e));
      }
    } finally {
      if (token === generation.current && mounted.current) setRestoring(false);
    }
  }, []);
  const refreshSystem = async () => {
    try {
      const value = await api.system();
      if (mounted.current) {
        setSystem(value);
        setServiceError(null);
      }
    } catch (e) {
      if (mounted.current) {
        setSystem(null);
        setServiceError(errorMessage(e));
      }
    }
  };
  useEffect(() => {
    mounted.current = true;
    void refreshSystem();
    const id = savedJob(),
      controller = new AbortController();
    const token = generation.current;
    if (id)
      api
        .job(id, controller.signal)
        .then((value) => {
          if (
            controller.signal.aborted ||
            token !== generation.current ||
            savedJob() !== id
          )
            return;
          setJob(value);
          setError(null);
          setRestoring(false);
        })
        .catch((e) => {
          if (controller.signal.aborted || token !== generation.current) return;
          if (e instanceof RequestError && e.status === 404) remember(null);
          setError(errorMessage(e));
          setRestoring(false);
        });
    return () => {
      mounted.current = false;
      controller.abort();
    };
  }, [retryCount]);
  useEffect(() => {
    if (!job) return;
    remember(job.id);
    if (terminal.has(job.status) || job.status === "awaiting_roi") return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const token = generation.current;
    const poll = async () => {
      try {
        const next = await api.job(job.id, controller.signal);
        if (controller.signal.aborted || token !== generation.current) return;
        setJob(next);
        setError(null);
      } catch (e) {
        if (controller.signal.aborted || token !== generation.current) return;
        setError(errorMessage(e));
      } finally {
        if (!controller.signal.aborted && token === generation.current)
          timer = setTimeout(poll, 750);
      }
    };
    timer = setTimeout(poll, 300);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [job?.id, job?.status, retryCount, pollRevision]);
  useEffect(() => {
    if (job?.status !== "completed" || !job.scan_id) return;
    const controller = new AbortController();
    const token = generation.current;
    api
      .scan(job.scan_id, controller.signal)
      .then((value) => {
        if (!controller.signal.aborted && token === generation.current) {
          setScan(value);
        }
      })
      .catch((e) => {
        if (!controller.signal.aborted && token === generation.current) {
          setError(errorMessage(e));
        }
      });
    return () => controller.abort();
  }, [job?.status, job?.scan_id, retryCount]);
  const action = async (work: () => Promise<Job>) => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError(null);
    const token = ++generation.current;
    try {
      const next = await work();
      if (token === generation.current && mounted.current) setJob(next);
    } catch (e) {
      if (token === generation.current && mounted.current)
        setError(errorMessage(e));
    } finally {
      lock.current = false;
      if (mounted.current) {
        setBusy(false);
        setPollRevision((value) => value + 1);
      }
    }
  };
  const upload = async (
    file: File,
    modality: Modality,
    details: CaseDetails,
    secondaryImage?: File,
  ) => {
    setScan(null);
    const signature = JSON.stringify({ modality, details });
    if (
      uploadIdentity.current?.file !== file ||
      uploadIdentity.current.secondaryImage !== secondaryImage ||
      uploadIdentity.current.signature !== signature
    ) {
      uploadIdentity.current = {
        file,
        secondaryImage,
        signature,
        key: crypto.randomUUID(),
      };
    }
    const key = uploadIdentity.current.key;
    await action(() =>
      api.upload(file, modality, details, key, secondaryImage),
    );
  };
  const confirm = async (roi: Box, secondaryRoi?: Box) => {
    if (job?.status === "awaiting_roi")
      await action(() => api.confirm(job.id, roi, secondaryRoi));
  };
  const cancel = async () => {
    if (job && !terminal.has(job.status))
      await action(() => api.cancel(job.id));
  };
  const reset = () => {
    if (busy || (job && !terminal.has(job.status))) return;
    generation.current++;
    setJob(null);
    setScan(null);
    setError(null);
    setRestoring(false);
    uploadIdentity.current = null;
    remember(null);
  };
  return (
    <SessionContext.Provider
      value={{
        system,
        serviceError,
        refreshSystem,
        job,
        scan,
        busy,
        error,
        restoring,
        upload,
        confirm,
        cancel,
        reset,
        retry: () => setRetryCount((v) => v + 1),
        clearError: () => setError(null),
        openJob,
      }}
    >
      {children}
    </SessionContext.Provider>
  );
}
export function useSession() {
  const value = useContext(SessionContext);
  if (!value) throw new Error("Screening session provider missing");
  return value;
}
