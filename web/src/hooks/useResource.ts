import { useEffect, useState } from "react";
import { errorMessage } from "../services/api";
export function useResource<T>(fetcher: (signal: AbortSignal) => Promise<T>) {
  const [value, setValue] = useState<T | null>(null),
    [error, setError] = useState<string | null>(null),
    [loading, setLoading] = useState(true),
    [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    fetcher(controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        setValue(data);
        setError(null);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(errorMessage(e));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [fetcher, revision]);
  return { value, error, loading, reload: () => setRevision((n) => n + 1) };
}
