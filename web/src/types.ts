import type {
  BiophysicalMetrics,
  IntegrationCapabilities,
  MultimodalFusion,
} from "./integration";
export type Modality = "conjunctiva" | "fingernail";
export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}
export interface Photo {
  url: string;
  name: string;
  width: number;
  height: number;
}
export interface CaseDetails {
  reference: string;
  age: string;
  sex: string;
  phone: string;
  notes: string;
}
export interface PatientDetails {
  reference: string;
  age: number | null;
  sex: "female" | "male" | "not-stated";
  phone: string;
  notes: string;
}
export interface FollowUp {
  version: 1;
  title: string;
  summary: string;
  steps: string[];
  model_input_note: string;
}
export interface ApiError {
  code: string;
  message: string;
}
export interface SystemStatus {
  capabilities?: IntegrationCapabilities;
  api_version: string;
  patient_intake_version?: number;
  ready: boolean;
  model: {
    name: string;
    sha256: string;
    temperature: number;
    regression_supported: boolean;
  };
  detector: { available: boolean };
  limits: { max_bytes: number; min_side: number };
  warnings: string[];
}
export type JobStatus =
  | "queued"
  | "detecting"
  | "awaiting_roi"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted";
export type StageName =
  | "validation"
  | "detection"
  | "roi"
  | "normalization"
  | "features"
  | "inference"
  | "explanation"
  | "saving";
export interface Job {
  dual_scan?: boolean;
  secondary?: {
    modality: Modality;
    image: { url: string; width: number; height: number } | null;
    detection: Job["detection"];
    roi: Box | null;
    error?: ApiError | null;
  } | null;
  id: string;
  status: JobStatus;
  stage: StageName;
  modality: Modality;
  created_at: string;
  updated_at: string;
  elapsed_seconds: number;
  image: { url: string; width: number; height: number } | null;
  detection: {
    success: boolean;
    roi: Box | null;
    type: string;
    confidence: number;
    message: string;
    retake_guidance: string | null;
  } | null;
  roi: Box | null;
  warnings: string[];
  stages: { name: StageName; status: string; elapsed_seconds?: number }[];
  scan_id: string | null;
  error: ApiError | null;
  patient: PatientDetails;
}
export interface Scan {
  biophysical_metrics?: BiophysicalMetrics | null;
  multimodal_fusion?: MultimodalFusion | null;
  pdf_url?: string | null;
  id: string;
  job_id: string;
  created_at: string;
  reference: string;
  age: number | null;
  sex: string;
  phone?: string;
  notes?: string;
  follow_up?: FollowUp;
  modality: Modality;
  probability_anemia: number;
  predicted_label: string;
  hb_g_dl: number | null;
  regression_status: string;
  threshold: number;
  temperature: number;
  model_sha256: string;
  normalization_version: string;
  roi: Box;
  image: { width: number; height: number };
  warnings: string[];
  artifacts: {
    oriented: string;
    roi: string;
    normalized: string;
    gradcam: string | null;
    heatmap?: string | null;
  };
  explanation: { status: string; message: string };
  features: number[];
  processing_seconds: number;
}
export interface Evidence {
  model_version: string;
  checkpoint_sha256: string;
  normalization_version: string;
  temperature: number;
  test_dataset_size: number;
  classification: {
    accuracy: number;
    sensitivity_recall: number;
    specificity: number;
    precision: number;
    f1_score: number;
    roc_auc: number;
    brier_score: number;
    confusion_matrix: { TN: number; FP: number; FN: number; TP: number };
  };
  hemoglobin_regression: {
    paired_samples: number;
    mae_g_dl: number;
    rmse_g_dl: number;
    r2_score: number;
    bland_altman: {
      mean_bias_g_dl: number;
      sd_bias: number;
      loa_upper_95_g_dl: number;
      loa_lower_95_g_dl: number;
    };
  };
  provenance: string;
  limitations: string[];
}
export interface Analytics {
  total_scans: number;
  by_modality: Record<Modality, number>;
  screen_positive: number;
  screen_negative: number;
  mean_probability: number | null;
  eye_hb_count: number;
  mean_eye_hb: number | null;
}
