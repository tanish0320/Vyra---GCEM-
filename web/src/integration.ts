/** Additive UI contracts. Existing Integration routes and Scan columns stay unchanged. */
export interface IntegrationCapabilities {
  dual_scan?: boolean;
  assistant?: boolean;
  stress_test?: boolean;
  colorimetry?: boolean;
  research?: boolean;
}
export interface BiophysicalMetrics {
  has_real_pixels: boolean;
  erythema_index: number | null;
  l_star: number | null;
  a_star: number | null;
  b_star: number | null;
  source?: string;
}
/** Legacy fusion probabilities are percentages, unlike Scan.probability_anemia. */
export interface MultimodalFusion {
  active: boolean;
  primary_modality: string;
  primary_prob: number;
  primary_class: string;
  secondary_modality: string;
  secondary_prob: number;
  secondary_class: string;
  secondary_image_url?: string | null;
  joint_probability: number;
  concordance_pct: number;
  agreement_level: string;
  method?: string;
  secondary_error?: string | null;
}
export interface AssistantResponse {
  is_emergency: boolean;
  response: string;
  model_used: string;
  rule?: string;
  hotlines?: string[];
  grounded_facts?: Record<string, unknown>;
}
export interface StressCondition {
  condition_id: string;
  name: string;
  category: string;
  probability: number;
  predicted_class: string;
  delta_prob: number;
  percent_shift: number;
  thumbnail_b64: string;
  description: string;
}
export interface StressMechanism {
  id: string;
  name: string;
  risk_level: string;
  sensitivity_score: number;
  evidence: string;
  mechanism_detail: string;
  clinical_implication: string;
}
export interface StressReport {
  modality: string;
  baseline_probability: number;
  baseline_class: string;
  conditions: StressCondition[];
  ranked_failure_mechanisms: StressMechanism[];
  overall_shortcut_index: number;
  overall_status: string;
  overall_class: string;
  big_reveal_note: string;
}
export interface ResearchOverview {
  summary: string;
  datasets: { name: string; description: string; count: number | null }[];
  plots: { title: string; url: string; description: string }[];
}
