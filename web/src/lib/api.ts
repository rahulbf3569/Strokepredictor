export type PatientInput = {
  gender: "Male" | "Female";
  age: number;
  hypertension: 0 | 1;
  heart_disease: 0 | 1;
  ever_married: "Yes" | "No";
  work_type: "Private" | "Self-employed" | "Govt_job" | "children" | "Never_worked";
  Residence_type: "Urban" | "Rural";
  avg_glucose_level: number;
  bmi: number;
  smoking_status: "never smoked" | "formerly smoked" | "smokes" | "Unknown";
};

export type PredictionResponse = {
  stroke_probability: number;
  stroke_risk: "high" | "low";
  threshold: number;
  model: string;
};

export type HealthResponse = {
  status: string;
  model_loaded: boolean;
};

export type ModelMetadata = {
  best_model: string;
  threshold: number;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc: number;
    avg_precision: number;
    confusion_matrix: number[][];
    threshold: number;
    best_model: string;
  };
  all_models: Record<
    string,
    {
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
      roc_auc: number;
      avg_precision: number;
      cv_avg_precision_mean?: number;
      cv_avg_precision_std?: number;
    }
  >;
  features: { numeric: string[]; categorical: string[] };
};

const BASE = process.env.NEXT_PUBLIC_API_URL || "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  predict: (payload: PatientInput) =>
    request<PredictionResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  metadata: () => request<ModelMetadata>("/metadata"),
};