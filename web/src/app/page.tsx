"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  HeartPulse,
  Activity,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  ChevronRight,
  Loader2,
} from "lucide-react";
import { api, type PatientInput, type PredictionResponse, type ModelMetadata, type HealthResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

type Tab = "predict" | "insights" | "about";

const WORK_TYPES = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"] as const;
const SMOKING = ["never smoked", "formerly smoked", "smokes", "Unknown"] as const;

const defaultInput: PatientInput = {
  gender: "Male",
  age: 50,
  hypertension: 0,
  heart_disease: 0,
  ever_married: "Yes",
  work_type: "Private",
  Residence_type: "Urban",
  avg_glucose_level: 110,
  bmi: 28,
  smoking_status: "never smoked",
};

export default function Home() {
  const [tab, setTab] = useState<Tab>("predict");
  const [input, setInput] = useState<PatientInput>(defaultInput);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<ModelMetadata | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: "offline", model_loaded: false }));
    api.metadata().then(setMeta).catch(() => {});
  }, []);

  const update = <K extends keyof PatientInput>(key: K, value: PatientInput[K]) =>
    setInput((s) => ({ ...s, [key]: value }));

  const submit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.predict(input);
      setResult(r);
    } catch (e: any) {
      setError(e?.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <Header tab={tab} setTab={setTab} health={health} />
      <main className="mx-auto max-w-7xl px-4 pb-24 pt-6 sm:px-6 lg:px-8">
        <AnimatePresence mode="wait">
          {tab === "predict" && (
            <motion.div
              key="predict"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.35 }}
            >
              <Hero onInsights={() => setTab("insights")} />
              <div className="mt-10 grid gap-6 lg:grid-cols-5">
                <div className="lg:col-span-3">
                  <PredictCard
                    input={input}
                    update={update}
                    submit={submit}
                    loading={loading}
                    onReset={() => {
                      setInput(defaultInput);
                      setResult(null);
                      setError(null);
                    }}
                  />
                </div>
                <div className="lg:col-span-2">
                  <ResultCard result={result} loading={loading} error={error} input={input} />
                </div>
              </div>
              <FeatureStrip />
            </motion.div>
          )}
          {tab === "insights" && (
            <motion.div
              key="insights"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.35 }}
            >
              <Insights meta={meta} />
            </motion.div>
          )}
          {tab === "about" && (
            <motion.div
              key="about"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.35 }}
            >
              <About />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
      <Footer />
    </div>
  );
}

function Header({
  tab,
  setTab,
  health,
}: {
  tab: Tab;
  setTab: (t: Tab) => void;
  health: HealthResponse | null;
}) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "predict", label: "Predict" },
    { id: "insights", label: "Model Insights" },
    { id: "about", label: "About" },
  ];
  return (
    <header className="sticky top-0 z-40 border-b border-white/5 bg-[#050816]/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-accent to-accent-2 glow">
            <Brain className="h-5 w-5 text-white" />
          </div>
          <div>
            <div className="text-sm font-semibold tracking-tight">StrokeRisk AI</div>
            <div className="text-xs text-muted">MLOps · Healthcare</div>
          </div>
        </div>
        <nav className="hidden items-center gap-1 rounded-full border border-white/10 bg-white/5 p-1 sm:flex">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={cn(
                "rounded-full px-4 py-1.5 text-sm transition",
                tab === t.id
                  ? "bg-white/10 text-white shadow"
                  : "text-muted hover:text-white"
              )}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs",
              health?.model_loaded
                ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
                : "border-rose-400/30 bg-rose-400/10 text-rose-300"
            )}
          >
            <span
              className={cn(
                "h-2 w-2 rounded-full",
                health?.model_loaded ? "bg-emerald-400" : "bg-rose-400"
              )}
            />
            {health?.model_loaded ? "Model online" : "Model offline"}
          </div>
        </div>
      </div>
      <div className="flex items-center justify-center gap-1 border-t border-white/5 px-4 py-2 sm:hidden">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={cn(
              "rounded-full px-3 py-1 text-xs transition",
              tab === t.id ? "bg-white/10 text-white" : "text-muted"
            )}
          >
            {t.label}
          </button>
        ))}
      </div>
    </header>
  );
}

function Hero({ onInsights }: { onInsights: () => void }) {
  return (
    <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-white/[0.04] to-transparent p-8 sm:p-12">
      <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-accent/20 blur-3xl" />
      <div className="absolute -bottom-20 -left-20 h-64 w-64 rounded-full bg-accent-2/20 blur-3xl" />
      <div className="relative grid gap-8 lg:grid-cols-2 lg:items-center">
        <div>
          <div className="inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accent/10 px-3 py-1 text-xs text-accent-2">
            <Sparkles className="h-3.5 w-3.5" />
            AI-powered clinical screening
          </div>
          <h1 className="mt-4 text-4xl font-bold tracking-tight sm:text-5xl">
            Predict stroke risk
            <span className="block bg-gradient-to-r from-accent via-accent-2 to-pink-400 bg-clip-text text-transparent">
              in seconds, with confidence.
            </span>
          </h1>
          <p className="mt-4 max-w-xl text-base text-muted">
            Enter patient demographics, lifestyle and clinical features. Our tuned ML model
            returns a calibrated stroke probability with a clinically-aligned threshold.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <a
              href="#predict-form"
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-accent to-accent-2 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-accent/30 transition hover:opacity-90"
            >
              Start assessment <ChevronRight className="h-4 w-4" />
            </a>
            <button
              onClick={onInsights}
              className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-white/10"
            >
              View model metrics
            </button>
          </div>
        </div>
        <div className="relative hidden lg:block">
          <BrainVisual />
        </div>
      </div>
    </section>
  );
}

function BrainVisual() {
  return (
    <div className="relative mx-auto h-64 w-64">
      <div className="absolute inset-0 rounded-full bg-gradient-to-br from-accent/40 to-accent-2/40 blur-2xl" />
      <div className="relative flex h-full w-full items-center justify-center rounded-full border border-white/10 bg-white/5 backdrop-blur">
        <HeartPulse className="h-24 w-24 text-accent-2 float" />
      </div>
      <div className="absolute -left-4 top-8 rounded-2xl glass px-3 py-2 text-xs">
        <div className="text-muted">Recall</div>
        <div className="font-semibold text-emerald-300">71.9%</div>
      </div>
      <div className="absolute -right-4 bottom-8 rounded-2xl glass px-3 py-2 text-xs">
        <div className="text-muted">ROC-AUC</div>
        <div className="font-semibold text-accent-2">0.845</div>
      </div>
    </div>
  );
}

function PredictCard({
  input,
  update,
  submit,
  loading,
  onReset,
}: {
  input: PatientInput;
  update: <K extends keyof PatientInput>(key: K, value: PatientInput[K]) => void;
  submit: () => void;
  loading: boolean;
  onReset: () => void;
}) {
  return (
    <div id="predict-form" className="glass rounded-3xl p-6 sm:p-8">
      <div className="mb-6 flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent/15 text-accent-2">
          <Stethoscope className="h-5 w-5" />
        </div>
        <div>
          <h2 className="text-lg font-semibold">Patient Assessment</h2>
          <p className="text-xs text-muted">All fields required for an accurate score.</p>
        </div>
      </div>

      <div className="grid gap-5 sm:grid-cols-2">
        <Field label="Gender">
          <Select value={input.gender} onChange={(v) => update("gender", v as any)}>
            <option>Male</option>
            <option>Female</option>
          </Select>
        </Field>
        <Field label={`Age — ${input.age}`}>
          <Slider min={1} max={100} value={input.age} onChange={(v) => update("age", v)} />
        </Field>
        <Field label="Hypertension">
          <Toggle value={input.hypertension} onChange={(v) => update("hypertension", v as any)} />
        </Field>
        <Field label="Heart disease">
          <Toggle value={input.heart_disease} onChange={(v) => update("heart_disease", v as any)} />
        </Field>
        <Field label="Ever married">
          <Select value={input.ever_married} onChange={(v) => update("ever_married", v as any)}>
            <option>Yes</option>
            <option>No</option>
          </Select>
        </Field>
        <Field label="Work type">
          <Select value={input.work_type} onChange={(v) => update("work_type", v as any)}>
            {WORK_TYPES.map((w) => (
              <option key={w}>{w}</option>
            ))}
          </Select>
        </Field>
        <Field label="Residence type">
          <Select value={input.Residence_type} onChange={(v) => update("Residence_type", v as any)}>
            <option>Urban</option>
            <option>Rural</option>
          </Select>
        </Field>
        <Field label={`Avg glucose — ${input.avg_glucose_level.toFixed(0)}`}>
          <Slider min={50} max={300} step={1} value={input.avg_glucose_level} onChange={(v) => update("avg_glucose_level", v)} />
        </Field>
        <Field label={`BMI — ${input.bmi.toFixed(1)}`}>
          <Slider min={10} max={60} step={0.1} value={input.bmi} onChange={(v) => update("bmi", v)} />
        </Field>
        <Field label="Smoking status">
          <Select value={input.smoking_status} onChange={(v) => update("smoking_status", v as any)}>
            {SMOKING.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </Select>
        </Field>
      </div>

      <div className="mt-7 flex gap-3">
        <button
          onClick={submit}
          disabled={loading}
          className="flex flex-1 items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-accent to-accent-2 px-6 py-3.5 text-sm font-semibold text-white shadow-lg shadow-accent/30 transition hover:opacity-95 disabled:opacity-60"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" /> Analyzing…
            </>
          ) : (
            <>
              <Activity className="h-4 w-4" /> Run prediction
            </>
          )}
        </button>
        <button
          onClick={onReset}
          disabled={loading}
          className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3.5 text-sm font-medium text-muted transition hover:bg-white/10 hover:text-white disabled:opacity-60"
        >
          Reset
        </button>
      </div>
    </div>
  );
}

function ResultCard({
  result,
  loading,
  error,
  input,
}: {
  result: PredictionResponse | null;
  loading: boolean;
  error: string | null;
  input: PatientInput;
}) {
  return (
    <div className="glass rounded-3xl p-6 sm:p-8">
      <div className="mb-4 flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent-2/15 text-accent-2">
          <Activity className="h-5 w-5" />
        </div>
        <div>
          <h2 className="text-lg font-semibold">Risk Assessment</h2>
          <p className="text-xs text-muted">Model output & calibrated decision.</p>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {loading && (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex h-64 flex-col items-center justify-center gap-3"
          >
            <Loader2 className="h-10 w-10 animate-spin text-accent-2" />
            <p className="text-sm text-muted">Running inference…</p>
          </motion.div>
        )}
        {!loading && error && (
          <motion.div
            key="error"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="rounded-2xl border border-rose-400/30 bg-rose-400/10 p-4 text-sm text-rose-200"
          >
            {error}
            <div className="mt-2 text-xs text-rose-300/70">
              Is the API running? Start with{" "}
              <code className="rounded bg-black/30 px-1">uvicorn api.main:app --port 8000</code>
            </div>
          </motion.div>
        )}
        {!loading && !error && !result && (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex h-64 flex-col items-center justify-center gap-3 text-center"
          >
            <div className="flex h-16 w-16 items-center justify-center rounded-full border border-white/10 bg-white/5">
              <HeartPulse className="h-7 w-7 text-muted" />
            </div>
            <p className="text-sm text-muted">
              Submit the form to see the stroke probability and risk band.
            </p>
          </motion.div>
        )}
        {!loading && result && (
          <motion.div
            key="result"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
            className="space-y-5"
          >
            <ProbabilityRing value={result.stroke_probability} risk={result.stroke_risk} />
            <div
              className={cn(
                "rounded-2xl border p-4 text-center",
                result.stroke_risk === "high"
                  ? "border-rose-400/40 bg-rose-400/10"
                  : "border-emerald-400/40 bg-emerald-400/10"
              )}
            >
              <div
                className={cn(
                  "text-lg font-bold",
                  result.stroke_risk === "high" ? "text-rose-300" : "text-emerald-300"
                )}
              >
                {result.stroke_risk === "high" ? "⚠ HIGH RISK" : "✓ LOW RISK"}
              </div>
              <div className="mt-1 text-xs text-muted">
                Threshold {result.threshold} · Model {result.model}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <Detail label="Probability" value={`${(result.stroke_probability * 100).toFixed(2)}%`} />
              <Detail label="Decision threshold" value={result.threshold.toString()} />
              <Detail label="Model" value={result.model} />
              <Detail label="Age / BMI" value={`${input.age} / ${input.bmi.toFixed(1)}`} />
            </div>
            <details className="rounded-xl border border-white/10 bg-black/20 p-3 text-xs">
              <summary className="cursor-pointer text-muted">Raw response</summary>
              <pre className="mt-2 overflow-auto text-[11px] text-white/70">
                {JSON.stringify(result, null, 2)}
              </pre>
            </details>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ProbabilityRing({ value, risk }: { value: number; risk: "high" | "low" }) {
  const pct = Math.round(value * 100);
  const color = risk === "high" ? "#ff5a7a" : "#34d399";
  const r = 52;
  const c = 2 * Math.PI * r;
  const offset = c - value * c;
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative h-40 w-40">
        <svg viewBox="0 0 120 120" className="-rotate-90">
          <circle cx="60" cy="60" r={r} stroke="rgba(255,255,255,0.08)" strokeWidth="10" fill="none" />
          <motion.circle
            cx="60"
            cy="60"
            r={r}
            stroke={color}
            strokeWidth="10"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={c}
            initial={{ strokeDashoffset: c }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.1, ease: "easeOut" }}
            style={{ filter: `drop-shadow(0 0 8px ${color})` }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className="text-3xl font-bold">{pct}%</div>
          <div className="text-[10px] uppercase tracking-wider text-muted">stroke prob.</div>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs font-medium text-muted">{label}</span>
      {children}
    </label>
  );
}

function Select({ value, onChange, children }: { value: string; onChange: (v: string) => void; children: React.ReactNode }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2.5 text-sm text-white outline-none transition focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
    >
      {children}
    </select>
  );
}

function Slider({
  min,
  max,
  step = 1,
  value,
  onChange,
}: {
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full accent-accent"
    />
  );
}

function Toggle({ value, onChange }: { value: 0 | 1; onChange: (v: 0 | 1) => void }) {
  return (
    <div className="flex gap-2">
      {[0, 1].map((v) => (
        <button
          key={v}
          type="button"
          onClick={() => onChange(v as 0 | 1)}
          className={cn(
            "flex-1 rounded-xl border px-3 py-2.5 text-sm transition",
            value === v
              ? "border-accent/60 bg-accent/20 text-white"
              : "border-white/10 bg-white/5 text-muted hover:text-white"
          )}
        >
          {v === 0 ? "No" : "Yes"}
        </button>
      ))}
    </div>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-3">
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className="mt-1 truncate text-sm font-medium">{value}</div>
    </div>
  );
}

function FeatureStrip() {
  const items = [
    { icon: ShieldCheck, title: "Clinically tuned", desc: "Threshold optimized for ≥70% recall." },
    { icon: Brain, title: "ML pipeline", desc: "Preprocessing + calibrated classifier." },
    { icon: Activity, title: "Real-time API", desc: "FastAPI backend, sub-second inference." },
  ];
  return (
    <div className="mt-10 grid gap-4 sm:grid-cols-3">
      {items.map((it) => (
        <div key={it.title} className="glass rounded-2xl p-5">
          <it.icon className="h-6 w-6 text-accent-2" />
          <div className="mt-3 text-sm font-semibold">{it.title}</div>
          <div className="mt-1 text-xs text-muted">{it.desc}</div>
        </div>
      ))}
    </div>
  );
}

function Insights({ meta }: { meta: ModelMetadata | null }) {
  if (!meta) {
    return (
      <div className="glass rounded-3xl p-10 text-center text-muted">
        <Loader2 className="mx-auto h-6 w-6 animate-spin" />
        <p className="mt-3 text-sm">Loading model metadata…</p>
      </div>
    );
  }
  const m = meta.metrics;
  const cards = [
    { label: "Accuracy", value: m.accuracy },
    { label: "Precision", value: m.precision },
    { label: "Recall", value: m.recall },
    { label: "F1", value: m.f1 },
    { label: "ROC-AUC", value: m.roc_auc },
    { label: "PR-AUC", value: m.avg_precision },
  ];
  return (
    <div className="space-y-6">
      <div className="glass rounded-3xl p-6 sm:p-8">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Model Insights</h2>
            <p className="mt-1 text-sm text-muted">
              Best model: <span className="text-accent-2">{meta.best_model}</span> · Threshold{" "}
              <span className="text-accent-2">{meta.threshold}</span>
            </p>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-xs text-muted">
            {meta.features.numeric.length} numeric · {meta.features.categorical.length} categorical
          </div>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {cards.map((c) => (
          <motion.div
            key={c.label}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-2xl p-5"
          >
            <div className="text-xs uppercase tracking-wider text-muted">{c.label}</div>
            <div className="mt-2 text-3xl font-bold">
              {c.label === "ROC-AUC" || c.label === "PR-AUC"
                ? c.value.toFixed(3)
                : (c.value * 100).toFixed(1) + "%"}
            </div>
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/5">
              <motion.div
                className="h-full rounded-full bg-gradient-to-r from-accent to-accent-2"
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(100, c.value * 100)}%` }}
                transition={{ duration: 0.9, ease: "easeOut" }}
              />
            </div>
          </motion.div>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="glass rounded-3xl p-6">
          <h3 className="text-base font-semibold">Model comparison</h3>
          <p className="mt-1 text-xs text-muted">Cross-validated average precision.</p>
          <div className="mt-4 overflow-hidden rounded-xl border border-white/10">
            <table className="w-full text-sm">
              <thead className="bg-white/5 text-xs text-muted">
                <tr>
                  <th className="px-3 py-2 text-left">Model</th>
                  <th className="px-3 py-2 text-right">CV AP</th>
                  <th className="px-3 py-2 text-right">ROC-AUC</th>
                  <th className="px-3 py-2 text-right">Recall</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(meta.all_models).map(([name, v]) => (
                  <tr
                    key={name}
                    className={cn(
                      "border-t border-white/5",
                      name === meta.best_model && "bg-accent/10"
                    )}
                  >
                    <td className="px-3 py-2 font-medium">
                      {name}
                      {name === meta.best_model && (
                        <span className="ml-2 rounded-full bg-accent/20 px-2 py-0.5 text-[10px] text-accent-2">
                          best
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-right">
                      {v.cv_avg_precision_mean?.toFixed(3) ?? "—"}
                    </td>
                    <td className="px-3 py-2 text-right">{v.roc_auc.toFixed(3)}</td>
                    <td className="px-3 py-2 text-right">{(v.recall * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="glass rounded-3xl p-6">
          <h3 className="text-base font-semibold">Confusion matrix</h3>
          <p className="mt-1 text-xs text-muted">At tuned threshold {meta.threshold}.</p>
          <ConfusionMatrix cm={m.confusion_matrix} />
        </div>
      </div>
    </div>
  );
}

function ConfusionMatrix({ cm }: { cm: number[][] }) {
  const labels = ["Pred 0", "Pred 1"];
  const rows = ["Actual 0", "Actual 1"];
  const max = Math.max(...cm.flat());
  return (
    <div className="mt-4 grid grid-cols-3 gap-2 text-center text-sm">
      <div />
      {labels.map((l) => (
        <div key={l} className="text-xs text-muted">
          {l}
        </div>
      ))}
      {cm.map((row, i) => (
        <div key={`row-${i}`} className="contents">
          <div className="flex items-center text-xs text-muted">{rows[i]}</div>
          {row.map((val, j) => (
            <div
              key={`${i}-${j}`}
              className="rounded-xl border border-white/10 p-4 font-semibold"
              style={{
                background: `rgba(124,92,255,${0.08 + (val / max) * 0.35})`,
              }}
            >
              {val}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function About() {
  return (
    <div className="glass rounded-3xl p-8 sm:p-12">
      <h2 className="text-2xl font-bold tracking-tight">About this project</h2>
      <p className="mt-4 max-w-3xl text-sm text-muted">
        An end-to-end MLOps project in the healthcare domain. Predicts whether a patient is
        likely to suffer a stroke from demographic, lifestyle and clinical features. Includes
        data validation, training, experiment comparison, model registry, serving via FastAPI,
        a Streamlit demo, drift monitoring, tests and CI/CD.
      </p>
      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        {[
          ["Dataset", "Stroke Prediction Dataset (Kaggle) · 5,110 records"],
          ["Target", "stroke (binary, ~4.9% positive)"],
          ["Features", "10 mixed numeric + categorical"],
          ["Serving", "FastAPI + Next.js frontend"],
        ].map(([k, v]) => (
          <div key={k} className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <div className="text-xs uppercase tracking-wider text-muted">{k}</div>
            <div className="mt-1 text-sm">{v}</div>
          </div>
        ))}
      </div>
      <div className="mt-8 text-xs text-muted">
        Disclaimer: This is a demo for educational purposes and not a medical device. Do not
        use for clinical decisions.
      </div>
    </div>
  );
}

function Footer() {
  return (
    <footer className="border-t border-white/5 py-6 text-center text-xs text-muted">
      StrokeRisk AI · MLOps demo · Built with Next.js
    </footer>
  );
}