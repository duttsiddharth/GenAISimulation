import React, { useState, useMemo, useEffect, useRef } from "react";
import {
  Boxes, Database, Cpu, Workflow, GitBranch, Beaker, ClipboardList,
  Play, Copy, Check, ChevronRight, Layers, Search, Sparkles, Terminal,
  Activity, Gauge, ArrowRight, RotateCcw, Zap, FileCode, ShieldCheck,
  Braces, Server, CircleDot, Target, BookOpen, ListChecks, Send,
} from "lucide-react";

/* ────────────────────────────────────────────────────────────────────────
   GenAI Developer — Contract Flight Deck
   A working simulator to practice, learn and deliver on the 6-month
   Toronto GenAI Developer contract. Retrieval, chunking and scoring are
   real (in-browser TF-IDF + BM25 hybrid). Generation is optional/live.
   ──────────────────────────────────────────────────────────────────────── */

/* ===== Retrieval engine (real, runs in the browser) ===================== */
const STOP = new Set(
  "a an the and or but if then of to in on for with as by at from into is are was were be been being this that these those it its you your we our they their do does did have has had will would can could should may might must not no more most such than so".split(" ")
);
const tokenize = (t) =>
  (t.toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => w.length > 1 && !STOP.has(w));

function splitSentences(t) {
  return (t.replace(/\s+/g, " ").match(/[^.!?]+[.!?]*/g) || [t]).map((s) => s.trim()).filter(Boolean);
}

function chunkText(text, strategy, size, overlap) {
  const clean = text.replace(/\r/g, "").trim();
  if (!clean) return [];
  if (strategy === "fixed") {
    const step = Math.max(1, size - overlap);
    const out = [];
    for (let i = 0; i < clean.length; i += step) out.push(clean.slice(i, i + size).trim());
    return out.filter(Boolean);
  }
  if (strategy === "sentence") {
    const sents = splitSentences(clean);
    const per = Math.max(1, Math.round(size / 90)); // approx sentences per chunk
    const ov = Math.min(per - 1, Math.max(0, Math.round(overlap / 90)));
    const out = [];
    for (let i = 0; i < sents.length; i += Math.max(1, per - ov)) {
      out.push(sents.slice(i, i + per).join(" "));
    }
    return out.filter(Boolean);
  }
  // recursive: paragraph-aware, pack sentences up to size
  const paras = clean.split(/\n{2,}/).flatMap((p) => splitSentences(p));
  const out = [];
  let cur = "";
  for (const s of paras) {
    if ((cur + " " + s).length > size && cur) {
      out.push(cur.trim());
      cur = overlap > 0 ? cur.slice(-overlap) + " " + s : s;
    } else {
      cur = cur ? cur + " " + s : s;
    }
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

function buildIndex(chunks) {
  const docs = chunks.map(tokenize);
  const N = docs.length || 1;
  const df = new Map();
  docs.forEach((d) => new Set(d).forEach((t) => df.set(t, (df.get(t) || 0) + 1)));
  const avgdl = docs.reduce((a, d) => a + d.length, 0) / N;
  const tfidf = docs.map((d) => {
    const tf = new Map();
    d.forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
    const vec = new Map();
    let norm = 0;
    tf.forEach((f, t) => {
      const w = (1 + Math.log(f)) * Math.log(N / (df.get(t) || 1));
      vec.set(t, w);
      norm += w * w;
    });
    norm = Math.sqrt(norm) || 1;
    vec.forEach((v, t) => vec.set(t, v / norm));
    return vec;
  });
  return { docs, df, N, avgdl, tfidf };
}

function search(query, chunks, index, alpha) {
  if (!chunks.length) return [];
  const q = tokenize(query);
  // vector (cosine on tf-idf)
  const qtf = new Map();
  q.forEach((t) => qtf.set(t, (qtf.get(t) || 0) + 1));
  const qvec = new Map();
  let qn = 0;
  qtf.forEach((f, t) => {
    const w = (1 + Math.log(f)) * Math.log(index.N / (index.df.get(t) || index.N));
    qvec.set(t, w);
    qn += w * w;
  });
  qn = Math.sqrt(qn) || 1;
  qvec.forEach((v, t) => qvec.set(t, v / qn));
  const cos = index.tfidf.map((vec) => {
    let s = 0;
    qvec.forEach((qv, t) => { if (vec.has(t)) s += qv * vec.get(t); });
    return s;
  });
  // keyword (BM25)
  const k1 = 1.5, b = 0.75;
  const bm25 = index.docs.map((d) => {
    const tf = new Map();
    d.forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
    let s = 0;
    q.forEach((t) => {
      const f = tf.get(t) || 0;
      if (!f) return;
      const idf = Math.log((index.N - (index.df.get(t) || 0) + 0.5) / ((index.df.get(t) || 0) + 0.5) + 1);
      s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * (d.length / (index.avgdl || 1))));
    });
    return s;
  });
  const norm = (arr) => { const m = Math.max(...arr, 1e-9); return arr.map((v) => v / m); };
  const nc = norm(cos), nb = norm(bm25);
  return chunks
    .map((c, i) => ({
      i, text: c, vec: nc[i], kw: nb[i],
      score: alpha * nc[i] + (1 - alpha) * nb[i],
    }))
    .sort((a, b2) => b2.score - a.score);
}

/* ===== Seed knowledge base (on-topic for the role) ====================== */
const SEED_KB = `Retrieval-Augmented Generation (RAG) grounds a language model's answers in an external knowledge base. Instead of relying only on parametric memory, the model retrieves relevant passages at query time and conditions its generation on them, which reduces hallucination and lets you cite sources.

Document ingestion is the first stage of any RAG pipeline. Raw sources such as PDFs, HTML pages, Confluence exports and support tickets are extracted, cleaned of markup and boilerplate, and normalised into plain text before anything else happens.

Chunking splits long documents into passages that fit the embedding model and the context window. Fixed-size chunking is fast but can cut sentences awkwardly. Recursive chunking respects paragraph and sentence boundaries. Overlap between chunks preserves context that would otherwise be lost at the boundary. Typical chunk sizes range from 256 to 1024 tokens with 10 to 20 percent overlap.

Embeddings turn each chunk into a dense vector that captures meaning. Similar passages land close together in vector space, so a query embedding can find semantically related chunks even when the exact keywords differ.

A vector database stores embeddings and serves approximate nearest-neighbour search at low latency. Popular choices include pgvector on Postgres, Pinecone, Weaviate, Qdrant, Milvus and Chroma. Indexes such as HNSW and IVF trade recall for speed.

Hybrid retrieval combines dense vector similarity with sparse keyword scoring such as BM25. Dense search captures meaning while sparse search captures exact terms, product codes and acronyms. Scores are normalised and fused, often with reciprocal rank fusion or a weighted sum, to get the best of both.

Reranking improves precision by passing the top candidates through a cross-encoder that scores query and passage together. It is slower than bi-encoder retrieval, so it is applied only to a small shortlist.

Prompt assembly stuffs the retrieved context into the model prompt along with the user question and system instructions. Keeping context ordered by relevance and staying within the token budget matters for answer quality.

Fine-tuning adapts a base model to a domain or task. Full fine-tuning updates all weights and is expensive. Parameter-efficient methods such as LoRA and QLoRA update small adapter matrices, which is cheaper and often enough for style, format and domain adaptation.

AI agents plan and act in a loop: they reason about a goal, choose a tool, observe the result and iterate until the task is done. Tool use, function calling and orchestration frameworks such as LangGraph coordinate these steps.

MLOps brings software discipline to models. CI/CD pipelines test and deploy code and models, containers package the runtime, and monitoring tracks latency, cost, drift and answer quality so regressions are caught in production.

ETL for GenAI extracts data from sources, transforms it by cleaning, de-duplicating, chunking and embedding, and loads the vectors into the store. Idempotent, incremental and observable pipelines keep the knowledge base fresh without full rebuilds.`;

/* ===== Module registry ================================================== */
const MODULES = [
  { id: "deck", label: "Flight Deck", icon: Gauge, tag: "overview" },
  { id: "rag", label: "RAG Lab", icon: Search, tag: "must-have" },
  { id: "etl", label: "ETL Studio", icon: Database, tag: "must-have" },
  { id: "api", label: "API Forge", icon: Braces, tag: "must-have" },
  { id: "agent", label: "Agent Loop", icon: Workflow, tag: "must-have" },
  { id: "mlops", label: "MLOps Pipeline", icon: GitBranch, tag: "nice-to-have" },
  { id: "arena", label: "Practice Arena", icon: Beaker, tag: "practice" },
  { id: "deliver", label: "Delivery Kit", icon: ClipboardList, tag: "deliver" },
];

/* ===== Small UI atoms =================================================== */
const Panel = ({ children, className = "" }) => (
  <div className={"rounded-xl border border-slate-700/60 bg-slate-900/50 " + className}>{children}</div>
);
const Chip = ({ children, tone = "cyan" }) => {
  const tones = {
    cyan: "border-cyan-500/40 text-cyan-300 bg-cyan-500/10",
    amber: "border-amber-500/40 text-amber-300 bg-amber-500/10",
    emerald: "border-emerald-500/40 text-emerald-300 bg-emerald-500/10",
    rose: "border-rose-500/40 text-rose-300 bg-rose-500/10",
    slate: "border-slate-600/50 text-slate-300 bg-slate-700/20",
  };
  return <span className={"inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[11px] font-medium " + tones[tone]}>{children}</span>;
};
function CopyBtn({ text }) {
  const [ok, setOk] = useState(false);
  return (
    <button
      onClick={() => { navigator.clipboard?.writeText(text); setOk(true); setTimeout(() => setOk(false), 1200); }}
      className="inline-flex items-center gap-1.5 rounded-md border border-slate-600 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-700/50 transition"
    >
      {ok ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} />}
      {ok ? "Copied" : "Copy"}
    </button>
  );
}
const Bar = ({ pct, tone = "cyan" }) => {
  const c = { cyan: "#22d3ee", amber: "#fbbf24", emerald: "#34d399", rose: "#fb7185" }[tone];
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-700/60">
      <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.min(100, pct)}%`, background: c }} />
    </div>
  );
};

/* ===== Readiness gauge (signature instrument) =========================== */
function Reactor({ score }) {
  const r = 52, C = 2 * Math.PI * r;
  const off = C * (1 - score / 100);
  const hue = score < 40 ? "#fb7185" : score < 70 ? "#fbbf24" : "#34d399";
  return (
    <div className="relative h-32 w-32">
      <svg viewBox="0 0 128 128" className="h-full w-full -rotate-90">
        <circle cx="64" cy="64" r={r} fill="none" stroke="#1e293b" strokeWidth="10" />
        <circle cx="64" cy="64" r={r} fill="none" stroke={hue} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={C} strokeDashoffset={off} style={{ transition: "stroke-dashoffset .6s ease, stroke .4s" }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-3xl font-bold text-slate-100">{Math.round(score)}</span>
        <span className="text-[10px] uppercase tracking-widest text-slate-500">ready</span>
      </div>
    </div>
  );
}

/* ===== RAG LAB ========================================================== */
function RagLab({ mark }) {
  const [doc, setDoc] = useState(SEED_KB);
  const [strategy, setStrategy] = useState("recursive");
  const [size, setSize] = useState(400);
  const [overlap, setOverlap] = useState(60);
  const [alpha, setAlpha] = useState(0.6);
  const [topK, setTopK] = useState(3);
  const [query, setQuery] = useState("How does hybrid retrieval mix keyword and vector search?");
  const [gen, setGen] = useState({ loading: false, text: "", err: "" });

  const chunks = useMemo(() => chunkText(doc, strategy, size, overlap), [doc, strategy, size, overlap]);
  const index = useMemo(() => buildIndex(chunks), [chunks]);
  const results = useMemo(() => search(query, chunks, index, alpha), [query, chunks, index, alpha]);
  const top = results.slice(0, topK);

  const prompt = useMemo(() => {
    const ctx = top.map((r, i) => `[${i + 1}] ${r.text}`).join("\n\n");
    return `System: You are a precise assistant. Answer ONLY from the context. Cite sources as [n]. If the answer is not present, say so.\n\nContext:\n${ctx}\n\nQuestion: ${query}\nAnswer:`;
  }, [top, query]);

  useEffect(() => { if (chunks.length && query) mark("rag", true); }, [chunks.length, query, mark]);

  async function generate() {
    setGen({ loading: true, text: "", err: "" });
    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-6",
          max_tokens: 1000,
          messages: [{ role: "user", content: prompt }],
        }),
      });
      const data = await res.json();
      const text = (data.content || []).filter((b) => b.type === "text").map((b) => b.text).join("\n").trim();
      setGen({ loading: false, text: text || "(no text returned)", err: "" });
    } catch (e) {
      setGen({ loading: false, text: "", err: "Live generation is unavailable here. The retrieval pipeline above is fully working — copy the assembled prompt into your own stack to generate." });
    }
  }

  return (
    <div className="space-y-5">
      <Header icon={Search} title="RAG Lab" sub="Chunk → embed → hybrid retrieve → assemble prompt. Retrieval runs live in your browser (TF-IDF + BM25 fusion)." />

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel className="p-4">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm font-semibold text-slate-200">1 · Knowledge source</span>
            <Chip tone="slate">{doc.length.toLocaleString()} chars</Chip>
          </div>
          <textarea value={doc} onChange={(e) => setDoc(e.target.value)}
            className="h-40 w-full resize-none rounded-lg border border-slate-700 bg-slate-950/60 p-3 font-mono text-xs text-slate-300 focus:border-cyan-500 focus:outline-none" />
        </Panel>

        <Panel className="p-4">
          <div className="mb-3 text-sm font-semibold text-slate-200">2 · Chunking strategy</div>
          <div className="mb-3 flex gap-2">
            {["fixed", "sentence", "recursive"].map((s) => (
              <button key={s} onClick={() => setStrategy(s)}
                className={"flex-1 rounded-lg border px-2 py-1.5 text-xs capitalize transition " +
                  (strategy === s ? "border-cyan-500 bg-cyan-500/15 text-cyan-200" : "border-slate-700 text-slate-400 hover:border-slate-600")}>
                {s}
              </button>
            ))}
          </div>
          <Slider label="Chunk size (chars)" value={size} min={120} max={900} step={20} onChange={setSize} />
          <Slider label="Overlap (chars)" value={overlap} min={0} max={200} step={10} onChange={setOverlap} />
          <div className="mt-2 flex items-center gap-2 text-xs text-slate-400">
            <Layers size={13} className="text-cyan-400" /> Produced <b className="text-slate-200">{chunks.length}</b> chunks · avg{" "}
            <b className="text-slate-200">{chunks.length ? Math.round(chunks.reduce((a, c) => a + c.length, 0) / chunks.length) : 0}</b> chars
          </div>
        </Panel>
      </div>

      <Panel className="p-4">
        <div className="mb-3 text-sm font-semibold text-slate-200">3 · Query &amp; hybrid retrieval</div>
        <div className="flex flex-col gap-3 sm:flex-row">
          <input value={query} onChange={(e) => setQuery(e.target.value)}
            className="flex-1 rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm text-slate-200 focus:border-cyan-500 focus:outline-none" />
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">top-k</span>
            {[1, 3, 5].map((k) => (
              <button key={k} onClick={() => setTopK(k)}
                className={"rounded-md border px-2.5 py-1 text-xs " + (topK === k ? "border-cyan-500 bg-cyan-500/15 text-cyan-200" : "border-slate-700 text-slate-400")}>{k}</button>
            ))}
          </div>
        </div>
        <div className="mt-3">
          <div className="mb-1 flex justify-between text-xs text-slate-400">
            <span>Keyword (BM25)</span><span className="text-cyan-300">α = {alpha.toFixed(2)} → weighting</span><span>Vector (cosine)</span>
          </div>
          <input type="range" min={0} max={1} step={0.05} value={alpha} onChange={(e) => setAlpha(+e.target.value)} className="w-full accent-cyan-400" />
        </div>

        <div className="mt-4 space-y-2">
          {top.map((r, i) => (
            <div key={r.i} className="rounded-lg border border-slate-700/70 bg-slate-950/40 p-3">
              <div className="mb-1.5 flex items-center justify-between text-[11px]">
                <span className="font-mono text-cyan-300">chunk #{r.i} · rank {i + 1}</span>
                <span className="flex gap-2 font-mono text-slate-400">
                  <span>vec {r.vec.toFixed(2)}</span><span>kw {r.kw.toFixed(2)}</span>
                  <span className="text-emerald-300">score {r.score.toFixed(2)}</span>
                </span>
              </div>
              <div className="mb-1"><Bar pct={r.score * 100} tone="emerald" /></div>
              <p className="text-xs leading-relaxed text-slate-300">{r.text}</p>
            </div>
          ))}
          {!top.length && <div className="text-xs text-slate-500">No chunks yet — add a source above.</div>}
        </div>
      </Panel>

      <Panel className="p-4">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-sm font-semibold text-slate-200">4 · Assembled prompt</span>
          <div className="flex gap-2">
            <CopyBtn text={prompt} />
            <button onClick={generate} disabled={gen.loading}
              className="inline-flex items-center gap-1.5 rounded-md border border-cyan-500/50 bg-cyan-500/15 px-3 py-1 text-xs font-medium text-cyan-200 hover:bg-cyan-500/25 disabled:opacity-50">
              {gen.loading ? <Activity size={13} className="animate-spin" /> : <Sparkles size={13} />}
              {gen.loading ? "Generating…" : "Generate (live)"}
            </button>
          </div>
        </div>
        <pre className="max-h-56 overflow-auto rounded-lg border border-slate-700 bg-slate-950/70 p-3 font-mono text-[11px] leading-relaxed text-slate-300 whitespace-pre-wrap">{prompt}</pre>
        {gen.text && (
          <div className="mt-3 rounded-lg border border-emerald-600/40 bg-emerald-500/5 p-3">
            <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-emerald-300"><Sparkles size={13} /> Grounded answer</div>
            <p className="whitespace-pre-wrap text-sm text-slate-200">{gen.text}</p>
          </div>
        )}
        {gen.err && <div className="mt-3 rounded-lg border border-amber-600/40 bg-amber-500/5 p-3 text-xs text-amber-200">{gen.err}</div>}
      </Panel>
    </div>
  );
}
function Slider({ label, value, min, max, step, onChange }) {
  return (
    <div className="mb-2">
      <div className="mb-1 flex justify-between text-xs text-slate-400"><span>{label}</span><span className="font-mono text-slate-300">{value}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} className="w-full accent-cyan-400" />
    </div>
  );
}

/* ===== ETL STUDIO ======================================================= */
const RAW_ROWS = [
  { id: 1, src: "confluence", body: "<p>Reset password via <b>Settings &gt; Security</b>.</p>", meta: "  " },
  { id: 2, src: "zendesk", body: "Reset password via Settings > Security.", meta: "dup" },
  { id: 3, src: "pdf", body: "VPN setup: install client, contact ops@corp.com   ", meta: "pii" },
  { id: 4, src: "html", body: "   ", meta: "empty" },
  { id: 5, src: "confluence", body: "Expense claims are filed in the Finance portal before month end.", meta: "" },
];
function EtlStudio({ mark }) {
  const stages = [
    { id: "clean", label: "Strip markup & whitespace", icon: Sparkles },
    { id: "dedup", label: "De-duplicate near matches", icon: Layers },
    { id: "pii", label: "Redact PII (emails)", icon: ShieldCheck },
    { id: "drop", label: "Drop empty records", icon: RotateCcw },
    { id: "chunk", label: "Chunk & embed (mock)", icon: Boxes },
  ];
  const [on, setOn] = useState({ clean: true, dedup: true, pii: true, drop: true, chunk: false });
  useEffect(() => { mark("etl", Object.values(on).filter(Boolean).length >= 3); }, [on, mark]);

  const out = useMemo(() => {
    let rows = RAW_ROWS.map((r) => ({ ...r }));
    if (on.clean) rows = rows.map((r) => ({ ...r, body: r.body.replace(/<[^>]+>/g, "").replace(/&gt;/g, ">").replace(/\s+/g, " ").trim() }));
    if (on.drop) rows = rows.filter((r) => r.body && r.body.trim().length > 0);
    if (on.dedup) { const seen = new Set(); rows = rows.filter((r) => { const k = r.body.toLowerCase().replace(/[^a-z0-9]/g, ""); if (seen.has(k)) return false; seen.add(k); return true; }); }
    if (on.pii) rows = rows.map((r) => ({ ...r, body: r.body.replace(/[\w.]+@[\w.]+/g, "[email]") }));
    let embedded = rows;
    if (on.chunk) embedded = rows.map((r) => ({ ...r, body: r.body, vec: "[" + Array.from({ length: 4 }, (_, i) => ((Math.sin(r.id * 7 + i) + 1) / 2).toFixed(2)).join(", ") + " …]" }));
    return embedded;
  }, [on]);

  const code = `# etl_pipeline.py — GenAI knowledge-base loader (idempotent, incremental)
from pipeline import extract, clean, dedupe, redact_pii, chunk, embed, upsert

def run(source):
    rows = extract(source)                 # PDFs / HTML / Confluence / tickets
    rows = clean(rows)                     # strip markup + normalise whitespace
    rows = [r for r in rows if r.body]     # drop empties
    rows = dedupe(rows, threshold=0.92)    # near-duplicate removal
    rows = redact_pii(rows)                # emails, phones, IDs
    docs = chunk(rows, size=400, overlap=60, strategy="recursive")
    vecs = embed(docs, model="text-embedding-3-small")
    upsert("pgvector", vecs, on_conflict="doc_hash")   # incremental, no full rebuild
    return {"ingested": len(vecs)}`;

  return (
    <div className="space-y-5">
      <Header icon={Database} title="ETL Studio" sub="Toggle transform stages and watch messy multi-source data become a clean, embeddable knowledge base." />
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel className="p-4">
          <div className="mb-3 text-sm font-semibold text-slate-200">Pipeline stages</div>
          <div className="space-y-2">
            {stages.map((s) => (
              <button key={s.id} onClick={() => setOn((o) => ({ ...o, [s.id]: !o[s.id] }))}
                className={"flex w-full items-center gap-3 rounded-lg border px-3 py-2.5 text-left text-sm transition " +
                  (on[s.id] ? "border-cyan-500/60 bg-cyan-500/10 text-slate-100" : "border-slate-700 text-slate-400 hover:border-slate-600")}>
                <s.icon size={16} className={on[s.id] ? "text-cyan-300" : "text-slate-500"} />
                <span className="flex-1">{s.label}</span>
                <span className={"h-4 w-8 rounded-full p-0.5 transition " + (on[s.id] ? "bg-cyan-500/70" : "bg-slate-600")}>
                  <span className="block h-3 w-3 rounded-full bg-white transition" style={{ transform: on[s.id] ? "translateX(16px)" : "none" }} />
                </span>
              </button>
            ))}
          </div>
          <div className="mt-3 flex items-center justify-between rounded-lg border border-slate-700 bg-slate-950/50 px-3 py-2 text-xs">
            <span className="text-slate-400">Records</span>
            <span className="font-mono text-slate-300">{RAW_ROWS.length} in <ArrowRight size={11} className="inline" /> <b className="text-emerald-300">{out.length}</b> out</span>
          </div>
        </Panel>

        <Panel className="p-4">
          <div className="mb-3 text-sm font-semibold text-slate-200">Output preview</div>
          <div className="space-y-2">
            {out.map((r) => (
              <div key={r.id} className="rounded-lg border border-slate-700/70 bg-slate-950/40 p-2.5">
                <div className="mb-1 flex items-center gap-2 text-[10px]"><Chip tone="slate">{r.src}</Chip>{r.vec && <Chip tone="cyan">embedded</Chip>}</div>
                <p className="font-mono text-[11px] text-slate-300">{r.body}</p>
                {r.vec && <p className="mt-1 font-mono text-[10px] text-cyan-400/70">{r.vec}</p>}
              </div>
            ))}
            {!out.length && <div className="text-xs text-slate-500">All records filtered out — turn off "Drop empty" to inspect.</div>}
          </div>
        </Panel>
      </div>
      <Panel className="p-4">
        <div className="mb-2 flex items-center justify-between"><span className="text-sm font-semibold text-slate-200">Reference implementation</span><CopyBtn text={code} /></div>
        <pre className="overflow-auto rounded-lg border border-slate-700 bg-slate-950/70 p-3 font-mono text-[11px] leading-relaxed text-slate-300">{code}</pre>
      </Panel>
    </div>
  );
}

/* ===== API FORGE ======================================================== */
function ApiForge({ mark }) {
  const [path, setPath] = useState("/v1/chat");
  const [model, setModel] = useState("gpt-4o-mini");
  const [stream, setStream] = useState(true);
  const [rag, setRag] = useState(true);
  const [auth, setAuth] = useState(true);
  useEffect(() => { mark("api", true); }, [mark]);

  const code = `# app.py — FastAPI service exposing a GenAI endpoint
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel${stream ? "\nfrom fastapi.responses import StreamingResponse" : ""}
from rag import retrieve  ${rag ? "# hybrid retriever" : ""}
from llm import client

app = FastAPI(title="GenAI Service")

class ChatRequest(BaseModel):
    query: str
    top_k: int = 4
    temperature: float = 0.2

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
${auth ? `
def verify(x_api_key: str = Header(...)):
    if not is_valid(x_api_key):
        raise HTTPException(401, "invalid api key")
` : ""}
@app.post("${path}"${stream ? "" : ", response_model=ChatResponse"})
def chat(req: ChatRequest${auth ? ", _=Depends(verify)" : ""}):
    ctx = ${rag ? "retrieve(req.query, k=req.top_k)" : "[]"}
    prompt = build_prompt(req.query, ctx)
    ${stream
      ? `def gen():
        for tok in client.stream("${model}", prompt, temperature=req.temperature):
            yield tok
    return StreamingResponse(gen(), media_type="text/event-stream")`
      : `answer = client.complete("${model}", prompt, temperature=req.temperature)
    return ChatResponse(answer=answer, sources=[c.id for c in ctx])`}

@app.get("/healthz")
def health():
    return {"status": "ok"}`;

  const toggles = [
    { k: stream, set: setStream, label: "Streaming (SSE)", icon: Zap },
    { k: rag, set: setRag, label: "RAG context injection", icon: Search },
    { k: auth, set: setAuth, label: "API-key auth", icon: ShieldCheck },
  ];
  return (
    <div className="space-y-5">
      <Header icon={Braces} title="API Forge" sub="Configure a production-shaped FastAPI endpoint that serves a GenAI model. Copy the generated scaffold straight into your repo." />
      <div className="grid gap-5 lg:grid-cols-3">
        <Panel className="p-4 lg:col-span-1">
          <div className="mb-3 text-sm font-semibold text-slate-200">Endpoint config</div>
          <label className="mb-1 block text-xs text-slate-400">Route path</label>
          <input value={path} onChange={(e) => setPath(e.target.value)} className="mb-3 w-full rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-2 font-mono text-xs text-slate-200 focus:border-cyan-500 focus:outline-none" />
          <label className="mb-1 block text-xs text-slate-400">Model</label>
          <select value={model} onChange={(e) => setModel(e.target.value)} className="mb-3 w-full rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-2 text-xs text-slate-200 focus:border-cyan-500 focus:outline-none">
            {["gpt-4o-mini", "claude-sonnet-4-6", "llama-3.3-70b", "gemini-1.5-pro"].map((m) => <option key={m}>{m}</option>)}
          </select>
          <div className="space-y-2">
            {toggles.map((t) => (
              <button key={t.label} onClick={() => t.set(!t.k)} className={"flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left text-xs transition " + (t.k ? "border-cyan-500/60 bg-cyan-500/10 text-slate-100" : "border-slate-700 text-slate-400")}>
                <t.icon size={14} className={t.k ? "text-cyan-300" : "text-slate-500"} /><span className="flex-1">{t.label}</span>
                <span className={"h-4 w-8 rounded-full p-0.5 " + (t.k ? "bg-cyan-500/70" : "bg-slate-600")}><span className="block h-3 w-3 rounded-full bg-white" style={{ transform: t.k ? "translateX(16px)" : "none" }} /></span>
              </button>
            ))}
          </div>
        </Panel>
        <Panel className="p-4 lg:col-span-2">
          <div className="mb-2 flex items-center justify-between"><span className="text-sm font-semibold text-slate-200"><FileCode size={14} className="mr-1 inline text-cyan-400" />app.py</span><CopyBtn text={code} /></div>
          <pre className="max-h-96 overflow-auto rounded-lg border border-slate-700 bg-slate-950/70 p-3 font-mono text-[11px] leading-relaxed text-slate-300">{code}</pre>
        </Panel>
      </div>
    </div>
  );
}

/* ===== AGENT LOOP ======================================================= */
const AGENT_TRACE = [
  { phase: "Think", text: "Goal: answer 'what changed in the Q3 refund policy?'. I need the current policy doc.", tone: "cyan" },
  { phase: "Act", text: "search_kb(query='Q3 refund policy change')", tone: "amber" },
  { phase: "Observe", text: "Retrieved 3 chunks. Top chunk mentions a new 14-day window, effective July.", tone: "emerald" },
  { phase: "Think", text: "I have the window but not the prior value. Need the previous policy for the diff.", tone: "cyan" },
  { phase: "Act", text: "search_kb(query='refund policy before July', top_k=2)", tone: "amber" },
  { phase: "Observe", text: "Prior policy used a 30-day window. Enough to compose the diff.", tone: "emerald" },
  { phase: "Think", text: "Both values in hand. Compose grounded answer with citations and stop.", tone: "cyan" },
  { phase: "Answer", text: "Refund window shrank from 30 to 14 days, effective July [1][2].", tone: "rose" },
];
function AgentLoop({ mark }) {
  const [step, setStep] = useState(0);
  useEffect(() => { if (step >= AGENT_TRACE.length - 1) mark("agent", true); }, [step, mark]);
  const toneColor = { cyan: "#22d3ee", amber: "#fbbf24", emerald: "#34d399", rose: "#fb7185" };
  return (
    <div className="space-y-5">
      <Header icon={Workflow} title="Agent Loop" sub="Step through a Reason → Act → Observe cycle. This is the control flow behind tool-using agents and workflow orchestration." />
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel className="p-5">
          <svg viewBox="0 0 320 240" className="w-full">
            {[
              { x: 160, y: 40, l: "Think", t: "cyan" },
              { x: 260, y: 130, l: "Act (tool)", t: "amber" },
              { x: 160, y: 210, l: "Observe", t: "emerald" },
              { x: 60, y: 130, l: "Decide", t: "rose" },
            ].map((n, i) => (
              <g key={i}>
                <circle cx={n.x} cy={n.y} r="34" fill="#0f172a" stroke={toneColor[n.t]} strokeWidth="2" />
                <text x={n.x} y={n.y + 4} textAnchor="middle" fill="#e2e8f0" fontSize="11" fontFamily="monospace">{n.l}</text>
              </g>
            ))}
            {[[160, 74, 232, 104], [232, 156, 190, 184], [130, 184, 88, 156], [88, 104, 130, 74]].map((a, i) => (
              <line key={i} x1={a[0]} y1={a[1]} x2={a[2]} y2={a[3]} stroke="#475569" strokeWidth="1.5" markerEnd="url(#ar)" />
            ))}
            <defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#475569" /></marker></defs>
          </svg>
          <p className="mt-2 text-center text-xs text-slate-500">The loop repeats until the agent decides the goal is met.</p>
        </Panel>
        <Panel className="p-4">
          <div className="mb-3 flex items-center justify-between">
            <span className="text-sm font-semibold text-slate-200">Execution trace</span>
            <div className="flex gap-2">
              <button onClick={() => setStep((s) => Math.max(0, s - 1))} className="rounded-md border border-slate-600 px-2 py-1 text-xs text-slate-300 hover:bg-slate-700/50">Back</button>
              <button onClick={() => setStep((s) => Math.min(AGENT_TRACE.length - 1, s + 1))} className="inline-flex items-center gap-1 rounded-md border border-cyan-500/50 bg-cyan-500/15 px-2.5 py-1 text-xs text-cyan-200"><Play size={12} />Step</button>
              <button onClick={() => setStep(0)} className="rounded-md border border-slate-600 px-2 py-1 text-xs text-slate-400"><RotateCcw size={12} /></button>
            </div>
          </div>
          <div className="space-y-2">
            {AGENT_TRACE.slice(0, step + 1).map((t, i) => (
              <div key={i} className="rounded-lg border border-slate-700/70 bg-slate-950/40 p-2.5">
                <div className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold" style={{ color: toneColor[t.tone] }}><CircleDot size={11} />{t.phase}</div>
                <p className="font-mono text-[11px] text-slate-300">{t.text}</p>
              </div>
            ))}
          </div>
          <div className="mt-3"><Bar pct={((step + 1) / AGENT_TRACE.length) * 100} tone="amber" /></div>
        </Panel>
      </div>
    </div>
  );
}

/* ===== MLOPS ============================================================ */
const MLOPS = {
  "CI / build": ["Unit + eval tests on every PR (pytest, promptfoo)", "Lint, type-check, secret scan", "Build & tag container image", "Pin model + dataset versions (DVC / MLflow)"],
  "CD / deploy": ["Blue-green or canary rollout on Kubernetes", "Automated rollback on failed eval gate", "Infra as code (Terraform / Helm)", "Feature-flag new model versions"],
  "Monitoring": ["Latency, cost & token usage dashboards", "Groundedness / hallucination scoring", "Embedding & data drift detection", "Structured request/response logging + tracing"],
  "Lifecycle": ["Model registry with promotion stages", "Scheduled re-embedding of the KB", "Feedback capture → eval set growth", "Runbook + on-call for model incidents"],
};
function MlopsBoard({ mark }) {
  const all = Object.values(MLOPS).flat();
  const [done, setDone] = useState({});
  const count = Object.values(done).filter(Boolean).length;
  useEffect(() => { mark("mlops", count >= 6); }, [count, mark]);
  return (
    <div className="space-y-5">
      <Header icon={GitBranch} title="MLOps Pipeline" sub="The nice-to-haves that turn a prototype into something operable. Tick what you can already stand up; the gaps are your study list." />
      <Panel className="p-4">
        <div className="mb-3 flex items-center justify-between"><span className="text-sm text-slate-300">Operability</span><span className="font-mono text-sm text-emerald-300">{count}/{all.length}</span></div>
        <Bar pct={(count / all.length) * 100} tone="emerald" />
      </Panel>
      <div className="grid gap-4 md:grid-cols-2">
        {Object.entries(MLOPS).map(([group, items]) => (
          <Panel key={group} className="p-4">
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-200"><Server size={14} className="text-cyan-400" />{group}</div>
            <div className="space-y-1.5">
              {items.map((it) => (
                <button key={it} onClick={() => setDone((d) => ({ ...d, [it]: !d[it] }))} className="flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left text-xs hover:bg-slate-800/40">
                  <span className={"mt-0.5 flex h-4 w-4 flex-none items-center justify-center rounded border " + (done[it] ? "border-emerald-500 bg-emerald-500/20" : "border-slate-600")}>{done[it] && <Check size={11} className="text-emerald-300" />}</span>
                  <span className={done[it] ? "text-slate-400 line-through" : "text-slate-300"}>{it}</span>
                </button>
              ))}
            </div>
          </Panel>
        ))}
      </div>
    </div>
  );
}

/* ===== PRACTICE ARENA =================================================== */
const QUIZ = [
  { q: "Overlap between chunks primarily helps because it…", a: ["Reduces vector DB storage", "Preserves context lost at chunk boundaries", "Speeds up embedding", "Removes duplicate text"], c: 1, why: "Overlap keeps sentences that straddle a boundary intact in at least one chunk, so retrieval doesn't lose that context." },
  { q: "Hybrid retrieval fuses dense vectors with BM25 because…", a: ["Vectors are always wrong", "Sparse search captures exact terms/acronyms vectors may miss", "BM25 is faster to train", "It avoids needing a vector DB"], c: 1, why: "Dense search captures meaning; sparse BM25 nails exact codes, acronyms and rare terms. Fusing them beats either alone." },
  { q: "LoRA / QLoRA are attractive because they…", a: ["Update all model weights", "Update small adapter matrices — cheaper than full fine-tuning", "Replace RAG entirely", "Only work on GPUs with 80GB"], c: 1, why: "Parameter-efficient fine-tuning trains small adapters, drastically cutting compute and memory versus full fine-tuning." },
  { q: "A cross-encoder reranker is applied only to a shortlist because…", a: ["It is cheaper than embedding", "It scores query+passage together and is slower", "It needs no GPU", "It replaces the vector index"], c: 1, why: "Cross-encoders are accurate but expensive, so you rerank only the top candidates from fast bi-encoder retrieval." },
  { q: "In an ETL job for GenAI, 'idempotent + incremental' means…", a: ["Rebuild the whole index each run", "Re-runs don't duplicate data and only new/changed docs are processed", "Data is never updated", "It runs exactly once"], c: 1, why: "Upserting on a stable key makes re-runs safe (idempotent) and lets you process only deltas (incremental) — no full rebuilds." },
  { q: "The core loop of a tool-using agent is…", a: ["Extract-Transform-Load", "Reason → Act → Observe, repeated until done", "Map-Reduce", "Compile-Link-Run"], c: 1, why: "Agents plan (reason), call a tool (act), read the result (observe) and iterate until the goal is satisfied." },
];
const CHALLENGES = [
  { t: "Chunk a document recursively", d: "Write a function that splits text on paragraph boundaries, then packs sentences into ~400-char chunks with 60-char overlap.", s: "def recursive_chunk(text, size=400, overlap=60):\n    import re\n    sents = re.findall(r'[^.!?]+[.!?]', text)\n    chunks, cur = [], ''\n    for s in sents:\n        if len(cur) + len(s) > size and cur:\n            chunks.append(cur.strip())\n            cur = cur[-overlap:] + s\n        else:\n            cur += s\n    if cur.strip(): chunks.append(cur.strip())\n    return chunks" },
  { t: "Reciprocal Rank Fusion", d: "Fuse two ranked lists (vector + keyword) into one using RRF with k=60.", s: "def rrf(rankings, k=60):\n    scores = {}\n    for ranked in rankings:              # list of doc-id lists\n        for rank, doc in enumerate(ranked):\n            scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)\n    return sorted(scores, key=scores.get, reverse=True)" },
  { t: "Guarded RAG prompt", d: "Assemble a prompt that forbids answering outside the retrieved context and requires citations.", s: "def build_prompt(question, chunks):\n    ctx = '\\n\\n'.join(f'[{i+1}] {c}' for i, c in enumerate(chunks))\n    return (\n        'Answer ONLY from the context. Cite sources as [n]. '\n        'If the answer is not present, reply \"Not in the provided sources.\"\\n\\n'\n        f'Context:\\n{ctx}\\n\\nQuestion: {question}\\nAnswer:'\n    )" },
];
function Arena({ mark }) {
  const [tab, setTab] = useState("quiz");
  const [answers, setAnswers] = useState({});
  const [reveal, setReveal] = useState({});
  const correct = QUIZ.filter((q, i) => answers[i] === q.c).length;
  const answered = Object.keys(answers).length;
  useEffect(() => { if (answered >= 4) mark("arena", correct / QUIZ.length >= 0.6); }, [answered, correct, mark]);
  return (
    <div className="space-y-5">
      <Header icon={Beaker} title="Practice Arena" sub="Test recall and reps. Quiz builds the mental model; coding katas rehearse the muscle memory you'll use on day one." />
      <div className="flex gap-2">
        {[["quiz", "Concept quiz"], ["code", "Coding katas"]].map(([k, l]) => (
          <button key={k} onClick={() => setTab(k)} className={"rounded-lg border px-4 py-1.5 text-sm " + (tab === k ? "border-cyan-500 bg-cyan-500/15 text-cyan-200" : "border-slate-700 text-slate-400")}>{l}</button>
        ))}
      </div>

      {tab === "quiz" && (
        <div className="space-y-4">
          <Panel className="flex items-center justify-between p-4"><span className="text-sm text-slate-300">Score</span><span className="font-mono text-lg text-emerald-300">{correct}/{QUIZ.length}</span></Panel>
          {QUIZ.map((q, i) => (
            <Panel key={i} className="p-4">
              <div className="mb-2 text-sm font-medium text-slate-200">{i + 1}. {q.q}</div>
              <div className="grid gap-2 sm:grid-cols-2">
                {q.a.map((opt, j) => {
                  const picked = answers[i] === j;
                  const show = answers[i] !== undefined;
                  const isC = j === q.c;
                  let cls = "border-slate-700 text-slate-300 hover:border-slate-500";
                  if (show && isC) cls = "border-emerald-500 bg-emerald-500/10 text-emerald-200";
                  else if (show && picked && !isC) cls = "border-rose-500 bg-rose-500/10 text-rose-200";
                  return <button key={j} disabled={show} onClick={() => setAnswers((a) => ({ ...a, [i]: j }))} className={"rounded-lg border px-3 py-2 text-left text-xs transition " + cls}>{opt}</button>;
                })}
              </div>
              {answers[i] !== undefined && <p className="mt-2 rounded-md bg-slate-800/40 p-2 text-xs text-slate-400"><b className="text-slate-300">Why:</b> {q.why}</p>}
            </Panel>
          ))}
        </div>
      )}

      {tab === "code" && (
        <div className="space-y-4">
          {CHALLENGES.map((c, i) => (
            <Panel key={i} className="p-4">
              <div className="mb-1 flex items-center justify-between">
                <span className="text-sm font-semibold text-slate-200"><Terminal size={14} className="mr-1 inline text-cyan-400" />{c.t}</span>
                <button onClick={() => setReveal((r) => ({ ...r, [i]: !r[i] }))} className="rounded-md border border-slate-600 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-700/50">{reveal[i] ? "Hide" : "Reveal"} solution</button>
              </div>
              <p className="mb-2 text-xs text-slate-400">{c.d}</p>
              {reveal[i] ? (
                <div><div className="mb-1 flex justify-end"><CopyBtn text={c.s} /></div><pre className="overflow-auto rounded-lg border border-slate-700 bg-slate-950/70 p-3 font-mono text-[11px] leading-relaxed text-slate-300">{c.s}</pre></div>
              ) : (
                <div className="rounded-lg border border-dashed border-slate-700 bg-slate-950/30 p-6 text-center text-xs text-slate-500">Try it yourself first — then reveal the reference solution.</div>
              )}
            </Panel>
          ))}
        </div>
      )}
    </div>
  );
}

/* ===== DELIVERY KIT ===================================================== */
const PLAN = [
  { window: "Days 1–30", tone: "cyan", goal: "Learn the system, ship something small", items: ["Map the existing RAG/ETL stack, data sources and cloud accounts (AWS/Azure/GCP)", "Get a local dev + container build running; pass the existing test suite", "Own one low-risk ETL fix and one API endpoint end-to-end", "Read prod dashboards; understand current latency, cost and quality baselines"] },
  { window: "Days 31–60", tone: "amber", goal: "Own a feature, raise the quality bar", items: ["Deliver a hybrid-retrieval or reranking improvement with before/after evals", "Add an eval gate to CI so quality regressions block merges", "Instrument groundedness + drift monitoring for one pipeline", "Pair with software/ops teams on a clean model deployment"] },
  { window: "Days 61–90", tone: "emerald", goal: "Extend capability, prove leverage", items: ["Prototype an agentic workflow (tool use + orchestration) for a real task", "Run a small fine-tuning / LoRA experiment vs a RAG baseline, document the trade-off", "Harden a pipeline for incremental, idempotent re-ingestion", "Write the runbook + onboarding notes so the work outlives the contract"] },
];
const INTERVIEW = [
  { q: "Walk me through a RAG pipeline you'd build for our docs.", a: "Ingest & clean sources → recursive chunking (~400 tokens, 15% overlap) → embed → store in pgvector/Qdrant → hybrid retrieval (vector + BM25, fused) → optional cross-encoder rerank → guarded prompt with citations → generate. I'd add eval (groundedness, answer relevance) and monitoring from day one." },
  { q: "How do you keep the knowledge base fresh without full rebuilds?", a: "Idempotent, incremental ETL: hash each chunk, upsert on the hash so re-runs don't duplicate, and process only new/changed docs on a schedule or via event triggers. Re-embed only what changed." },
  { q: "When would you fine-tune instead of using RAG?", a: "RAG for factual freshness and citing sources. Fine-tuning (usually LoRA/QLoRA) for style, format, tone or a narrow task the base model handles poorly. Often both: fine-tune behaviour, RAG the facts." },
  { q: "How do you catch a silent quality regression in production?", a: "Eval gate in CI on a labelled set, plus live monitoring: groundedness scoring, embedding/data drift, latency and cost, structured logging with tracing, and a feedback loop that grows the eval set." },
  { q: "Describe your MLOps setup for a GenAI service.", a: "Versioned code/data/models, containerised runtime, CI running unit + prompt evals, canary deploy on Kubernetes with automated rollback on failed gates, and dashboards for latency, cost, drift and quality." },
];
function DeliveryKit() {
  const [openQ, setOpenQ] = useState(null);
  return (
    <div className="space-y-5">
      <Header icon={ClipboardList} title="Delivery Kit" sub="Turn readiness into a plan. A 30/60/90 you can paste into a proposal, plus interview answers mapped to the exact JD." />
      <div className="grid gap-4 lg:grid-cols-3">
        {PLAN.map((p) => (
          <Panel key={p.window} className="p-4">
            <div className="mb-1 flex items-center gap-2"><Target size={15} style={{ color: { cyan: "#22d3ee", amber: "#fbbf24", emerald: "#34d399" }[p.tone] }} /><span className="text-sm font-semibold text-slate-100">{p.window}</span></div>
            <p className="mb-3 text-xs italic text-slate-400">{p.goal}</p>
            <ul className="space-y-2">
              {p.items.map((it) => <li key={it} className="flex gap-2 text-xs text-slate-300"><ChevronRight size={13} className="mt-0.5 flex-none text-slate-500" />{it}</li>)}
            </ul>
          </Panel>
        ))}
      </div>
      <Panel className="p-4">
        <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-200"><BookOpen size={15} className="text-cyan-400" />Interview drill — JD-mapped answers</div>
        <div className="space-y-2">
          {INTERVIEW.map((it, i) => (
            <div key={i} className="rounded-lg border border-slate-700/70 bg-slate-950/40">
              <button onClick={() => setOpenQ(openQ === i ? null : i)} className="flex w-full items-center justify-between px-3 py-2.5 text-left text-sm text-slate-200">
                <span>{it.q}</span><ChevronRight size={15} className={"flex-none text-slate-500 transition " + (openQ === i ? "rotate-90" : "")} />
              </button>
              {openQ === i && <p className="border-t border-slate-800 px-3 py-2.5 text-xs leading-relaxed text-slate-300">{it.a}</p>}
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

/* ===== FLIGHT DECK (dashboard) ========================================== */
const SKILL_MAP = [
  { skill: "Python ETL for GenAI", jd: "must", mod: "etl" },
  { skill: "Cloud GenAI apps (AWS/Azure/GCP)", jd: "must", mod: "api" },
  { skill: "APIs integrating GenAI models", jd: "must", mod: "api" },
  { skill: "Hybrid RAG & optimisation", jd: "must", mod: "rag" },
  { skill: "Chunking · vector DB · retrieval", jd: "must", mod: "rag" },
  { skill: "AI agents & orchestration", jd: "must", mod: "agent" },
  { skill: "Basic fine-tuning (LoRA/QLoRA)", jd: "must", mod: "arena" },
  { skill: "DevOps/MLOps · CI/CD · monitoring", jd: "nice", mod: "mlops" },
];
function FlightDeck({ progress, readiness, go }) {
  return (
    <div className="space-y-5">
      <Panel className="flex flex-col items-center gap-6 p-6 sm:flex-row">
        <Reactor score={readiness} />
        <div className="flex-1">
          <h2 className="text-lg font-semibold text-slate-100">Contract readiness</h2>
          <p className="mt-1 max-w-xl text-sm text-slate-400">
            A single instrument aggregating every lab you complete. Work through the must-have modules first — RAG, ETL, APIs and the agent loop are the spine of this role. The number climbs as you practise.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Chip tone="cyan">6-month contract</Chip><Chip tone="slate">Toronto · 4 days onsite</Chip><Chip tone="amber">GenAI Developer</Chip>
          </div>
        </div>
      </Panel>

      <div>
        <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-300"><Target size={15} className="text-cyan-400" />JD → capability map</div>
        <div className="grid gap-3 sm:grid-cols-2">
          {SKILL_MAP.map((s) => {
            const done = progress[s.mod];
            return (
              <button key={s.skill} onClick={() => go(s.mod)} className="flex items-center gap-3 rounded-xl border border-slate-700/60 bg-slate-900/50 p-3 text-left transition hover:border-cyan-500/60">
                <span className={"flex h-9 w-9 flex-none items-center justify-center rounded-lg " + (done ? "bg-emerald-500/15 text-emerald-300" : "bg-slate-800 text-slate-500")}>{done ? <Check size={17} /> : <CircleDot size={15} />}</span>
                <span className="flex-1">
                  <span className="block text-sm text-slate-200">{s.skill}</span>
                  <span className="text-[11px] text-slate-500">{s.jd === "must" ? "Must-have" : "Nice-to-have"} · opens {MODULES.find((m) => m.id === s.mod).label}</span>
                </span>
                <ArrowRight size={15} className="text-slate-600" />
              </button>
            );
          })}
        </div>
      </div>

      <Panel className="p-4">
        <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-200"><ListChecks size={15} className="text-cyan-400" />How to use this deck</div>
        <ol className="space-y-1.5 text-xs text-slate-400">
          <li><b className="text-slate-300">1 · RAG Lab</b> — build the retrieval pipeline hands-on; it actually runs.</li>
          <li><b className="text-slate-300">2 · ETL Studio</b> — see raw data become an embeddable KB.</li>
          <li><b className="text-slate-300">3 · API Forge</b> — generate the FastAPI service that serves it.</li>
          <li><b className="text-slate-300">4 · Agent Loop</b> — internalise the reason-act-observe cycle.</li>
          <li><b className="text-slate-300">5 · MLOps + Arena</b> — operability and reps.</li>
          <li><b className="text-slate-300">6 · Delivery Kit</b> — 30/60/90 plan + interview drill to close.</li>
        </ol>
      </Panel>
    </div>
  );
}

/* ===== Shared header ==================================================== */
function Header({ icon: Icon, title, sub }) {
  return (
    <div className="flex items-start gap-3">
      <span className="mt-0.5 flex h-9 w-9 flex-none items-center justify-center rounded-lg bg-cyan-500/15 text-cyan-300"><Icon size={18} /></span>
      <div><h2 className="text-lg font-semibold text-slate-100">{title}</h2><p className="mt-0.5 max-w-2xl text-sm text-slate-400">{sub}</p></div>
    </div>
  );
}

/* ===== ROOT ============================================================= */
export default function GenAIDevSimulator() {
  const [active, setActive] = useState("deck");
  const [progress, setProgress] = useState({});
  const mark = useRef((id, done) => setProgress((p) => (p[id] === done ? p : { ...p, [id]: done }))).current;

  const readiness = useMemo(() => {
    const weights = { rag: 22, etl: 16, api: 16, agent: 14, mlops: 12, arena: 12, deliver: 8 };
    const total = Object.values(weights).reduce((a, b) => a + b, 0);
    const got = Object.entries(weights).reduce((a, [k, w]) => a + (progress[k] ? w : 0), 0);
    return (got / total) * 100;
  }, [progress]);

  useEffect(() => { if (active === "deliver") mark("deliver", true); }, [active, mark]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200" style={{ fontFamily: "ui-sans-serif, system-ui, sans-serif" }}>
      {/* top bar */}
      <div className="sticky top-0 z-10 border-b border-slate-800 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-cyan-500/15 text-cyan-300"><Cpu size={18} /></span>
          <div className="flex-1">
            <div className="text-sm font-semibold tracking-tight text-slate-100">GenAI Developer · Contract Flight Deck</div>
            <div className="text-[11px] text-slate-500">Practice · learn · deliver — 6-month Toronto contract</div>
          </div>
          <div className="flex items-center gap-2">
            <div className="hidden text-right sm:block"><div className="text-[10px] uppercase tracking-widest text-slate-500">Readiness</div><div className="font-mono text-sm text-slate-200">{Math.round(readiness)}%</div></div>
            <div className="h-16 w-16"><Reactor score={readiness} /></div>
          </div>
        </div>
      </div>

      <div className="mx-auto flex max-w-6xl flex-col gap-5 px-4 py-5 lg:flex-row">
        {/* nav */}
        <nav className="flex gap-2 overflow-x-auto lg:w-52 lg:flex-none lg:flex-col lg:overflow-visible">
          {MODULES.map((m) => {
            const on = active === m.id;
            const done = progress[m.id];
            return (
              <button key={m.id} onClick={() => setActive(m.id)}
                className={"flex flex-none items-center gap-2.5 rounded-lg border px-3 py-2 text-sm transition lg:w-full " +
                  (on ? "border-cyan-500/60 bg-cyan-500/10 text-slate-100" : "border-transparent text-slate-400 hover:bg-slate-900/60")}>
                <m.icon size={16} className={on ? "text-cyan-300" : "text-slate-500"} />
                <span className="flex-1 text-left">{m.label}</span>
                {done && <Check size={13} className="text-emerald-400" />}
              </button>
            );
          })}
        </nav>

        {/* content */}
        <main className="min-w-0 flex-1">
          {active === "deck" && <FlightDeck progress={progress} readiness={readiness} go={setActive} />}
          {active === "rag" && <RagLab mark={mark} />}
          {active === "etl" && <EtlStudio mark={mark} />}
          {active === "api" && <ApiForge mark={mark} />}
          {active === "agent" && <AgentLoop mark={mark} />}
          {active === "mlops" && <MlopsBoard mark={mark} />}
          {active === "arena" && <Arena mark={mark} />}
          {active === "deliver" && <DeliveryKit />}
        </main>
      </div>

      <footer className="mx-auto max-w-6xl px-4 pb-8 pt-2 text-center text-[11px] text-slate-600">
        Retrieval, scoring, chunking and ETL run entirely in your browser. Live generation is optional. Built as a practice rig for the GenAI Developer contract.
      </footer>
    </div>
  );
}
