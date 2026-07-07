import { useEffect, useMemo, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area,
} from "recharts";
import { Activity, ChevronRight, Filter, ShieldCheck, Check, X, Sun, Moon } from "lucide-react";

/* ────────────────────────────────────────────────────────────────────────
   Live data: GET /api/dashboard (served by `agenteval dashboard`).
   Payload shape mirrors AGENT_EVAL_SPEC.md (AgentRun / CheckResult):
   { run: {id, judge, kappa}, agents: [...], tasks: {agentId: [...]}, trend: {agentId: [...]} }
   ──────────────────────────────────────────────────────────────────────── */

const CATS = {
  tool_use:     { label: "tool use",     color: "#2F6FEB" },
  reasoning:    { label: "reasoning",    color: "#7A4FD0" },
  long_horizon: { label: "long horizon", color: "#0E8C86" },
  adversarial:  { label: "adversarial",  color: "#C25A2E" },
};

const METRICS = [
  { key: "passK",   label: "pass@k",       unit: "%", higher: true },
  { key: "pass1",   label: "pass@1",       unit: "%", higher: true },
  { key: "faith",   label: "faithfulness", unit: "%", higher: true },
  { key: "costRun", label: "cost / run",   unit: "",  higher: false },
  { key: "p50",     label: "latency p50",  unit: "",  higher: false },
  { key: "p95",     label: "latency p95",  unit: "",  higher: false },
];

const B_COLOR = "#E08A2B"; // Compare mode: agent B series (amber vs indigo A)

/* ──────────────────────────────────────────────────────────────────────── */

const fmtMs = (ms) => (ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`);
const fmtCost = (c) => (c === 0 ? "$0.00" : c < 0.001 ? "<$0.001" : `$${c.toFixed(3)}`);
const fmtVal = (key, v) =>
  v == null ? "—" : key === "costRun" ? fmtCost(v) : key === "p50" || key === "p95" ? fmtMs(v) : String(v);
const stepChip = { llm_call: "llm", tool_call: "tool", tool_result: "result" };

function PassDots({ passed, total }) {
  return (
    <span className="dots" title={`${passed}/${total} samples passed`}>
      {Array.from({ length: total }).map((_, i) => (
        <span key={i} className={"dot " + (i < passed ? "dot-pass" : "dot-fail")} />
      ))}
    </span>
  );
}

function Kpi({ label, value, unit, sub }) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}<span className="kpi-unit">{unit}</span></div>
      <div className="kpi-sub">{sub}</div>
    </div>
  );
}

function TraceWaterfall({ trace, T }) {
  const stepColor = { llm_call: T.accent, tool_call: "#0E8C86", tool_result: T.tick };
  const total = Math.max(trace.reduce((s, x) => s + x.ms, 0), 1);
  let acc = 0;
  return (
    <div className="wf">
      {trace.map((s, i) => {
        const left = (acc / total) * 100;
        const width = Math.max((s.ms / total) * 100, 2);
        acc += s.ms;
        return (
          <div className="wf-row" key={i}>
            <div className="wf-gutter">
              <span className="chip" style={{ color: stepColor[s.type], background: stepColor[s.type] + "22" }}>
                {stepChip[s.type]}
              </span>
              <span className="wf-name">{s.label}</span>
              {s.tokensIn != null && <span className="wf-tok">{s.tokensIn}→{s.tokensOut} tok</span>}
            </div>
            <div className="wf-track">
              <div className="wf-bar" style={{ left: `${left}%`, width: `${width}%`, background: stepColor[s.type] }} />
              <span className="wf-ms" style={{ left: `${Math.min(left + width, 82)}%` }}>{fmtMs(s.ms)}</span>
            </div>
            {s.note && <div className="wf-note">{s.note}</div>}
          </div>
        );
      })}
    </div>
  );
}

function TracePanel({ task, T }) {
  return (
    <div className="trace">
      {(task.prompt || task.output) && (
        <div className="io">
          {task.prompt && <div className="io-row"><span className="io-tag">prompt</span><span>{task.prompt}</span></div>}
          {task.output && <div className="io-row"><span className="io-tag">output</span><span className={task.ok ? "" : "io-bad"}>{task.output}</span></div>}
        </div>
      )}
      <div className="trace-cols">
        <div>
          <div className="section-label">execution trace</div>
          <TraceWaterfall trace={task.trace} T={T} />
        </div>
        <div>
          <div className="section-label">checks</div>
          <div className="checks">
            {task.checks.map((c, i) => (
              <div className="check" key={i}>
                <span className={"cbadge " + (c.passed ? "cpass" : "cfail")}>
                  {c.passed ? <Check size={11} strokeWidth={3} /> : <X size={11} strokeWidth={3} />}
                </span>
                <div className="check-body">
                  <div className="check-top">
                    <span className="check-name">{c.name}</span>
                    <span className="check-score">{c.score.toFixed(2)}</span>
                  </div>
                  {c.detail && <div className="check-detail">{c.detail}</div>}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function AgentEvalDashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [dark, setDark] = useState(false);
  const [view, setView] = useState("scorecard"); // scorecard | compare
  const [filter, setFilter] = useState("all");   // all | failures
  const [open, setOpen] = useState(null);
  const [curId, setCurId] = useState(null);
  const [aId, setAId] = useState(null);
  const [bId, setBId] = useState(null);

  useEffect(() => {
    fetch("/api/dashboard")
      .then((r) => { if (!r.ok) throw new Error(`API ${r.status}`); return r.json(); })
      .then((d) => {
        if (!d.agents.length) throw new Error("no runs in the database yet");
        setData(d);
        const sorted = [...d.agents].sort((x, y) => y.passK - x.passK);
        setCurId(sorted[0].id);
        setAId(sorted[0].id);
        setBId(sorted[sorted.length - 1].id);
      })
      .catch((e) => setError(e.message));
  }, []);

  const T = dark
    ? { accent: "#7B84F2", accentLite: "#565FB0", grid: "#232A38", tick: "#6B7488", tipBg: "#1B2130", tipBorder: "#2A3242", tipLabel: "#E7EAF0", fail: "#EA6B74" }
    : { accent: "#4B54C9", accentLite: "#AEB3EE", grid: "#E6E9EF", tick: "#9AA3B2", tipBg: "#FFFFFF", tipBorder: "#E6E9EF", tipLabel: "#161A22", fail: "#D64550" };
  const tipStyle = { background: T.tipBg, border: `1px solid ${T.tipBorder}`, borderRadius: 8, boxShadow: "0 6px 20px #00000026", fontFamily: "var(--mono)", fontSize: 11, padding: "6px 9px", color: T.tipLabel };
  const tipLabelStyle = { fontFamily: "var(--mono)", fontSize: 11, color: T.tipLabel };

  const AGENTS = data?.agents ?? [];
  const RUN = data?.run ?? {};
  const current = AGENTS.find((x) => x.id === curId) ?? AGENTS[0];
  const TASKS = useMemo(() => (data && current ? data.tasks[current.id] ?? [] : []), [data, current]);
  const TREND = data && current ? data.trend[current.id] ?? [] : [];
  const CATBARS = current ? Object.entries(current.cats).map(([cat, passk]) => ({ cat, passk })) : [];
  const FW_CHART = AGENTS.map((a) => ({ name: a.short, pass1: a.pass1, passk: a.passK }));
  const SUMMARY = current && {
    passK: current.passK, pass1: current.pass1, faithfulness: current.faith,
    costRun: current.costRun, p50: current.p50, p95: current.p95,
    tasks: TASKS.length, samples: TASKS.reduce((s, t) => s + t.samples, 0),
    k: TASKS[0]?.samples ?? 1,
  };

  const failCount = TASKS.filter((t) => !t.ok).length;
  const shown = filter === "all" ? TASKS : TASKS.filter((t) => !t.ok);

  const a = AGENTS.find((x) => x.id === aId) ?? AGENTS[0];
  const b = AGENTS.find((x) => x.id === bId) ?? AGENTS[AGENTS.length - 1];
  const catCompare = a && b
    ? Object.keys(CATS).filter((c) => c in a.cats || c in b.cats)
        .map((c) => ({ cat: CATS[c].label, A: a.cats[c] ?? 0, B: b.cats[c] ?? 0 }))
    : [];
  const pickA = (id) => { if (id === bId) setBId(aId); setAId(id); };
  const pickB = (id) => { if (id === aId) setAId(bId); setBId(id); };

  if (error || !data || !current) {
    return (
      <div className={"root" + (dark ? " dark" : "")}>
        <style>{css}</style>
        <div className="app" style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "60vh", flexDirection: "column", gap: 10 }}>
          <span className="mark"><Activity size={15} strokeWidth={2.4} /></span>
          {error
            ? <><div style={{ fontWeight: 600 }}>couldn't load results ({error})</div>
                <div style={{ fontSize: 12.5, color: "var(--soft)" }}>run <code>agenteval run</code>, <code>agenteval evaluate</code>, then <code>agenteval dashboard</code></div></>
            : <div style={{ fontSize: 13, color: "var(--soft)" }}>loading scorecard…</div>}
        </div>
      </div>
    );
  }

  return (
    <div className={"root" + (dark ? " dark" : "")}>
      <style>{css}</style>
      <div className="app">
        <header className="topbar">
          <div className="brand">
            <span className="mark"><Activity size={15} strokeWidth={2.4} /></span>
            <span className="brand-name">agenteval</span>
            <span className="brand-sub">{view === "compare" ? "compare" : "scorecard"}</span>
          </div>
          <div className="top-right">
            {view === "scorecard" && AGENTS.length > 1 && (
              <div className="pick">
                <label>agent</label>
                <select className="sel" value={current.id} onChange={(e) => { setCurId(e.target.value); setOpen(null); }}>
                  {AGENTS.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
                </select>
              </div>
            )}
            <div className="seg">
              <button className={view === "scorecard" ? "on" : ""} onClick={() => setView("scorecard")}>Scorecard</button>
              <button className={view === "compare" ? "on" : ""} onClick={() => setView("compare")}>Compare</button>
            </div>
            {view === "scorecard" && RUN.kappa != null && (
              <span className="kappa" title="Judge calibrated against a human-scored golden set">
                <ShieldCheck size={13} strokeWidth={2.2} />
                judge κ {RUN.kappa.toFixed(2)} · calibrated
              </span>
            )}
            <button className="theme-btn" onClick={() => setDark(!dark)} aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}>
              {dark ? <Sun size={15} /> : <Moon size={15} />}
            </button>
          </div>
        </header>

        {view === "scorecard" ? (
          <>
            <section className="kpis">
              <Kpi label="pass@k" value={SUMMARY.passK} unit="%" sub={`${SUMMARY.samples} samples · k=${SUMMARY.k}`} />
              <Kpi label="pass@1" value={SUMMARY.pass1} unit="%" sub={`${SUMMARY.passK - SUMMARY.pass1}pt gap → variance`} />
              <Kpi label="faithfulness" value={SUMMARY.faithfulness ?? "—"} unit={SUMMARY.faithfulness != null ? "%" : ""} sub="claims grounded in tools" />
              <Kpi label="cost / run" value={fmtCost(SUMMARY.costRun)} unit="" sub={`over ${SUMMARY.tasks} tasks`} />
              <Kpi label="latency p95" value={(SUMMARY.p95 / 1000).toFixed(1)} unit="s" sub={`p50 ${fmtMs(SUMMARY.p50)}`} />
            </section>

            <section className="grid2">
              <div className="card">
                <div className="card-head">
                  <h2>Frameworks compared</h2>
                  <div className="legend">
                    <span><i style={{ background: T.accentLite }} />pass@1</span>
                    <span><i style={{ background: T.accent }} />pass@k</span>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={186}>
                  <BarChart data={FW_CHART} barGap={3} barCategoryGap={26} margin={{ top: 6, right: 6, left: -20, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke={T.grid} />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: T.tick, fontFamily: "var(--mono)" }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: T.tick }} axisLine={false} tickLine={false} width={36} />
                    <Tooltip cursor={{ fill: "#80808014" }} contentStyle={tipStyle} labelStyle={tipLabelStyle} itemStyle={tipLabelStyle} />
                    <Bar dataKey="pass1" name="pass@1" fill={T.accentLite} radius={[3, 3, 0, 0]} />
                    <Bar dataKey="passk" name="pass@k" fill={T.accent} radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <div className="fw-stats">
                  {AGENTS.map((f) => (
                    <div className="fw-stat" key={f.id}>
                      <span className="fw-name">{f.short}</span>
                      <span>{fmtCost(f.costRun)}</span>
                      <span>{fmtMs(f.p50)}</span>
                      <span>faith {f.faith != null ? `${f.faith}%` : "—"}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="col-right">
                <div className="card">
                  <div className="card-head"><h2>By category</h2></div>
                  <div className="cats">
                    {CATBARS.map((c) => (
                      <div className="cat-row" key={c.cat}>
                        <span className="cat-chip" style={{ color: CATS[c.cat].color, background: CATS[c.cat].color + "22" }}>
                          {CATS[c.cat].label}
                        </span>
                        <div className="cat-bar"><div style={{ width: `${c.passk}%`, background: CATS[c.cat].color }} /></div>
                        <span className="cat-pct">{c.passk}%</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="card">
                  <div className="card-head"><h2>pass@k over runs</h2></div>
                  <ResponsiveContainer width="100%" height={72}>
                    <AreaChart data={TREND} margin={{ top: 8, right: 6, left: 6, bottom: 0 }}>
                      <defs>
                        <linearGradient id="spark" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={T.accent} stopOpacity={0.26} />
                          <stop offset="100%" stopColor={T.accent} stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <Tooltip cursor={{ stroke: T.grid }} contentStyle={tipStyle} labelStyle={tipLabelStyle} itemStyle={tipLabelStyle} />
                      <Area dataKey="passK" stroke={T.accent} strokeWidth={2} fill="url(#spark)" dot={TREND.length < 2} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </section>

            <section className="card table-card">
              <div className="card-head table-head">
                <h2>Tasks</h2>
                <div className="filters">
                  <Filter size={13} strokeWidth={2} className="filter-ic" />
                  <button className={filter === "all" ? "fbtn on" : "fbtn"} onClick={() => setFilter("all")}>
                    all <span>{TASKS.length}</span>
                  </button>
                  <button className={filter === "failures" ? "fbtn on" : "fbtn"} onClick={() => setFilter("failures")}>
                    failures <span>{failCount}</span>
                  </button>
                </div>
              </div>

              <div className="thead">
                <span>task</span><span>category</span><span className="ta-c">pass@k</span>
                <span className="ta-r">cost</span><span className="ta-r">p50</span><span className="ta-r">judge</span><span />
              </div>

              <div className="rows">
                {shown.map((t) => {
                  const isOpen = open === t.id;
                  return (
                    <div className="rowwrap" key={t.id}>
                      <button className={"row" + (isOpen ? " row-open" : "")} onClick={() => setOpen(isOpen ? null : t.id)}>
                        <span className="tid">{t.id}</span>
                        <span>
                          <span className="cat-chip sm" style={{ color: CATS[t.category].color, background: CATS[t.category].color + "22" }}>
                            {CATS[t.category].label}
                          </span>
                        </span>
                        <span className="ta-c"><PassDots passed={t.passed} total={t.samples} /></span>
                        <span className="ta-r mono">{fmtCost(t.cost)}</span>
                        <span className="ta-r mono">{fmtMs(t.p50)}</span>
                        <span className="ta-r mono" style={{ color: t.judge >= 0.6 ? "inherit" : T.fail }}>{t.judge.toFixed(2)}</span>
                        <span className="ta-r"><ChevronRight size={15} className={"chev" + (isOpen ? " chev-open" : "")} /></span>
                      </button>
                      {isOpen && <TracePanel task={t} T={T} />}
                    </div>
                  );
                })}
              </div>
            </section>
          </>
        ) : (
          <>
            <section className="card cmp-picker">
              <div className="cmp-head">
                <div className="pick">
                  <label>A</label>
                  <select className="sel" value={a.id} onChange={(e) => pickA(e.target.value)}>
                    {AGENTS.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
                  </select>
                </div>
                <span className="vs">vs</span>
                <div className="pick">
                  <label>B</label>
                  <select className="sel" value={b.id} onChange={(e) => pickB(e.target.value)}>
                    {AGENTS.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
                  </select>
                </div>
              </div>
            </section>

            <section className="grid2 cmp-grid">
              <div className="card">
                <div className="card-head"><h2>Headline metrics</h2></div>
                <div className="diff">
                  <div className="diff-row diff-h">
                    <span>metric</span><span className="ta-r">{a.short}</span><span className="ta-r">{b.short}</span><span className="ta-r">Δ</span>
                  </div>
                  {METRICS.map((m) => {
                    const av = a[m.key], bv = b[m.key];
                    if (av == null || bv == null) {
                      return (
                        <div className="diff-row" key={m.key}>
                          <span className="diff-metric">{m.label}</span>
                          <span className="dval">{fmtVal(m.key, av)}{av != null ? m.unit : ""}</span>
                          <span className="dval">{fmtVal(m.key, bv)}{bv != null ? m.unit : ""}</span>
                          <span className="ddelta">—</span>
                        </div>
                      );
                    }
                    const raw = av - bv;
                    const tie = raw === 0;
                    const favorsA = m.higher ? raw > 0 : raw < 0;
                    const mag = Math.abs(raw);
                    const delta = m.unit === "%" ? `${mag}pt` : m.key === "costRun" ? `$${mag.toFixed(3)}` : fmtMs(mag);
                    return (
                      <div className="diff-row" key={m.key}>
                        <span className="diff-metric">{m.label}</span>
                        <span className={"dval" + (!tie && favorsA ? " win" : "")}>{fmtVal(m.key, av)}{m.unit}</span>
                        <span className={"dval" + (!tie && !favorsA ? " win" : "")}>{fmtVal(m.key, bv)}{m.unit}</span>
                        <span className="ddelta">{tie ? "—" : delta}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="card">
                <div className="card-head">
                  <h2>By category · pass@k</h2>
                  <div className="legend">
                    <span><i style={{ background: T.accent }} />{a.short}</span>
                    <span><i style={{ background: B_COLOR }} />{b.short}</span>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={222}>
                  <BarChart data={catCompare} barGap={3} barCategoryGap={20} margin={{ top: 6, right: 6, left: -20, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke={T.grid} />
                    <XAxis dataKey="cat" tick={{ fontSize: 10, fill: T.tick, fontFamily: "var(--mono)" }} axisLine={false} tickLine={false} interval={0} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: T.tick }} axisLine={false} tickLine={false} width={34} />
                    <Tooltip cursor={{ fill: "#80808014" }} contentStyle={tipStyle} labelStyle={tipLabelStyle} itemStyle={tipLabelStyle} />
                    <Bar dataKey="A" name={a.short} fill={T.accent} radius={[3, 3, 0, 0]} />
                    <Bar dataKey="B" name={b.short} fill={B_COLOR} radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>
          </>
        )}

        <footer className="foot">
          {RUN.id}{RUN.judge ? ` · judge ${RUN.judge}` : ""} · programmatic + LLM-as-judge · pass@k
        </footer>
      </div>
    </div>
  );
}

const css = `
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

.root{
  --sans:'Space Grotesk',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --ink:#161A22; --soft:#586172; --faint:#9AA3B2;
  --bg:#F1F3F6; --surf:#FFFFFF; --line:#E6E9EF; --line2:#D3D8E0;
  --accent:#4B54C9; --pass:#1F9D6B; --fail:#D64550;
  --hover:#F7F8FB; --open-bg:#F5F6FC; --accent-soft:#EBEDFB;
  --kappa-bg:#E4F4EC; --kappa-bd:#C7E8D6;
  font-family:var(--sans); color:var(--ink); background:var(--bg);
  min-height:100vh; -webkit-font-smoothing:antialiased;
}
.root.dark{
  --ink:#E7EAF0; --soft:#A2ABBC; --faint:#6B7488;
  --bg:#0E1219; --surf:#151A24; --line:#232A38; --line2:#2E3648;
  --accent:#7B84F2; --pass:#3BC08A; --fail:#EA6B74;
  --hover:#1A2029; --open-bg:#1C222E; --accent-soft:#232A45;
  --kappa-bg:#122320; --kappa-bd:#1E3B2F;
}
.root *{box-sizing:border-box;}
body{margin:0;}
.app{max-width:1080px; margin:0 auto; padding:26px 22px 40px; font-variant-numeric:tabular-nums;}
.app h2{font-size:13px; font-weight:600; margin:0; letter-spacing:.01em; color:var(--ink);}
.mono,.kpi-value,.tid,.wf-name,.wf-tok,.wf-ms,.check-name,.check-score,.cat-pct,.kappa,.fw-stat span,.dval,.ddelta,.vs,.sel{font-family:var(--mono);}

.topbar{display:flex; align-items:center; justify-content:space-between; gap:16px; padding-bottom:18px; flex-wrap:wrap;}
.brand{display:flex; align-items:baseline; gap:9px;}
.mark{display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px; border-radius:7px; background:var(--accent); color:#fff; align-self:center;}
.brand-name{font-weight:700; font-size:17px; letter-spacing:-.01em;}
.brand-sub{font-size:12px; color:var(--faint); letter-spacing:.14em; text-transform:uppercase;}
.top-right{display:flex; align-items:center; gap:10px; flex-wrap:wrap;}
.kappa{display:inline-flex; align-items:center; gap:6px; font-size:11.5px; color:var(--pass); background:var(--kappa-bg); border:1px solid var(--kappa-bd); padding:5px 9px; border-radius:7px;}

.seg{display:inline-flex; background:var(--surf); border:1px solid var(--line); border-radius:9px; padding:2px;}
.seg button{font-family:var(--sans); font-size:11.5px; padding:5px 13px; border:none; background:transparent; color:var(--soft); border-radius:7px; cursor:pointer; transition:all .13s;}
.seg button.on{background:var(--accent-soft); color:var(--accent); font-weight:500;}
.seg button:hover:not(.on){color:var(--ink);}
.theme-btn{width:32px; height:32px; border:1px solid var(--line); background:var(--surf); border-radius:9px; color:var(--soft); display:inline-flex; align-items:center; justify-content:center; cursor:pointer; transition:all .13s;}
.theme-btn:hover{color:var(--ink); border-color:var(--line2);}

.kpis{display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:14px;}
.kpi{background:var(--surf); border:1px solid var(--line); border-radius:12px; padding:14px 15px;}
.kpi-label{font-size:10.5px; letter-spacing:.13em; text-transform:uppercase; color:var(--faint); margin-bottom:9px;}
.kpi-value{font-size:29px; font-weight:600; line-height:1; letter-spacing:-.02em;}
.kpi-unit{font-size:15px; color:var(--soft); margin-left:2px; font-weight:500;}
.kpi-sub{font-size:11px; color:var(--soft); margin-top:8px;}

.grid2{display:grid; grid-template-columns:1.62fr 1fr; gap:12px; margin-bottom:12px;}
.cmp-grid{grid-template-columns:1fr 1fr;}
.col-right{display:flex; flex-direction:column; gap:12px;}
.card{background:var(--surf); border:1px solid var(--line); border-radius:12px; padding:15px 16px;}
.card-head{display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;}
.legend{display:flex; gap:12px; font-size:11px; color:var(--soft);}
.legend span{display:inline-flex; align-items:center; gap:5px;}
.legend i{width:9px; height:9px; border-radius:2px; display:inline-block;}

.fw-stats{margin-top:10px; border-top:1px solid var(--line); padding-top:9px; display:flex; flex-direction:column; gap:6px;}
.fw-stat{display:grid; grid-template-columns:1.1fr .8fr .8fr 1fr; font-size:11px; color:var(--soft);}
.fw-stat .fw-name{color:var(--ink);}

.cats{display:flex; flex-direction:column; gap:11px;}
.cat-row{display:grid; grid-template-columns:96px 1fr 40px; align-items:center; gap:10px;}
.cat-bar{height:7px; background:var(--bg); border-radius:4px; overflow:hidden;}
.cat-bar>div{height:100%; border-radius:4px;}
.cat-pct{font-size:11.5px; color:var(--soft); text-align:right;}
.cat-chip{font-size:10.5px; padding:3px 8px; border-radius:6px; font-weight:500; text-align:center; justify-self:start;}
.cat-chip.sm{font-size:10px; padding:2px 7px;}

.cmp-picker{margin-bottom:12px; padding:14px 16px;}
.cmp-head{display:flex; align-items:center; gap:14px; flex-wrap:wrap;}
.pick{display:flex; align-items:center; gap:8px;}
.pick label{font-size:10.5px; letter-spacing:.13em; text-transform:uppercase; color:var(--faint); font-weight:600;}
.sel{font-size:12px; color:var(--ink); background:var(--surf); border:1px solid var(--line); border-radius:8px; padding:7px 10px; cursor:pointer; max-width:240px;}
.sel:hover{border-color:var(--line2);}
.vs{font-size:11px; color:var(--faint); letter-spacing:.1em;}

.diff{display:flex; flex-direction:column;}
.diff-row{display:grid; grid-template-columns:1.25fr 1fr 1fr .7fr; align-items:center; gap:8px; padding:10px 4px; border-bottom:1px solid var(--line);}
.diff-h{font-size:10px; letter-spacing:.1em; text-transform:uppercase; color:var(--faint); padding-bottom:8px;}
.diff-h span{font-family:var(--sans);}
.diff-metric{font-size:12.5px;}
.dval{font-size:12.5px; text-align:right; padding:5px 9px; border-radius:7px; color:var(--soft);}
.dval.win{background:var(--accent-soft); color:var(--accent); font-weight:600;}
.ddelta{font-size:11px; color:var(--faint); text-align:right;}

.table-card{padding-top:15px;}
.table-head{margin-bottom:10px;}
.filters{display:flex; align-items:center; gap:6px;}
.filter-ic{color:var(--faint); margin-right:2px;}
.fbtn{font-family:var(--sans); font-size:11.5px; color:var(--soft); background:transparent; border:1px solid var(--line); border-radius:7px; padding:5px 10px; cursor:pointer; display:inline-flex; gap:6px; align-items:center; transition:all .14s;}
.fbtn span{font-family:var(--mono); color:var(--faint);}
.fbtn:hover{border-color:var(--line2);}
.fbtn.on{border-color:var(--accent); color:var(--accent); background:var(--accent-soft);}
.fbtn.on span{color:var(--accent);}

.thead,.row{display:grid; grid-template-columns:1.9fr 1.1fr 1fr .7fr .7fr .7fr 34px; align-items:center; gap:10px;}
.thead{font-size:10px; letter-spacing:.11em; text-transform:uppercase; color:var(--faint); padding:0 6px 8px; border-bottom:1px solid var(--line);}
.ta-r{text-align:right;} .ta-c{text-align:center;}
.rows{display:flex; flex-direction:column;}
.rowwrap{border-bottom:1px solid var(--line);}
.row{width:100%; text-align:left; background:transparent; border:none; border-radius:8px; padding:11px 6px; cursor:pointer; font-family:var(--sans); font-size:13px; color:var(--ink); transition:background .12s;}
.row:hover{background:var(--hover);}
.row-open{background:var(--open-bg);}
.tid{font-size:12.5px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
.chev{color:var(--faint); transition:transform .16s;}
.chev-open{transform:rotate(90deg); color:var(--accent);}

.dots{display:inline-flex; gap:4px; justify-content:center;}
.dot{width:8px; height:8px; border-radius:50%;}
.dot-pass{background:var(--pass);}
.dot-fail{background:transparent; border:1.5px solid var(--fail);}

.trace{padding:4px 6px 16px; animation:reveal .18s ease;}
@keyframes reveal{from{opacity:0; transform:translateY(-4px);} to{opacity:1; transform:none;}}
.io{background:var(--bg); border-radius:9px; padding:11px 13px; margin-bottom:14px; display:flex; flex-direction:column; gap:7px;}
.io-row{display:flex; gap:10px; font-size:12.5px; line-height:1.5;}
.io-tag{font-family:var(--mono); font-size:10px; letter-spacing:.1em; text-transform:uppercase; color:var(--faint); flex:0 0 46px; padding-top:2px;}
.io-bad{color:var(--fail);}
.trace-cols{display:grid; grid-template-columns:1.35fr 1fr; gap:24px;}
.section-label{font-size:10px; letter-spacing:.12em; text-transform:uppercase; color:var(--faint); margin-bottom:10px;}

.wf{display:flex; flex-direction:column; gap:9px;}
.wf-row{display:grid; grid-template-columns:1fr; gap:3px;}
.wf-gutter{display:flex; align-items:center; gap:8px; font-size:11.5px;}
.chip{font-family:var(--mono); font-size:9.5px; padding:2px 6px; border-radius:5px; font-weight:600; letter-spacing:.02em;}
.wf-name{color:var(--ink);}
.wf-tok{color:var(--faint); font-size:10.5px; margin-left:auto;}
.wf-track{position:relative; height:9px; background:var(--bg); border-radius:5px;}
.wf-bar{position:absolute; top:0; height:9px; border-radius:5px; min-width:3px;}
.wf-ms{position:absolute; top:-1px; font-size:9.5px; color:var(--faint); transform:translateX(6px); white-space:nowrap;}
.wf-note{font-size:10.5px; color:var(--soft); padding-left:2px;}

.checks{display:flex; flex-direction:column; gap:10px;}
.check{display:flex; gap:9px; align-items:flex-start;}
.cbadge{flex:0 0 17px; height:17px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; color:#fff; margin-top:1px;}
.cpass{background:var(--pass);} .cfail{background:var(--fail);}
.check-body{flex:1; min-width:0;}
.check-top{display:flex; justify-content:space-between; gap:8px; align-items:baseline;}
.check-name{font-size:11.5px; color:var(--ink); word-break:break-word;}
.check-score{font-size:11px; color:var(--soft);}
.check-detail{font-size:11px; color:var(--soft); margin-top:2px; line-height:1.45;}

.foot{margin-top:16px; font-family:var(--mono); font-size:10.5px; color:var(--faint); text-align:center;}

.row:focus-visible,.fbtn:focus-visible,.seg button:focus-visible,.theme-btn:focus-visible,.sel:focus-visible{outline:2px solid var(--accent); outline-offset:2px;}

@media (max-width:760px){
  .kpis{grid-template-columns:repeat(2,1fr);}
  .grid2,.cmp-grid{grid-template-columns:1fr;}
  .trace-cols{grid-template-columns:1fr; gap:18px;}
  .thead{display:none;}
  .row{grid-template-columns:1fr auto; grid-auto-rows:auto; gap:6px 10px; padding:12px 6px;}
  .row>.ta-r,.row>.ta-c{text-align:left;}
}
@media (prefers-reduced-motion:reduce){
  .root *{animation:none !important; transition:none !important;}
}
`;
