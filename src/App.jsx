import React, { useMemo, useState } from "react";
import {
  Activity, BarChart3, Bell, BrainCircuit, ChevronDown, CircleHelp,
  Database, Download, FileBarChart, FlaskConical, Gauge, Menu, Search,
  Settings2, Sparkles, Target, Upload, Wallet, X, Zap
} from "lucide-react";
import Plot from "react-plotly.js";
import "./App.css";

const nav = [
  { label: "Overview", icon: BarChart3, group: "WORKSPACE" },
  { label: "Upload Dataset", icon: Upload, group: "WORKSPACE" },
  { label: "Causal Analysis", icon: FlaskConical, group: "ANALYTICS" },
  { label: "Budget Optimization", icon: Wallet, group: "ANALYTICS" },
  { label: "Reports", icon: FileBarChart, group: "ANALYTICS" }
];

const demoResult = {
  ate: 0.084,
  customers_analysed: 12480,
  ite: [0.03, 0.07, 0.11, 0.14, 0.09, 0.18, 0.05, 0.12],
  note: "Illustrative demo output — connect the backend to calculate results from your dataset."
};

async function apiRequest(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || data.message || `Request failed (${response.status})`);
  return data;
}

function App() {
  const [page, setPage] = useState("Overview");
  const [mobileNav, setMobileNav] = useState(false);
  const [toast, setToast] = useState("");
  const [dataset, setDataset] = useState(null);
  const [uploadState, setUploadState] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [analysisMode, setAnalysisMode] = useState("demo");
  const [budget, setBudget] = useState(25000);
  const [customers, setCustomers] = useState(5000);
  const [reports, setReports] = useState(null);
  const [busy, setBusy] = useState(false);

  const notify = (message) => {
    setToast(message);
    window.setTimeout(() => setToast(""), 3500);
  };

  const runAnalysis = async (treatment = "offer_received", outcome = "purchase") => {
    setBusy(true);
    try {
      const result = await apiRequest("/api/causal-analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset: uploadState?.dataset_id || dataset?.name || "", treatment, outcome })
      });
      setAnalysis(result);
      setAnalysisMode("backend");
      notify("Analysis completed using the backend.");
    } catch {
      setAnalysis(demoResult);
      setAnalysisMode("demo");
      notify("Backend unavailable: showing clearly labelled demo output.");
    } finally {
      setBusy(false);
    }
  };

  const uploadDataset = async (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      notify("Please select a CSV file.");
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      notify("File must be smaller than 50 MB.");
      return;
    }
    setDataset(file);
    setBusy(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const result = await apiRequest("/api/upload-dataset", { method: "POST", body: form });
      setUploadState(result);
      notify("Dataset uploaded to backend successfully.");
    } catch {
      setUploadState({ dataset_id: file.name, rows: null, columns: null, localOnly: true });
      notify("CSV selected locally. Backend upload endpoint is not connected yet.");
    } finally {
      setBusy(false);
    }
  };

  const loadReports = async () => {
    try {
      const result = await apiRequest("/api/reports");
      setReports(result);
      notify("Report loaded from backend.");
    } catch {
      setReports({
        ate: analysis?.ate ?? null,
        customers_analysed: analysis?.customers_analysed ?? null,
        revenue_impact: null,
        budget_utilization: null,
        demo: true
      });
      notify("Backend unavailable: report contains only available demo/session data.");
    }
  };

  const optimize = useMemo(() => {
    const amount = Math.max(0, Number(budget) || 0);
    const people = Math.max(1, Number(customers) || 1);
    const estimatedLift = analysis?.ate ?? demoResult.ate;
    return { amount, people, estimatedLift, spendPerCustomer: amount / people, estimatedIncrementalConversions: Math.round(people * estimatedLift) };
  }, [budget, customers, analysis]);

  const exportCSV = () => {
    const data = reports || { ate: analysis?.ate ?? "", customers_analysed: analysis?.customers_analysed ?? "", note: "No backend report loaded" };
    const csv = "metric,value\n" + Object.entries(data).map(([k, v]) => `"${k}","${Array.isArray(v) ? v.join(";") : String(v ?? "")}"`).join("\n");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8;" }));
    link.download = "econocausal-report.csv";
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const pageTitle = page === "Overview" ? "Overview" : page;

  return (
    <div className="shell">
      {mobileNav && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Activity size={22} /></div>
          <div><strong>Econo<span>Causal</span></strong><small>PRICING INTELLIGENCE</small></div>
          <button className="icon-btn sidebar-close" onClick={() => setMobileNav(false)} aria-label="Close menu"><X size={18} /></button>
        </div>
        <div className="workspace-pill"><div className="workspace-dot" /><div><b>Research workspace</b><small>Local environment</small></div><ChevronDown size={15} /></div>
        {["WORKSPACE", "ANALYTICS"].map(group => <div className="nav-group" key={group}>
          <p className="nav-label">{group}</p>
          {nav.filter(n => n.group === group).map(item => {
            const Icon = item.icon;
            return <button key={item.label} className={`nav-item ${page === item.label ? "active" : ""}`} onClick={() => { setPage(item.label); setMobileNav(false); }}>
              <Icon size={18} strokeWidth={1.8} /><span>{item.label}</span>{item.label === "Causal Analysis" && <span className="nav-new">DML</span>}
            </button>;
          })}
        </div>)}
        <div className="sidebar-bottom">
          <div className="help-card"><div className="help-icon"><CircleHelp size={18} /></div><b>Need a hand?</b><p>Explore the workflow or connect your backend.</p><button onClick={() => notify("Workflow: upload CSV → run DML → optimize budget → export report.")}>View quick guide <span>↗</span></button></div>
          <div className="user-row"><div className="avatar">EC</div><div><b>Project workspace</b><small>Student research project</small></div><Settings2 size={17} /></div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <button className="icon-btn mobile-menu" onClick={() => setMobileNav(true)} aria-label="Open menu"><Menu size={20} /></button>
          <div className="breadcrumbs">Workspace <span>/</span> <b>{pageTitle}</b></div>
          <div className="top-actions"><div className="searchbox"><Search size={16} /><input placeholder="Search workspace..." onKeyDown={e => e.key === "Enter" && notify("Search is not connected yet.")} /></div><button className="icon-btn notification" onClick={() => notify("You're all caught up.")} aria-label="Notifications"><Bell size={18} /><i /></button><div className="top-avatar">V</div></div>
        </header>

        <div className="content">
          {page === "Overview" && <Overview dataset={dataset} analysis={analysis} analysisMode={analysisMode} setPage={setPage} runAnalysis={runAnalysis} busy={busy} />}
          {page === "Upload Dataset" && <UploadPage dataset={dataset} uploadState={uploadState} busy={busy} onUpload={uploadDataset} />}
          {page === "Causal Analysis" && <AnalysisPage dataset={dataset} analysis={analysis} mode={analysisMode} busy={busy} runAnalysis={runAnalysis} />}
          {page === "Budget Optimization" && <BudgetPage budget={budget} setBudget={setBudget} customers={customers} setCustomers={setCustomers} optimize={optimize} />}
          {page === "Reports" && <ReportsPage reports={reports} loadReports={loadReports} exportCSV={exportCSV} />}
          <footer className="footer"><span>© 2026 EconoCausal</span><span><span className="status-dot" /> Dynamic Pricing via Double Machine Learning</span></footer>
        </div>
      </main>
      {toast && <div className="toast"><Zap size={17} />{toast}<button onClick={() => setToast("")}><X size={15} /></button></div>}
    </div>
  );
}

function PageIntro({ eyebrow, title, subtitle, action }) {
  return <div className="page-intro"><div><div className="eyebrow"><span />{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>{action}</div>;
}

function Overview({ dataset, analysis, analysisMode, setPage, runAnalysis, busy }) {
  const hasResult = !!analysis;
  const stats = [
    { label: "Customers analysed", value: analysis?.customers_analysed?.toLocaleString() ?? "—", note: "Available after analysis", icon: Database, color: "purple" },
    { label: "Average treatment effect", value: analysis?.ate != null ? `${(analysis.ate * 100).toFixed(1)}%` : "—", note: "Estimated causal lift", icon: Target, color: "green" },
    { label: "Dataset status", value: dataset ? "Selected" : "Not loaded", note: dataset?.name || "Upload a CSV to begin", icon: Activity, color: "blue" },
    { label: "Model status", value: hasResult ? (analysisMode === "backend" ? "Connected" : "Demo result") : "Not run", note: hasResult ? (analysisMode === "backend" ? "Backend response received" : "Illustrative only") : "Run a causal analysis", icon: BrainCircuit, color: "orange" }
  ];
  return <>
    <PageIntro eyebrow="CAUSAL INTELLIGENCE PLATFORM" title={<>Make pricing decisions with <span className="gradient-text">causal evidence.</span></>} subtitle="Measure the incremental impact of offers, understand customer response, and plan campaign spend with Double Machine Learning." action={<div className="live-chip"><span /> Workspace ready</div>} />
    <section className="hero-card"><div className="hero-copy"><div className="hero-tag"><Sparkles size={14} /> ECONOMICS MEETS MACHINE LEARNING</div><h2>Move beyond correlation.<br /><span>Find what actually works.</span></h2><p>Estimate treatment effects while controlling for confounding factors — then turn those insights into smarter campaign decisions.</p><div className="hero-buttons"><button className="btn-primary" onClick={() => setPage("Causal Analysis")}>Run causal analysis <span>→</span></button><button className="btn-ghost" onClick={() => setPage("Upload Dataset")}> <Upload size={16} /> Upload dataset</button></div></div><div className="hero-visual"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="visual-core"><BrainCircuit size={47} /></div><div className="float-card float-top"><span className="float-dot green-dot" /> Causal effect <b>{analysis?.ate != null ? `${(analysis.ate * 100).toFixed(1)}%` : "Awaiting data"}</b></div><div className="float-card float-bottom"><BarChart3 size={17} /><div><small>Decision engine</small><b>Evidence-led pricing</b></div></div></div></section>
    <div className="section-heading"><div><h2>Workspace snapshot</h2><p>Live values appear when the backend returns results.</p></div><span className="period-label"><Activity size={14} /> Current session</span></div>
    <div className="stats-grid">{stats.map((s) => {const Icon=s.icon; return <div className="stat-card" key={s.label}><div className={`stat-icon ${s.color}`}><Icon size={19}/></div><p>{s.label}</p><strong>{s.value}</strong><small>{s.note}</small></div>;})}</div>
    <div className="dashboard-grid"><div className="panel chart-panel"><div className="panel-heading"><div><h3>Offer response illustration</h3><p>Example only — not calculated from your dataset</p></div><span className="demo-badge">ILLUSTRATIVE</span></div><Plot data={[{x:["No offer","Offer received"],y:[35,52],type:"bar",marker:{color:["#c7c9d9","#7c6cf2"],borderRadius:8},text:["35%","52%"],textposition:"outside",hovertemplate:"%{x}<br>Example rate: %{y}%<extra></extra>"}]} layout={{autosize:true,height:270,margin:{l:36,r:20,t:20,b:45},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{family:"Inter, sans-serif",color:"#7c8196",size:11},yaxis:{range:[0,65],ticksuffix:"%",gridcolor:"#eef0f7",zeroline:false},xaxis:{fixedrange:true},showlegend:false}} config={{displayModeBar:false,responsive:true}} style={{width:"100%"}} /></div><div className="panel workflow-panel"><div className="panel-heading"><div><h3>Get started</h3><p>Your analysis workflow</p></div><div className="round-spark"><Zap size={16}/></div></div><div className="workflow-list"><WorkflowStep number="01" title="Upload your dataset" text={dataset ? dataset.name : "Start with a clean CSV file"} done={!!dataset} onClick={() => setPage("Upload Dataset")} /><WorkflowStep number="02" title="Estimate causal effects" text="Choose treatment and outcome variables" done={!!analysis} onClick={() => setPage("Causal Analysis")} /><WorkflowStep number="03" title="Optimize your budget" text="Compare spend and estimated lift" done={false} onClick={() => setPage("Budget Optimization")} /></div><button className="text-link" onClick={() => runAnalysis()} disabled={busy}>{busy ? "Working..." : "Try demo analysis"} <span>→</span></button></div></div>
    {analysisMode === "demo" && analysis && <p className="notice"><CircleHelp size={16}/>{analysis.note || "Demo values are illustrative and must not be treated as real model findings."}</p>}
  </>;
}

function WorkflowStep({ number, title, text, done, onClick }) { return <button className="workflow-step" onClick={onClick}><span className={`step-number ${done?"step-done":""}`}>{done?"✓":number}</span><span className="step-copy"><b>{title}</b><small>{text}</small></span><span className="step-arrow">↗</span></button>; }

function UploadPage({ dataset, uploadState, busy, onUpload }) {
  const [drag, setDrag] = useState(false);
  return <><PageIntro eyebrow="DATA MANAGEMENT" title="Upload your dataset" subtitle="Bring your campaign and customer data into the workspace to prepare it for causal analysis."/><div className="upload-layout"><div className={`upload-zone ${drag?"dragging":""}`} onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);onUpload(e.dataTransfer.files?.[0])}}><div className="upload-icon"><Upload size={26}/></div><h2>Drop your CSV file here</h2><p>or select a file from your computer</p><label className="btn-primary file-picker">{busy?"Processing...":"Choose CSV file"}<input type="file" accept=".csv,text/csv" disabled={busy} onChange={e=>onUpload(e.target.files?.[0])}/></label><div className="file-rules"><span><span className="tiny-check">✓</span> CSV format</span><span><span className="tiny-check">✓</span> Up to 50 MB</span><span><span className="tiny-check">✓</span> Customer-level rows recommended</span></div></div><div className="panel requirements"><div className="panel-heading"><div><h3>Dataset checklist</h3><p>Recommended fields for analysis</p></div><Database size={20} className="muted-icon"/></div>{[["Treatment","e.g. offer_received or discount"],["Outcome","e.g. purchase, revenue, amount_spent"],["Confounders","e.g. age, channel, prior purchases"],["Customer ID","Optional identifier for each customer"]].map(([a,b])=><div className="requirement-row" key={a}><div className="req-check">✓</div><div><b>{a}</b><small>{b}</small></div></div>)}<div className="tip-box"><CircleHelp size={17}/><p>Make sure the treatment and outcome columns are clearly defined. Your backend team should confirm the exact column names supported by the model.</p></div></div></div>{dataset&&<div className="panel file-summary"><div className="file-badge"><FileBarChart size={22}/></div><div className="file-info"><b>{dataset.name}</b><small>{(dataset.size/1024).toFixed(1)} KB · CSV file</small><span>{uploadState?.localOnly?"Selected locally — backend upload not connected":uploadState?.dataset_id?`Dataset ID: ${uploadState.dataset_id}`:"Selected"}</span></div><div className={`upload-status ${uploadState?.localOnly?"status-warn":"status-ok"}`}>{uploadState?.localOnly?"Local only":"Selected"}</div></div>}</>;
}

function AnalysisPage({ dataset, analysis, mode, busy, runAnalysis }) {
  const [treatment,setTreatment]=useState("offer_received");
  const [outcome,setOutcome]=useState("purchase");
  return <><PageIntro eyebrow="CAUSAL INFERENCE" title="Causal analysis" subtitle="Estimate the impact of a treatment while adjusting for observed customer characteristics." action={<div className="method-chip"><BrainCircuit size={16}/> Double Machine Learning</div>}/><div className="analysis-layout"><div className="panel analysis-form"><div className="panel-heading"><div><h3>Analysis configuration</h3><p>Choose the treatment and target outcome.</p></div><FlaskConical size={20} className="muted-icon"/></div><div className="field"><label>Dataset</label><div className="dataset-selected"><Database size={17}/><span>{dataset?.name || "No CSV selected"}</span><small>{dataset?"Ready":"Required for real analysis"}</small></div></div><div className="field"><label htmlFor="treatment">Treatment variable</label><select id="treatment" value={treatment} onChange={e=>setTreatment(e.target.value)}><option value="offer_received">Offer received</option><option value="discount">Discount</option><option value="offer_completed">Offer completed</option></select><small>Variable that represents the intervention.</small></div><div className="field"><label htmlFor="outcome">Outcome variable</label><select id="outcome" value={outcome} onChange={e=>setOutcome(e.target.value)}><option value="purchase">Purchase</option><option value="revenue">Revenue</option><option value="amount_spent">Amount spent</option></select><small>Business outcome you want to measure.</small></div><div className="method-note"><BrainCircuit size={18}/><p><b>DML workflow</b><br/>Estimate nuisance functions, control for observed covariates, and calculate the average treatment effect.</p></div><button className="btn-primary full-width" disabled={busy} onClick={()=>runAnalysis(treatment,outcome)}>{busy?"Running analysis...":<><Zap size={16}/> Run causal analysis</>}</button>{!dataset&&<p className="small-warning">No dataset selected. Without a connected backend, the button will show demo output only.</p>}</div><div className="analysis-results"><div className="panel result-panel"><div className="panel-heading"><div><h3>Analysis results</h3><p>Most recent response</p></div><span className={`result-badge ${analysis?(mode==="backend"?"real-badge":"demo-badge"):"pending-badge"}`}>{analysis?(mode==="backend"?"BACKEND RESULT":"DEMO RESULT"):"AWAITING RUN"}</span></div>{analysis?<><div className="ate-box"><span>Average treatment effect (ATE)</span><strong>{analysis.ate!=null?`${(Number(analysis.ate)*100).toFixed(2)}%`:"Not returned"}</strong><small>{mode==="backend"?"Returned by connected backend":"Illustrative value — not a real estimate"}</small></div><div className="result-metrics"><div><small>Customers analysed</small><b>{analysis.customers_analysed?.toLocaleString?.()??"—"}</b></div><div><small>Individual effects</small><b>{Array.isArray(analysis.ite)?analysis.ite.length:"—"}</b></div></div>{mode==="demo"&&<p className="notice"><CircleHelp size={16}/>{analysis.note}</p>}</>:<div className="empty-state"><div className="empty-icon"><FlaskConical size={25}/></div><h3>No analysis yet</h3><p>Configure your variables and run an analysis. Results will appear here when available.</p></div>}</div><div className="panel explainer"><h3>How to interpret ATE</h3><p>The Average Treatment Effect summarizes the estimated average change in the selected outcome caused by the treatment, under the model assumptions. Its scale depends on how your outcome is encoded.</p></div></div></div></>;
}

function BudgetPage({ budget,setBudget,customers,setCustomers,optimize }) {
  return <><PageIntro eyebrow="CAMPAIGN PLANNING" title="Budget optimization" subtitle="Explore spend scenarios and estimated incremental outcomes. These calculations are planning illustrations, not a solver recommendation." action={<div className="method-chip"><Gauge size={16}/> Scenario planner</div>}/><div className="budget-layout"><div className="panel budget-controls"><div className="panel-heading"><div><h3>Scenario inputs</h3><p>Adjust the plan to compare spending levels.</p></div><Settings2 size={20} className="muted-icon"/></div><div className="field"><label htmlFor="budget">Campaign budget (₹)</label><input id="budget" type="number" min="0" value={budget} onChange={e=>setBudget(e.target.value)}/><input aria-label="Budget slider" className="range" type="range" min="0" max="200000" step="1000" value={Math.min(200000,Math.max(0,Number(budget)||0))} onChange={e=>setBudget(e.target.value)}/><div className="range-labels"><span>₹0</span><span>₹2,00,000</span></div></div><div className="field"><label htmlFor="customers">Target customer count</label><input id="customers" type="number" min="1" value={customers} onChange={e=>setCustomers(e.target.value)}/></div><div className="tip-box"><CircleHelp size={17}/><p>The displayed lift uses the most recent analysis ATE if available; otherwise it uses an illustrative 8.4% assumption.</p></div></div><div className="budget-results"><div className="budget-highlight"><div className="budget-highlight-top"><span>SCENARIO SUMMARY</span><Wallet size={19}/></div><p>Planned campaign budget</p><strong>{optimize.amount.toLocaleString("en-IN",{style:"currency",currency:"INR",maximumFractionDigits:0})}</strong><div className="budget-divider"/><div className="budget-mini"><div><small>Spend per target customer</small><b>{optimize.spendPerCustomer.toLocaleString("en-IN",{style:"currency",currency:"INR",maximumFractionDigits:2})}</b></div><div><small>Illustrative incremental outcomes</small><b>{optimize.estimatedIncrementalConversions.toLocaleString("en-IN")}</b></div></div></div><div className="panel"><h3>What this scenario means</h3><p className="body-copy">This simple scenario multiplies the target audience by the selected treatment-effect estimate. It does not allocate budget across segments or solve a constrained optimization problem.</p><div className="scenario-callout"><Target size={19}/><div><b>Next step for the full model</b><p>Backend should use individual treatment effects, segment costs, and campaign constraints to recommend an allocation.</p></div></div></div></div></div></>;
}

function ReportsPage({ reports,loadReports,exportCSV }) {
  return <><PageIntro eyebrow="EXPORT & GOVERNANCE" title="Reports" subtitle="Retrieve available analysis results and export a CSV summary for your project documentation." action={<button className="btn-primary" onClick={loadReports}><Activity size={16}/> Load latest report</button>}/><div className="panel reports-panel"><div className="panel-heading"><div><h3>Analysis report</h3><p>Report data is loaded from the backend when the endpoint is available.</p></div><div className="report-icon"><FileBarChart size={21}/></div></div>{reports?<><div className="report-grid">{[["Average treatment effect",reports.ate==null?"—":`${(Number(reports.ate)*100).toFixed(2)}%`],["Customers analysed",reports.customers_analysed?.toLocaleString?.()??"—"],["Revenue impact",reports.revenue_impact==null?"—":String(reports.revenue_impact)],["Budget utilization",reports.budget_utilization==null?"—":String(reports.budget_utilization)]].map(([k,v])=><div className="report-value" key={k}><small>{k}</small><b>{v}</b></div>)}</div>{reports.demo&&<p className="notice"><CircleHelp size={16}/>Some fields are missing because the backend report endpoint is not connected or has no results yet.</p>}<button className="btn-secondary" onClick={exportCSV}><Download size={16}/> Export report as CSV</button></>:<div className="empty-state"><div className="empty-icon"><FileBarChart size={25}/></div><h3>Your report will appear here</h3><p>Load the latest report after running an analysis. Empty fields are intentionally not filled with fabricated metrics.</p><button className="btn-secondary" onClick={loadReports}>Load latest report</button></div>}</div><div className="panel report-note"><CircleHelp size={18}/><p><b>Research integrity:</b> clearly label demo values, document treatment/outcome definitions, and report uncertainty intervals when the backend model provides them.</p></div></>;
}

export default App;
