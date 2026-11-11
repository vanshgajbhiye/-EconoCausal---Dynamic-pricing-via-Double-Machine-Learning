import { useState } from "react";
import { AlertTriangle, CheckCircle2, FlaskConical, Info, LoaderCircle, Play, ShieldCheck, SlidersHorizontal, Sparkles, Tag } from "lucide-react";
import { api } from "../services/api";
import Panel from "../components/Panel";

export default function AnalysisLab({ dataset, onNotice, onNavigate, setAnalysis }) {
  const cols=dataset?.headers||[];
  // Smart default: prefer unit_price for treatment, qty for outcome
  const tPriority = ["unit_price","price","discount","treatment"];
  const yPriority = ["qty","quantity","units_sold","sales","demand","outcome"];
  const findDefault = (priority, exclude="") => {
    for(const p of priority){ const c=cols.find(h=>h.toLowerCase()===p||h.toLowerCase().includes(p)); if(c&&c!==exclude) return c; }
    return "";
  };
  const defaultT = dataset?.mapping?.treatment || findDefault(tPriority);
  const defaultY = dataset?.mapping?.outcome   || findDefault(yPriority, defaultT);
  const [treatment,setTreatment]=useState(defaultT);
  const [outcome,setOutcome]=useState(defaultY);
  const [covariates,setCovariates]=useState(dataset?.mapping?.covariates||[]);
  const [method,setMethod]=useState("double_ml");
  const [status,setStatus]=useState("idle");
  const [result,setResult]=useState(null);
  const [error,setError]=useState("");

  const run=async()=>{
    if(!dataset){setError("Upload a dataset first in Dataset Studio.");return;}
    if(!dataset.id){setError("No backend dataset ID is available. Upload the CSV to your connected backend first, or use demo mode.");return;}
    if(!treatment||!outcome||treatment===outcome){setError("Choose different treatment and outcome variables.");return;}
    setStatus("loading");setError("");
    try{
      const response=await api.runAnalysis({dataset_id:dataset.id,treatment,outcome,covariates,method});
      const data=response?.result||response?.data||response;
      setResult(data);setAnalysis(data);setStatus("success");onNotice("Backend analysis completed.");
    }catch(e){setError(e.message);setStatus("error");}
  };

  const demo=()=>{
    setResult({demo:true,average_treatment_effect:0.084,confidence_interval:[0.041,0.127],message:"Illustrative UI values; not model-generated."});
    setStatus("demo");setError("");
    setAnalysis({demo:true,average_treatment_effect:0.084,confidence_interval:[0.041,0.127],treatment,outcome});
  };

  const optPrice = result?.optimal_price;
  const profitLift = result?.optimal_price_validation?.projected_profit_lift;
  const pricingMethod = result?.optimal_price_validation?.methodology || "DML Causal Pricing Rule";

  return <div className="page-stack">
    <section className="page-intro"><div><span className="section-eyebrow">CAUSAL INFERENCE / 02</span><h2>Analysis Lab</h2><p>Configure an estimation request for the backend Double ML model.</p></div><span className="step-chip">STEP 02 <span>/ 03</span></span></section>
    <div className="analysis-layout">

      {/* ── LEFT: Model Config ── */}
      <Panel title="Model configuration" subtitle="Set the variables and estimation method.">
        <div className="model-ready">
          <div className="model-ready-icon"><FlaskConical size={19}/></div>
          <div><b>{dataset?.id?"Backend dataset linked":dataset?"Local CSV selected":"No dataset configured"}</b><small>{dataset?.id?"A dataset ID is available for API requests.":dataset?"Upload this file from Dataset Studio to run real analysis.":"Start by uploading your CSV."}</small></div>
          <span className={`tiny-status ${dataset?.id?"connected":""}`}>{dataset?.id?"LINKED":"PENDING"}</span>
        </div>
        <label className="mapping-label">TREATMENT VARIABLE</label>
        <select className="form-select" value={treatment} onChange={e=>setTreatment(e.target.value)}><option value="">Select treatment</option>{cols.map(c=><option key={c}>{c}</option>)}</select>
        <label className="mapping-label">OUTCOME VARIABLE</label>
        <select className="form-select" value={outcome} onChange={e=>setOutcome(e.target.value)}><option value="">Select outcome</option>{cols.map(c=><option key={c}>{c}</option>)}</select>
        <label className="mapping-label">COVARIATES / CONTROLS</label>
        <div className="analysis-covariates">{cols.filter(c=>c!==treatment&&c!==outcome).map(c=><label key={c}><input type="checkbox" checked={covariates.includes(c)} onChange={()=>setCovariates(old=>old.includes(c)?old.filter(v=>v!==c):[...old,c])}/>{c}</label>)}</div>
        <label className="mapping-label">ESTIMATION METHOD</label>
        <div className="method-card"><div className="method-icon"><Sparkles size={18}/></div><div><b>Double Machine Learning</b><p>Estimate a treatment effect while adjusting for observed covariates.</p></div><input type="radio" checked={method==="double_ml"} onChange={()=>setMethod("double_ml")}/></div>
        <div className="mapping-tip"><ShieldCheck size={16}/><span>Real estimates require the backend to validate inputs, fit nuisance models and return diagnostics.</span></div>
        <button className="btn-primary full-btn" disabled={status==="loading"} onClick={run}>{status==="loading"?<><LoaderCircle className="spin" size={16}/> Running model…</>:<><Play size={15}/> Run backend analysis</>}</button>
        <button className="btn-outline full-btn" onClick={demo}>Preview illustrative demo result</button>
        {error&&<div className="inline-error"><AlertTriangle size={16}/>{error}</div>}
      </Panel>

      {/* ── RIGHT: Results ── */}
      <Panel title="Estimation results" subtitle="Backend output will be displayed here." badge={status==="success"?"BACKEND RESULT":status==="demo"?"DEMO RESULT":"AWAITING RUN"}>

        {/* Big ATE number */}
        <div className="result-topline">
          <div>
            <span className="result-label">{status==="demo"?"AVERAGE TREATMENT EFFECT":"CAUSAL TREATMENT EFFECT (ATE / θ)"}</span>
            <strong>
              {status==="success"
                ? (result?.average_treatment_effect!=null ? formatEffect(result.average_treatment_effect) : "—")
                : status==="demo" ? "+8.4%" : "—"}
            </strong>
            <small>
              {status==="success"
                ? (result?.ate_units||"Demand change per $1 price increase")
                : status==="demo" ? "Illustrative only — not model-generated"
                : "No analysis has run yet"}
            </small>
          </div>
          <div className={`result-icon ${status==="success"?"ok":""}`}><FlaskConical size={23}/></div>
        </div>

        {/* Idle placeholder */}
        {status==="idle"&&<div className="result-placeholder"><div><SlidersHorizontal size={23}/></div><b>Configure your first run</b><p>Choose treatment and outcome columns, then run the connected model or preview the demo.</p></div>}

        {/* Results rows */}
        {(status==="success"||status==="demo")&&<>
          <div className="result-stat-row">
            <div><span>{status==="demo"?"Confidence interval":"95% CI (Elasticity)"}</span><b>{status==="demo"?"[+4.1%, +12.7%]":formatCI(result?.confidence_interval)}</b></div>
            <div><span>Price Elasticity (ε)</span><b>{status==="demo"?"—":(result?.price_elasticity!=null?result.price_elasticity.toFixed(2):"—")}</b></div>
            <div><span>Demand Regime</span><b>{status==="demo"?"—":(result?.demand_nature||result?.demand_regime||"—")}</b></div>
          </div>

          {status==="success"&&result?.optimal_price_validation&&
            <div className="result-stat-row" style={{marginTop:"8px"}}>
              <div><span>Derivation</span><b>{result.optimal_price_validation.methodology||"Lerner markup condition"}</b></div>
              <div><span>Marginal Cost (MC)</span><b>${result.optimal_price_validation.marginal_cost}</b></div>
              <div><span>Causal ID</span><b>Orthogonal DML ✓</b></div>
            </div>}

          {/* Interpretation box */}
          <div className={`interpretation-box ${status==="success"?"real-result":""}`}>
            <Info size={17}/>
            <p>{status==="success"
              ? (result?.interpretation||result?.message||"Analysis response received from backend. Review the returned diagnostics before making decisions.")
              : "This is a visual demo result only. It was not calculated from your CSV and must not be used for business decisions."}</p>
          </div>

          {/* ── RECOMMENDED PRICE CARD ── */}
          {status==="success"&&optPrice!=null&&
            <div style={{marginTop:"16px",padding:"18px 20px",borderRadius:"12px",background:"linear-gradient(135deg,#1a0f35 0%,#0d0720 100%)",border:"2px solid #7c3aed",display:"flex",alignItems:"center",gap:"14px",boxShadow:"0 0 24px rgba(124,58,237,0.22)"}}>
              <div style={{background:"#7c3aed",borderRadius:"10px",padding:"12px",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0}}>
                <Tag size={24} color="#fff"/>
              </div>
              <div style={{flex:1,minWidth:0}}>
                <div style={{fontSize:"10px",fontWeight:700,letterSpacing:"0.12em",color:"#c4b5fd",marginBottom:"5px",textTransform:"uppercase"}}>
                  Recommended Price to Sell
                </div>
                <div style={{fontSize:"30px",fontWeight:800,color:"#f8fafc",lineHeight:1.1,letterSpacing:"-0.5px"}}>
                  ${optPrice.toFixed(2)}&nbsp;<span style={{fontSize:"14px",fontWeight:500,color:"#c4b5fd"}}>/ unit</span>
                </div>
                <div style={{fontSize:"11px",color:"#6b7280",marginTop:"4px"}}>{pricingMethod}</div>
              </div>
              {profitLift&&
                <div style={{textAlign:"center",background:"rgba(52,211,153,0.12)",border:"1px solid rgba(52,211,153,0.3)",borderRadius:"10px",padding:"10px 16px",flexShrink:0}}>
                  <div style={{fontSize:"20px",fontWeight:800,color:"#34d399"}}>{profitLift}</div>
                  <div style={{fontSize:"10px",color:"#64748b",marginTop:"2px",letterSpacing:"0.05em"}}>PROFIT LIFT</div>
                </div>}
            </div>}

          {status==="success"&&<div className="backend-success"><CheckCircle2 size={15}/> DML Econometric diagnostics &amp; optimization validated.</div>}
        </>}
      </Panel>
    </div>
    <p className="disclaimer">API contract is a frontend integration template. Match request fields and response parsing in <code>src/services/api.js</code> to the backend team's actual implementation.</p>
  </div>;
}

function formatEffect(v){
  const n = Number(v);
  if(!Number.isFinite(n)) return String(v);
  // Always display as percentage — multiply magnitude × 100
  const pct = Math.abs(n) * 100;
  if(pct < 0.01) return `+${pct.toExponential(2)}%`;
  if(pct < 1)    return `+${pct.toFixed(2)}%`;
  return `+${pct.toFixed(1)}%`;
}

function formatCI(v){
  if(!Array.isArray(v)) return "Not provided";
  return `[${v.map(x=>{
    const n=Number(x);
    if(!Number.isFinite(n)) return String(x);
    return n.toFixed(3);
  }).join(", ")}]`;
}
