import { useState } from "react";
import { AlertTriangle, ArrowRight, CheckCircle2, FlaskConical, Info, LoaderCircle, Play, ShieldCheck, SlidersHorizontal, Sparkles } from "lucide-react";
import { api } from "../services/api";
import Panel from "../components/Panel";

export default function AnalysisLab({ dataset, onNotice, onNavigate, setAnalysis }) {
  const cols=dataset?.headers||[];
  const [treatment,setTreatment]=useState(dataset?.mapping?.treatment||"");
  const [outcome,setOutcome]=useState(dataset?.mapping?.outcome||"");
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
      setResult(response);setAnalysis(response);setStatus("success");onNotice("Backend analysis completed.");
    }catch(e){setError(e.message);setStatus("error");}
  };
  const demo=()=>{setResult({demo:true,average_treatment_effect:0.084,confidence_interval:[0.041,0.127],message:"Illustrative UI values; not model-generated."});setStatus("demo");setError("");setAnalysis({demo:true,average_treatment_effect:0.084,confidence_interval:[0.041,0.127],treatment,outcome});};
  return <div className="page-stack">
    <section className="page-intro"><div><span className="section-eyebrow">CAUSAL INFERENCE / 02</span><h2>Analysis Lab</h2><p>Configure an estimation request for the backend Double ML model.</p></div><span className="step-chip">STEP 02 <span>/ 03</span></span></section>
    <div className="analysis-layout"><Panel title="Model configuration" subtitle="Set the variables and estimation method."><div className="model-ready"><div className="model-ready-icon"><FlaskConical size={19}/></div><div><b>{dataset?.id?"Backend dataset linked":dataset?"Local CSV selected":"No dataset configured"}</b><small>{dataset?.id?"A dataset ID is available for API requests.":dataset?"Upload this file from Dataset Studio to run real analysis.":"Start by uploading your CSV."}</small></div><span className={`tiny-status ${dataset?.id?"connected":""}`}>{dataset?.id?"LINKED":"PENDING"}</span></div><label className="mapping-label">TREATMENT VARIABLE</label><select className="form-select" value={treatment} onChange={e=>setTreatment(e.target.value)}><option value="">Select treatment</option>{cols.map(c=><option key={c}>{c}</option>)}</select><label className="mapping-label">OUTCOME VARIABLE</label><select className="form-select" value={outcome} onChange={e=>setOutcome(e.target.value)}><option value="">Select outcome</option>{cols.map(c=><option key={c}>{c}</option>)}</select><label className="mapping-label">COVARIATES / CONTROLS</label><div className="analysis-covariates">{cols.filter(c=>c!==treatment&&c!==outcome).map(c=><label key={c}><input type="checkbox" checked={covariates.includes(c)} onChange={()=>setCovariates(old=>old.includes(c)?old.filter(v=>v!==c):[...old,c])}/>{c}</label>)}</div><label className="mapping-label">ESTIMATION METHOD</label><div className="method-card"><div className="method-icon"><Sparkles size={18}/></div><div><b>Double Machine Learning</b><p>Estimate a treatment effect while adjusting for observed covariates.</p></div><input type="radio" checked={method==="double_ml"} onChange={()=>setMethod("double_ml")}/></div><div className="mapping-tip"><ShieldCheck size={16}/><span>Real estimates require the backend to validate inputs, fit nuisance models and return diagnostics.</span></div><button className="btn-primary full-btn" disabled={status==="loading"} onClick={run}>{status==="loading"?<><LoaderCircle className="spin" size={16}/> Running model…</>:<><Play size={15}/> Run backend analysis</>}</button><button className="btn-outline full-btn" onClick={demo}>Preview illustrative demo result</button>{error&&<div className="inline-error"><AlertTriangle size={16}/>{error}</div>}</Panel>
    <Panel title="Estimation results" subtitle="Backend output will be displayed here." badge={status==="success"?"BACKEND RESULT":status==="demo"?"DEMO RESULT":"AWAITING RUN"}><div className="result-topline"><div><span className="result-label">AVERAGE TREATMENT EFFECT</span><strong>{status==="success"?(result?.average_treatment_effect!=null?formatEffect(result.average_treatment_effect):"Returned") : status==="demo"?"+8.4%":"—"}</strong><small>{status==="success"?"Returned by the connected backend":status==="demo"?"Illustrative only — not model-generated":"No analysis has run yet"}</small></div><div className={`result-icon ${status==="success"?"ok":""}`}><FlaskConical size={23}/></div></div>{status==="idle"&&<div className="result-placeholder"><div><SlidersHorizontal size={23}/></div><b>Configure your first run</b><p>Choose treatment and outcome columns, then run the connected model or preview the demo.</p></div>}{(status==="success"||status==="demo")&&<><div className="result-stat-row"><div><span>Confidence interval</span><b>{status==="demo"?"[+4.1%, +12.7%]":formatCI(result?.confidence_interval)}</b></div><div><span>Method</span><b>Double ML</b></div><div><span>Dataset</span><b>{dataset?.name||"—"}</b></div></div><div className={`interpretation-box ${status==="success"?"real-result":""}`}><Info size={17}/><p>{status==="success"?(result?.interpretation||result?.message||"Analysis response received from backend. Review the returned diagnostics before making decisions."):"This is a visual demo result only. It was not calculated from your CSV and must not be used for business decisions."}</p></div>{status==="success"&&<div className="backend-success"><CheckCircle2 size={15}/> API response received successfully.</div>}</>}</Panel></div>
    <p className="disclaimer">API contract is a frontend integration template. Match request fields and response parsing in <code>src/services/api.js</code> to the backend team's actual implementation.</p>
  </div>;
}
function formatEffect(v){const n=Number(v);return Number.isFinite(n)?`${(n*100).toFixed(2)}%`:String(v);}
function formatCI(v){return Array.isArray(v)?`[${v.map(x=>`${(Number(x)*100).toFixed(2)}%`).join(", ")}]`:"Not provided";}
