import { useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, Check, CheckCircle2, ChevronDown, CloudUpload, Database, FileSpreadsheet, Info, LoaderCircle, RefreshCw, Send, Trash2, Upload } from "lucide-react";
import { api } from "../services/api";
import Panel from "../components/Panel";

function parseCSVLine(line) {
  const out=[]; let cur="", quoted=false;
  for(let i=0;i<line.length;i++){const c=line[i]; if(c==='"'&&line[i+1]==='"'&&quoted){cur+='"';i++;}else if(c==='"'){quoted=!quoted;}else if(c===","&&!quoted){out.push(cur.trim());cur="";}else cur+=c;}
  out.push(cur.trim()); return out;
}
export default function DatasetStudio({ dataset, setDataset, onNotice, onNavigate }) {
  const inputRef=useRef(null);
  const [file,setFile]=useState(dataset?.file||null);
  const [headers,setHeaders]=useState(dataset?.headers||[]);
  const [preview,setPreview]=useState(dataset?.preview||[]);
  const [mapping,setMapping]=useState(dataset?.mapping||{treatment:"",outcome:"",covariates:[]});
  const [uploading,setUploading]=useState(false);
  const [error,setError]=useState("");
  const [drag,setDrag]=useState(false);
  const [backendId,setBackendId]=useState(dataset?.id||null);
  const [backendStatus,setBackendStatus]=useState(dataset?.backendStatus||"local");

  useEffect(()=>{ if(dataset?.file){setFile(dataset.file);setHeaders(dataset.headers||[]);setPreview(dataset.preview||[]);setMapping(dataset.mapping||{treatment:"",outcome:"",covariates:[]});setBackendId(dataset.id||null);setBackendStatus(dataset.backendStatus||"local");}},[dataset]);

  const readFile=async f=>{
    setError(""); if(!f)return;
    if(!f.name.toLowerCase().endsWith(".csv")){setError("Please select a .csv file.");return;}
    if(f.size>50*1024*1024){setError("File is larger than 50 MB. Please choose a smaller CSV or adjust the backend upload limit.");return;}
    setFile(f);setBackendId(null);setBackendStatus("local");
    try{
      const text=await f.text(); const lines=text.split(/\r?\n/).filter(line=>line.trim()).slice(0,7);
      if(!lines.length){setError("The CSV file appears to be empty.");return;}
      const cols=parseCSVLine(lines[0]); setHeaders(cols);
      setPreview(lines.slice(1).map(parseCSVLine));
      const lower=cols.map(c=>c.toLowerCase());
      const t=lower.find(c=>["price","discount","promotion","treatment"].includes(c));
      const y=lower.find(c=>["units_sold","sales","revenue","conversion_rate","outcome"].includes(c));
      setMapping({treatment:cols[lower.indexOf(t)]||cols[0]||"",outcome:cols[lower.indexOf(y)]||cols[Math.min(1,cols.length-1)]||"",covariates:[]});
      setDataset({file:f,name:f.name,size:f.size,headers:cols,preview:lines.slice(1).map(parseCSVLine),mapping:{treatment:cols[lower.indexOf(t)]||cols[0]||"",outcome:cols[lower.indexOf(y)]||cols[Math.min(1,cols.length-1)]||"",covariates:[]},backendStatus:"local"});
    }catch(e){setError("Could not read this CSV file. Please check the file encoding and format.");}
  };
  const changeMapping=(key,value)=>{const next={...mapping,[key]:value};setMapping(next);if(file)setDataset({...dataset,file,name:file.name,size:file.size,headers,preview,mapping:next,backendStatus});};
  const toggleCovariate=col=>{const list=mapping.covariates||[];changeMapping("covariates",list.includes(col)?list.filter(x=>x!==col):[...list,col]);};
  const uploadToBackend=async()=>{
    if(!file){setError("Choose a CSV file first.");return;}
    if(!mapping.treatment||!mapping.outcome||mapping.treatment===mapping.outcome){setError("Select different treatment and outcome columns.");return;}
    setUploading(true);setError("");
    try{
      const result=await api.uploadDataset(file,mapping);
      const id=result.dataset_id||result.id||result.dataset?.id;
      setBackendId(id||null);setBackendStatus("uploaded");
      const next={file,name:file.name,size:file.size,headers,preview,mapping,id,backendStatus:"uploaded",backendResponse:result};
      setDataset(next);onNotice("Dataset uploaded to backend successfully.");
    }catch(e){setError(e.message);setBackendStatus("local");}
    finally{setUploading(false);}
  };
  const remove=()=>{setFile(null);setHeaders([]);setPreview([]);setBackendId(null);setBackendStatus("local");setMapping({treatment:"",outcome:"",covariates:[]});setDataset(null);setError("");if(inputRef.current)inputRef.current.value="";};
  return <div className="page-stack">
    <section className="page-intro"><div><span className="section-eyebrow">DATA PREPARATION / 01</span><h2>Dataset Studio</h2><p>Inspect your CSV, map the causal variables and upload it when the backend is connected.</p></div><span className="step-chip">STEP 01 <span>/ 03</span></span></section>
    <div className="dataset-layout"><div className="dataset-main">
      <section className={`drop-card ${drag?"dragging":""}`} onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);readFile(e.dataTransfer.files?.[0]);}}>
        <input ref={inputRef} type="file" accept=".csv,text/csv" hidden onChange={e=>readFile(e.target.files?.[0])}/>
        <div className="drop-art"><CloudUpload size={27}/><span>CSV</span></div><h3>{file?file.name:"Drop your dataset into the workspace"}</h3><p>{file?`${(file.size/1024).toFixed(1)} KB · parsed in your browser`:"CSV files only · up to 50 MB · preview is processed locally"}</p><button className="btn-primary" onClick={()=>inputRef.current?.click()}><Upload size={16}/>{file?"Choose another file":"Browse CSV files"}</button><div className="drop-security"><span><Check size={13}/> Local preview</span><span><Check size={13}/> Column mapping</span><span><Info size={13}/> Upload requires backend</span></div>
      </section>
      {error&&<div className="inline-error"><AlertCircle size={17}/><span>{error}</span></div>}
      {file&&headers.length>0&&<Panel title="Data preview" subtitle={`${headers.length} columns · first ${preview.length} rows shown`} badge="LOCAL PREVIEW"><div className="table-scroll"><table className="data-table"><thead><tr>{headers.map((h,i)=><th key={i}>{h||`Column ${i+1}`}</th>)}</tr></thead><tbody>{preview.length?preview.map((row,i)=><tr key={i}>{headers.map((_,j)=><td key={j}>{row[j]??"—"}</td>)}</tr>):<tr><td colSpan={headers.length}>Header row detected. No data rows found in the preview.</td></tr>}</tbody></table></div><div className="table-note"><Info size={14}/> Preview is read locally in your browser; it is not sent anywhere until you press upload.</div></Panel>}
      {!file&&<Panel title="What your dataset needs" subtitle="You can map columns from your own naming convention."><div className="requirements-grid"><div><b>01 <span>Treatment</span></b><p>Price, discount, promotion or another variable whose effect you want to estimate.</p></div><div><b>02 <span>Outcome</span></b><p>Units sold, revenue, conversion or the business result you want to measure.</p></div><div><b>03 <span>Covariates</span></b><p>Customer, product, seasonality and other confounders relevant to your study.</p></div></div></Panel>}
    </div>
    <aside className="mapping-column"><Panel title="Variable mapping" subtitle="Assign the roles for your causal model."><div className="mapping-status"><span className={file?"status-check":"status-pending"}>{file?<CheckCircle2 size={15}/>:<Database size={15}/>}</span><div><b>{file?"Dataset ready to configure":"Waiting for a CSV"}</b><small>{file?"Review your variable roles below":"Upload a file to see available columns"}</small></div></div><label className="mapping-label">TREATMENT VARIABLE <span>Required</span></label><div className="select-wrap"><select value={mapping.treatment} disabled={!headers.length} onChange={e=>changeMapping("treatment",e.target.value)}><option value="">Select a column</option>{headers.map(h=><option key={h} value={h}>{h}</option>)}</select><ChevronDown size={15}/></div><label className="mapping-label">OUTCOME VARIABLE <span>Required</span></label><div className="select-wrap"><select value={mapping.outcome} disabled={!headers.length} onChange={e=>changeMapping("outcome",e.target.value)}><option value="">Select a column</option>{headers.map(h=><option key={h} value={h}>{h}</option>)}</select><ChevronDown size={15}/></div><label className="mapping-label">COVARIATES <span>Optional</span></label><div className="covariate-list">{headers.filter(h=>h!==mapping.treatment&&h!==mapping.outcome).map(h=><label key={h} className="covariate-option"><input type="checkbox" checked={(mapping.covariates||[]).includes(h)} onChange={()=>toggleCovariate(h)}/><span>{h}</span></label>)}{!headers.length&&<p className="muted-help">Available columns appear here.</p>}</div><div className="mapping-tip"><Info size={15}/><span>Choose distinct treatment and outcome variables. The backend should validate types and missing values too.</span></div><button className="btn-primary full-btn" disabled={!file||uploading||!mapping.treatment||!mapping.outcome||mapping.treatment===mapping.outcome} onClick={uploadToBackend}>{uploading?<><LoaderCircle className="spin" size={16}/> Uploading…</>:<><Send size={15}/> Upload to backend</>}</button><button className="btn-outline full-btn" disabled={!file} onClick={()=>{if(file){setDataset({...dataset,file,name:file.name,size:file.size,headers,preview,mapping,backendStatus:"local"});onNotice("Dataset and mapping saved in this session.");onNavigate("Analysis Lab");}}}>Continue in demo mode <span>→</span></button>{backendStatus==="uploaded"&&<div className="backend-success"><CheckCircle2 size={15}/> Backend accepted dataset {backendId?`(${backendId})`:""}.</div>}</Panel><div className="privacy-note"><span className="privacy-icon"><Info size={16}/></span><p><b>Data handling</b><br/>Local preview stays in your browser. Clicking “Upload to backend” sends the CSV and selected mapping to the configured API.</p></div></aside></div>
  </div>;
}