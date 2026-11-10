import { useState } from "react";
import { X } from "lucide-react";
import Sidebar from "./components/Sidebar.jsx";
import Topbar from "./components/Topbar.jsx";
import Overview from "./pages/Overview.jsx";
import DatasetStudio from "./pages/DatasetStudio.jsx";
import AnalysisLab from "./pages/AnalysisLab.jsx";
import BudgetPlanner from "./pages/BudgetPlanner.jsx";
import Reports from "./pages/Reports.jsx";
import "./styles/app.css";

export default function App(){
  const [page,setPage]=useState("Overview");
  const [menuOpen,setMenuOpen]=useState(false);
  const [dataset,setDataset]=useState(null);
  const [analysis,setAnalysis]=useState(null);
  const [notice,setNotice]=useState("");
  const onNotice=msg=>{setNotice(msg);window.clearTimeout(window.__ecToast);window.__ecToast=window.setTimeout(()=>setNotice(""),4000);};
  const props={dataset,setDataset,analysis,setAnalysis,onNotice,onNavigate:setPage};
  return <div className="app-root"><Sidebar page={page} setPage={setPage} open={menuOpen} close={()=>setMenuOpen(false)}/><div className="main-shell"><Topbar page={page} onMenu={()=>setMenuOpen(true)} onNotice={onNotice}/><main className="content-area">{page==="Overview"&&<Overview {...props}/ >}{page==="Dataset Studio"&&<DatasetStudio {...props}/ >}{page==="Analysis Lab"&&<AnalysisLab {...props}/ >}{page==="Budget Planner"&&<BudgetPlanner {...props}/ >}{page==="Reports & Exports"&&<Reports {...props}/ >}</main><footer className="app-footer"><span>ECONOCAUSAL <i/> CAUSAL PRICING INTELLIGENCE</span><span>Frontend prototype · Backend integration ready</span></footer></div>{notice&&<div className="toast"><span className="toast-led"/>{notice}<button onClick={()=>setNotice("")} aria-label="Dismiss"><X size={15}/></button></div>}</div>;
}