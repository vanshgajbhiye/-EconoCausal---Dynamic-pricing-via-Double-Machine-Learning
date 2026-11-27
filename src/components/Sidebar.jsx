import { Activity, BarChart3, Boxes, CircleHelp, Database, FileChartColumn, FlaskConical, Gauge, LayoutDashboard, Settings, Upload, Wallet, X, BrainCircuit } from "lucide-react";

const groups = [
  { title: "WORKSPACE", items: [{ label: "Overview", icon: LayoutDashboard }, { label: "Dataset Studio", icon: Database }, { label: "Analysis Lab", icon: FlaskConical }] },
  { title: "INTELLIGENCE", items: [{ label: "Budget Planner", icon: Wallet }, { label: "Reports & Exports", icon: FileChartColumn }] }
];
export default function Sidebar({ page, setPage, open, close }) {
  return <>
    <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
      <div className="brand"><div className="brand-symbol"><BrainCircuit size={21}/></div><div><strong>Econo<span>Causal</span></strong><small>CAUSAL PRICING OS</small></div><button className="close-nav" onClick={close}><X size={18}/></button></div>
      <div className="workspace-switch"><div className="workspace-glyph">EC</div><div><b>Research workspace</b><small>Dynamic pricing project</small></div><span className="switch-arrow">⌄</span></div>
      <nav>{groups.map(group=><div className="nav-group" key={group.title}><div className="nav-caption">{group.title}</div>{group.items.map(item=><button key={item.label} className={`nav-link ${page===item.label?"selected":""}`} onClick={()=>{setPage(item.label);close();}}><item.icon size={18}/><span>{item.label}</span>{item.label==="Dataset Studio"&&<i className="nav-indicator"/>}</button>)}</div>)}</nav>
      <div className="sidebar-spacer"/>
      <div className="sidebar-bottom-card"><div className="help-badge"><CircleHelp size={18}/></div><b>Build with confidence</b><p>Upload data, map variables, then estimate causal effects.</p><button onClick={()=>setPage("Reports & Exports")}>Open project guide <span>↗</span></button></div>
      <button className="nav-link settings-link"><Settings size={18}/><span>Settings</span></button>
      <div className="sidebar-user"><div className="user-initials">VG</div><div><b>Project contributor</b><small>Frontend workspace</small></div><span className="user-more">•••</span></div>
    </aside>
    {open&&<button className="sidebar-scrim" onClick={close} aria-label="Close menu"/>}
  </>;
}