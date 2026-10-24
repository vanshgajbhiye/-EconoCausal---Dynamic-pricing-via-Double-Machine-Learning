import { Bell, Menu, Search, ChevronDown, Sparkles } from "lucide-react";
export default function Topbar({ page, onMenu, onNotice }) {
  const subtitle = page==="Overview" ? "Your causal pricing workspace, at a glance." : ({
    "Dataset Studio":"Prepare and map your data before analysis.",
    "Analysis Lab":"Configure your treatment, outcome and controls.",
    "Budget Planner":"Explore budget scenarios and allocation assumptions.",
    "Reports & Exports":"Review analysis status and export results."
  }[page] || "Your causal pricing workspace.");
  return <header className="topbar"><div className="topbar-left"><button className="menu-button" onClick={onMenu}><Menu size={20}/></button><div><div className="crumb">Workspace <span>/</span> <b>{page}</b></div><h1>{page}</h1><p>{subtitle}</p></div></div><div className="topbar-right"><label className="search-field"><Search size={16}/><input placeholder="Search workspace"/></label><button className="icon-action" onClick={()=>onNotice("You're all caught up.")}><Bell size={18}/><i/></button><div className="topbar-separator"/><div className="top-user"><div className="user-initials">VG</div><div><b>Vansh</b><small>Analyst</small></div><ChevronDown size={15}/></div></div></header>;
}