export default function Panel({ title, subtitle, badge, children, className="" }) {
  return <section className={`panel ${className}`}><div className="panel-head"><div><h2>{title}</h2>{subtitle&&<p>{subtitle}</p>}</div>{badge&&<span className="panel-badge">{badge}</span>}</div>{children}</section>;
}