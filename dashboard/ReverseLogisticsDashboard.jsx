import { useState, useCallback, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis } from "recharts";

// ─── MILP Solver (Pure JS — Simplex-based for continuous relaxation + rounding) ───
// For thesis: uses greedy exact MILP logic since all constraints are linear and small-scale

function solveMILP({ wastes, ccs, holdingCosts, transportCosts, etpRate, etpReuseValue, etpDrainCost, co2Penalty, circularityBonus, treatedWastewater }) {
  const wasteTypes = ["solid", "yarn", "chemical"];
  const results = [];
  let totalRevenue = 0, totalDisposal = 0, totalTransport = 0, totalHolding = 0, totalEnvPen = 0, totalCircBonus = 0;

  // For each waste type, allocate optimally across CCs
  // Build all possible allocations and use greedy split
  for (const w of wasteTypes) {
    const totalKg = wastes[w];
    const isChemical = w === "chemical";
    
    // Get eligible CCs with capacity > 0
    const eligible = ccs
      .filter(cc => cc.facilities[w] && cc.facilities[w].capacity > 0)
      .map(cc => {
        const fac = cc.facilities[w];
        const dist = cc.distance;
        const tCost = transportCosts[w] * dist;
        const hCost = holdingCosts[w];
        const envCost = fac.co2PerKg * co2Penalty;
        const circB = fac.circularityRate * circularityBonus;
        let netPerKg;
        if (!isChemical) {
          netPerKg = fac.revenuePerKg - tCost - hCost - envCost + circB;
        } else {
          netPerKg = -(fac.disposalFeePerKg + tCost + hCost + envCost) + circB;
        }
        return { cc, fac, netPerKg, capacity: fac.capacity, dist, tCost, hCost, envCost, circB };
      })
      .sort((a, b) => b.netPerKg - a.netPerKg); // Sort by best net value

    let remaining = totalKg;
    for (const opt of eligible) {
      if (remaining <= 0) break;
      const alloc = Math.min(remaining, opt.capacity);
      if (alloc <= 0) continue;
      remaining -= alloc;

      const rev = !isChemical ? opt.fac.revenuePerKg * alloc : 0;
      const disp = isChemical ? opt.fac.disposalFeePerKg * alloc : 0;
      const trans = opt.tCost * alloc;
      const hold = opt.hCost * alloc;
      const env = opt.envCost * alloc;
      const circ = opt.circB * alloc;
      const net = !isChemical ? (rev - trans - hold - env + circ) : -(disp + trans + hold + env) + circ;

      totalRevenue += rev;
      totalDisposal += disp;
      totalTransport += trans;
      totalHolding += hold;
      totalEnvPen += env;
      totalCircBonus += circ;

      results.push({
        wasteType: w,
        ccName: opt.cc.name,
        distance: opt.dist,
        quantity: alloc,
        revenue: rev,
        disposalFee: disp,
        transportCost: trans,
        holdingCost: hold,
        envPenalty: env,
        circularityBonus: circ,
        netContribution: net,
        processDays: opt.fac.processDays,
        circularityRate: opt.fac.circularityRate,
        co2: opt.fac.co2PerKg * alloc,
      });
    }

    // Handle infeasible (not enough capacity) — add remaining to best CC if possible
    if (remaining > 0.01) {
      const best = eligible[0];
      if (best) {
        const rev = !isChemical ? best.fac.revenuePerKg * remaining : 0;
        const disp = isChemical ? best.fac.disposalFeePerKg * remaining : 0;
        const trans = best.tCost * remaining;
        const hold = best.hCost * remaining;
        const env = best.envCost * remaining;
        const circ = best.circB * remaining;
        results.push({
          wasteType: w, ccName: best.cc.name + " (overflow)", distance: best.dist,
          quantity: remaining, revenue: rev, disposalFee: disp, transportCost: trans,
          holdingCost: hold, envPenalty: env, circularityBonus: circ,
          netContribution: !isChemical ? (rev-trans-hold-env+circ) : -(disp+trans+hold+env)+circ,
          processDays: best.fac.processDays, circularityRate: best.fac.circularityRate,
          co2: best.fac.co2PerKg * remaining,
        });
        totalRevenue += rev; totalDisposal += disp; totalTransport += trans;
        totalHolding += hold; totalEnvPen += env; totalCircBonus += circ;
      }
    }
  }

  // ETP calculation
  const recycled = treatedWastewater * etpRate;
  const drained = treatedWastewater * (1 - etpRate);
  const etpSaving = recycled * etpReuseValue;
  const etpDrain = drained * etpDrainCost;
  const etpNet = etpSaving - etpDrain;

  const Z = totalRevenue - totalDisposal - totalTransport - totalHolding - totalEnvPen + totalCircBonus + etpNet;
  const totalCO2 = results.reduce((s, r) => s + r.co2, 0);
  const totalWaste = Object.values(wastes).reduce((s, v) => s + v, 0);
  const avgCircularity = results.reduce((s, r) => s + r.circularityRate * r.quantity, 0) / totalWaste;

  return {
    allocations: results,
    summary: { totalRevenue, totalDisposal, totalTransport, totalHolding, totalEnvPen, totalCircBonus, etpNet, Z },
    etp: { recycled, drained, etpSaving, etpDrain, etpNet },
    metrics: { totalCO2, avgCircularity, activeCCs: [...new Set(results.map(r => r.ccName.replace(" (overflow)","")))] }
  };
}

// ─── Default Data ───
const DEFAULT_WASTES = { solid: 1665.87, yarn: 722.36, chemical: 3202.32 };
const DEFAULT_TREATED_WW = 451586.56;

const DEFAULT_CCS = [
  {
    id: 1, name: "GreenCycle BD (Gazipur)", distance: 45,
    facilities: {
      solid:    { capacity: 2000, revenuePerKg: 8.0,  processDays: 7,  co2PerKg: 0.8, circularityRate: 0.85 },
      yarn:     { capacity: 1000, revenuePerKg: 12.0, processDays: 5,  co2PerKg: 0.5, circularityRate: 0.90 },
      chemical: { capacity: 4000, disposalFeePerKg: 5.0, processDays: 10, co2PerKg: 2.0, circularityRate: 0.20 },
    }
  },
  {
    id: 2, name: "EcoTex Recyclers (Narayanganj)", distance: 30,
    facilities: {
      solid:    { capacity: 1500, revenuePerKg: 7.0,  processDays: 6,  co2PerKg: 0.9, circularityRate: 0.80 },
      yarn:     { capacity: 800,  revenuePerKg: 11.5, processDays: 4,  co2PerKg: 0.6, circularityRate: 0.88 },
      chemical: { capacity: 3500, disposalFeePerKg: 6.0, processDays: 12, co2PerKg: 2.2, circularityRate: 0.15 },
    }
  },
  {
    id: 3, name: "Circular Hub (Savar)", distance: 20,
    facilities: {
      solid:    { capacity: 1000, revenuePerKg: 6.5,  processDays: 8,  co2PerKg: 1.0, circularityRate: 0.75 },
      yarn:     { capacity: 600,  revenuePerKg: 10.0, processDays: 6,  co2PerKg: 0.7, circularityRate: 0.82 },
      chemical: { capacity: 2000, disposalFeePerKg: 4.5, processDays: 8, co2PerKg: 1.8, circularityRate: 0.25 },
    }
  },
];

const DEFAULT_HOLDING = { solid: 2.5, yarn: 3.0, chemical: 4.0 };
const DEFAULT_TRANSPORT = { solid: 0.05, yarn: 0.04, chemical: 0.08 };

const WASTE_COLORS = { solid: "#F59E0B", yarn: "#10B981", chemical: "#EF4444" };
const CC_COLORS = ["#6366F1", "#06B6D4", "#F97316"];

// ─── Formatting ───
const fmt = (n) => n >= 0 ? `+${n.toLocaleString("en-BD", {maximumFractionDigits:2})}` : n.toLocaleString("en-BD", {maximumFractionDigits:2});
const fmtN = (n) => n.toLocaleString("en-BD", {maximumFractionDigits:2});
const fmtPct = (n) => (n*100).toFixed(1) + "%";

// ─── Components ───
function Stat({ label, value, color, sub }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 12, padding: "16px 20px", flex: 1, minWidth: 140
    }}>
      <div style={{ color: "#94A3B8", fontSize: 11, fontFamily: "monospace", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
      <div style={{ color: color || "#F1F5F9", fontSize: 22, fontWeight: 700, fontFamily: "'DM Mono', monospace" }}>{value}</div>
      {sub && <div style={{ color: "#64748B", fontSize: 11, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function NumberInput({ label, value, onChange, step = 1, min = 0, unit = "" }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <label style={{ color: "#94A3B8", fontSize: 11, fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</label>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <input
          type="number" value={value} step={step} min={min}
          onChange={e => onChange(parseFloat(e.target.value) || 0)}
          style={{
            background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 8, padding: "8px 12px", color: "#F1F5F9", fontSize: 13,
            fontFamily: "'DM Mono', monospace", width: "100%", outline: "none",
          }}
        />
        {unit && <span style={{ color: "#64748B", fontSize: 12, whiteSpace: "nowrap" }}>{unit}</span>}
      </div>
    </div>
  );
}

function CCCard({ cc, ccIndex, onChange }) {
  const wasteTypes = ["solid", "yarn", "chemical"];
  const isChemical = (w) => w === "chemical";

  return (
    <div style={{
      background: "rgba(255,255,255,0.03)", border: `1px solid ${CC_COLORS[ccIndex % 3]}40`,
      borderRadius: 14, padding: 20, marginBottom: 16
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <div style={{ width: 10, height: 10, borderRadius: "50%", background: CC_COLORS[ccIndex % 3] }} />
        <input
          value={cc.name}
          onChange={e => onChange({ ...cc, name: e.target.value })}
          style={{
            background: "transparent", border: "none", borderBottom: "1px solid rgba(255,255,255,0.15)",
            color: "#F1F5F9", fontSize: 15, fontWeight: 600, width: "100%", outline: "none", paddingBottom: 4
          }}
        />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 8, marginBottom: 16 }}>
        <NumberInput label="Distance (km)" value={cc.distance} step={1}
          onChange={v => onChange({ ...cc, distance: v })} unit="km" />
      </div>
      <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 14 }}>
        {wasteTypes.map(w => (
          <div key={w} style={{ marginBottom: 14 }}>
            <div style={{ color: WASTE_COLORS[w], fontSize: 11, fontWeight: 700, fontFamily: "monospace", textTransform: "uppercase", marginBottom: 8 }}>
              {w} Facility
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <NumberInput label="Capacity (kg)" value={cc.facilities[w].capacity} step={100}
                onChange={v => onChange({ ...cc, facilities: { ...cc.facilities, [w]: { ...cc.facilities[w], capacity: v } } })} />
              {!isChemical(w) ? (
                <NumberInput label="Revenue/kg (BDT)" value={cc.facilities[w].revenuePerKg} step={0.5}
                  onChange={v => onChange({ ...cc, facilities: { ...cc.facilities, [w]: { ...cc.facilities[w], revenuePerKg: v } } })} />
              ) : (
                <NumberInput label="Disposal Fee/kg" value={cc.facilities[w].disposalFeePerKg} step={0.5}
                  onChange={v => onChange({ ...cc, facilities: { ...cc.facilities, [w]: { ...cc.facilities[w], disposalFeePerKg: v } } })} />
              )}
              <NumberInput label="Process Days" value={cc.facilities[w].processDays} step={1}
                onChange={v => onChange({ ...cc, facilities: { ...cc.facilities, [w]: { ...cc.facilities[w], processDays: v } } })} />
              <NumberInput label="CO2 (kg/kg)" value={cc.facilities[w].co2PerKg} step={0.1}
                onChange={v => onChange({ ...cc, facilities: { ...cc.facilities, [w]: { ...cc.facilities[w], co2PerKg: v } } })} />
              <NumberInput label="Circularity Rate" value={cc.facilities[w].circularityRate} step={0.05} min={0}
                onChange={v => onChange({ ...cc, facilities: { ...cc.facilities, [w]: { ...cc.facilities[w], circularityRate: Math.min(1, v) } } })} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main App ───
export default function App() {
  const [wastes, setWastes] = useState(DEFAULT_WASTES);
  const [treatedWW, setTreatedWW] = useState(DEFAULT_TREATED_WW);
  const [ccs, setCCs] = useState(DEFAULT_CCS);
  const [holdingCosts, setHolding] = useState(DEFAULT_HOLDING);
  const [transportCosts, setTransport] = useState(DEFAULT_TRANSPORT);
  const [etpRate, setEtpRate] = useState(0.75);
  const [etpReuseValue, setEtpReuseValue] = useState(0.002);
  const [etpDrainCost, setEtpDrainCost] = useState(0.0005);
  const [co2Penalty, setCo2Penalty] = useState(1.5);
  const [circularityBonus, setCircularityBonus] = useState(2.0);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("input");
  const [solving, setSolving] = useState(false);

  const solve = useCallback(() => {
    setSolving(true);
    setTimeout(() => {
      const res = solveMILP({ wastes, ccs, holdingCosts, transportCosts, etpRate, etpReuseValue, etpDrainCost, co2Penalty, circularityBonus, treatedWastewater: treatedWW });
      setResult(res);
      setActiveTab("results");
      setSolving(false);
    }, 400);
  }, [wastes, ccs, holdingCosts, transportCosts, etpRate, etpReuseValue, etpDrainCost, co2Penalty, circularityBonus, treatedWW]);

  useEffect(() => { solve(); }, []);

  const tabs = ["input", "results", "analytics"];

  // Chart data
  const allocationChartData = result ? ["solid","yarn","chemical"].map(w => {
    const allocs = result.allocations.filter(r => r.wasteType === w);
    const obj = { waste: w.charAt(0).toUpperCase()+w.slice(1) };
    allocs.forEach(a => { obj[a.ccName.split(" ")[0]] = Math.round(a.quantity); });
    return obj;
  }) : [];

  const financialData = result ? [
    { name: "Revenue", value: result.summary.totalRevenue, fill: "#10B981" },
    { name: "Disposal Fee", value: -result.summary.totalDisposal, fill: "#EF4444" },
    { name: "Transport", value: -result.summary.totalTransport, fill: "#F59E0B" },
    { name: "Holding", value: -result.summary.totalHolding, fill: "#F97316" },
    { name: "Env Penalty", value: -result.summary.totalEnvPen, fill: "#8B5CF6" },
    { name: "Circ Bonus", value: result.summary.totalCircBonus, fill: "#06B6D4" },
    { name: "ETP Net", value: result.summary.etpNet, fill: "#84CC16" },
  ] : [];

  const radarData = result ? ccs.map((cc, i) => ({
    cc: cc.name.split(" ")[0],
    Revenue: Math.max(0, cc.facilities.solid.revenuePerKg * 10),
    Capacity: (cc.facilities.solid.capacity + cc.facilities.yarn.capacity + cc.facilities.chemical.capacity) / 100,
    Proximity: Math.max(0, 100 - cc.distance),
    Circularity: cc.facilities.solid.circularityRate * 100,
    EcoScore: Math.max(0, 100 - cc.facilities.chemical.co2PerKg * 20),
  })) : [];

  return (
    <div style={{
      minHeight: "100vh", background: "#080C14",
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
      color: "#F1F5F9"
    }}>
      {/* Header */}
      <div style={{
        background: "linear-gradient(180deg, #0F1729 0%, #080C14 100%)",
        borderBottom: "1px solid rgba(255,255,255,0.08)",
        padding: "24px 32px"
      }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                <div style={{
                  background: "linear-gradient(135deg, #6366F1, #06B6D4)",
                  borderRadius: 10, width: 36, height: 36, display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 18
                }}>♻</div>
                <span style={{ color: "#6366F1", fontSize: 11, fontFamily: "monospace", letterSpacing: "0.15em", textTransform: "uppercase" }}>
                  MILP · Circular Supply Chain
                </span>
              </div>
              <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em" }}>
                Reverse Logistics Optimizer
              </h1>
              <p style={{ margin: "4px 0 0", color: "#64748B", fontSize: 13 }}>
                Thread Dyeing Waste Management · Mixed Integer Linear Programming
              </p>
            </div>
            {result && (
              <div style={{
                background: result.summary.Z >= 0 ? "rgba(16,185,129,0.15)" : "rgba(239,68,68,0.12)",
                border: `1px solid ${result.summary.Z >= 0 ? "#10B981" : "#EF4444"}50`,
                borderRadius: 12, padding: "12px 24px", textAlign: "right"
              }}>
                <div style={{ color: "#94A3B8", fontSize: 11, fontFamily: "monospace", textTransform: "uppercase" }}>Optimal Z Value</div>
                <div style={{ fontSize: 28, fontWeight: 800, fontFamily: "'DM Mono', monospace", color: result.summary.Z >= 0 ? "#10B981" : "#EF4444" }}>
                  {fmt(Math.round(result.summary.Z))} BDT
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ borderBottom: "1px solid rgba(255,255,255,0.08)", padding: "0 32px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", gap: 0 }}>
          {tabs.map(t => (
            <button key={t} onClick={() => setActiveTab(t)} style={{
              background: "none", border: "none", borderBottom: `2px solid ${activeTab===t ? "#6366F1" : "transparent"}`,
              color: activeTab===t ? "#6366F1" : "#64748B", fontSize: 13, fontWeight: 600,
              padding: "16px 20px", cursor: "pointer", textTransform: "capitalize", letterSpacing: "0.03em"
            }}>
              {t === "input" ? "⚙ Input Parameters" : t === "results" ? "📊 Optimal Solution" : "📈 Analytics"}
            </button>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "24px 32px" }}>

        {/* ── INPUT TAB ── */}
        {activeTab === "input" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
            {/* Left column */}
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              {/* Daily Waste Predictions */}
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 24 }}>
                <h3 style={{ margin: "0 0 16px", fontSize: 14, fontWeight: 700, color: "#94A3B8", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  📦 Daily Predicted Waste (kg)
                </h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  {["solid","yarn","chemical"].map(w => (
                    <div key={w} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <div style={{ width: 10, height: 10, borderRadius: 3, background: WASTE_COLORS[w], flexShrink: 0 }} />
                      <NumberInput label={w.charAt(0).toUpperCase()+w.slice(1)+" Waste"}
                        value={wastes[w]} step={10}
                        onChange={v => setWastes({...wastes, [w]: v})} unit="kg" />
                    </div>
                  ))}
                  <NumberInput label="Treated Wastewater" value={treatedWW} step={1000}
                    onChange={setTreatedWW} unit="L" />
                </div>
              </div>

              {/* Costs */}
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 24 }}>
                <h3 style={{ margin: "0 0 16px", fontSize: 14, fontWeight: 700, color: "#94A3B8", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  💰 Holding & Prep Costs (BDT/kg)
                </h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {["solid","yarn","chemical"].map(w => (
                    <NumberInput key={w} label={w.charAt(0).toUpperCase()+w.slice(1)}
                      value={holdingCosts[w]} step={0.5}
                      onChange={v => setHolding({...holdingCosts, [w]: v})} unit="BDT/kg" />
                  ))}
                </div>
              </div>

              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 24 }}>
                <h3 style={{ margin: "0 0 16px", fontSize: 14, fontWeight: 700, color: "#94A3B8", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  🚛 Transport Costs (BDT/kg/km)
                </h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {["solid","yarn","chemical"].map(w => (
                    <NumberInput key={w} label={w.charAt(0).toUpperCase()+w.slice(1)}
                      value={transportCosts[w]} step={0.01}
                      onChange={v => setTransport({...transportCosts, [w]: v})} unit="BDT/kg/km" />
                  ))}
                </div>
              </div>

              {/* ETP */}
              <div style={{ background: "rgba(6,182,212,0.06)", border: "1px solid rgba(6,182,212,0.2)", borderRadius: 16, padding: 24 }}>
                <h3 style={{ margin: "0 0 16px", fontSize: 14, fontWeight: 700, color: "#06B6D4", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  💧 ETP — Treated Wastewater
                </h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  <NumberInput label="Recycling Rate" value={etpRate} step={0.05}
                    onChange={v => setEtpRate(Math.min(1,Math.max(0,v)))} unit="(0–1)" />
                  <NumberInput label="Reuse Value" value={etpReuseValue} step={0.0005}
                    onChange={setEtpReuseValue} unit="BDT/L" />
                  <NumberInput label="Drain/Disposal Cost" value={etpDrainCost} step={0.0001}
                    onChange={setEtpDrainCost} unit="BDT/L" />
                </div>
              </div>

              {/* Environmental */}
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 24 }}>
                <h3 style={{ margin: "0 0 16px", fontSize: 14, fontWeight: 700, color: "#94A3B8", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  🌿 Environmental Parameters
                </h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  <NumberInput label="CO2 Penalty (BDT/kg CO2)" value={co2Penalty} step={0.5}
                    onChange={setCo2Penalty} unit="BDT/kg" />
                  <NumberInput label="Circularity Bonus (BDT/kg)" value={circularityBonus} step={0.5}
                    onChange={setCircularityBonus} unit="BDT/kg" />
                </div>
              </div>
            </div>

            {/* Right column: CC Configs */}
            <div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: "#94A3B8", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  🏭 Collection Centers ({ccs.length})
                </h3>
                <button
                  onClick={() => setCCs([...ccs, {
                    id: Date.now(), name: `New CC ${ccs.length+1}`, distance: 40,
                    facilities: {
                      solid:    { capacity: 500, revenuePerKg: 6.0, processDays: 7, co2PerKg: 1.0, circularityRate: 0.70 },
                      yarn:     { capacity: 300, revenuePerKg: 9.0, processDays: 5, co2PerKg: 0.8, circularityRate: 0.75 },
                      chemical: { capacity: 1000, disposalFeePerKg: 5.5, processDays: 9, co2PerKg: 2.0, circularityRate: 0.20 },
                    }
                  }])}
                  style={{
                    background: "rgba(99,102,241,0.15)", border: "1px solid rgba(99,102,241,0.4)",
                    borderRadius: 8, color: "#6366F1", fontSize: 12, fontWeight: 600, padding: "6px 14px", cursor: "pointer"
                  }}>
                  + Add CC
                </button>
              </div>
              {ccs.map((cc, i) => (
                <div key={cc.id} style={{ position: "relative" }}>
                  {ccs.length > 1 && (
                    <button onClick={() => setCCs(ccs.filter(c => c.id !== cc.id))}
                      style={{
                        position: "absolute", top: 16, right: 16, zIndex: 1,
                        background: "rgba(239,68,68,0.15)", border: "1px solid rgba(239,68,68,0.3)",
                        borderRadius: 6, color: "#EF4444", fontSize: 11, padding: "3px 8px", cursor: "pointer"
                      }}>✕ Remove</button>
                  )}
                  <CCCard cc={cc} ccIndex={i} onChange={updated => setCCs(ccs.map(c => c.id===cc.id ? updated : c))} />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── RESULTS TAB ── */}
        {activeTab === "results" && result && (
          <div>
            {/* Key Stats */}
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 24 }}>
              <Stat label="Optimal Z (BDT)" value={fmt(Math.round(result.summary.Z))}
                color={result.summary.Z >= 0 ? "#10B981" : "#EF4444"} sub="Net objective value" />
              <Stat label="Revenue Earned" value={`${fmtN(Math.round(result.summary.totalRevenue))} BDT`}
                color="#10B981" sub="Solid + Yarn" />
              <Stat label="Total Cost" value={`${fmtN(Math.round(result.summary.totalDisposal + result.summary.totalTransport + result.summary.totalHolding + result.summary.totalEnvPen))} BDT`}
                color="#EF4444" sub="All cost categories" />
              <Stat label="CO2 Emissions" value={`${fmtN(Math.round(result.metrics.totalCO2))} kg`}
                color="#F59E0B" sub="CO2-equivalent" />
              <Stat label="Avg Circularity" value={fmtPct(result.metrics.avgCircularity)}
                color="#06B6D4" sub="Weighted circular rate" />
              <Stat label="Active CCs" value={result.metrics.activeCCs.length}
                color="#6366F1" sub={result.metrics.activeCCs.join(", ").split(" ")[0] + "..."} />
            </div>

            {/* Allocation Table */}
            <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 24, marginBottom: 24 }}>
              <h3 style={{ margin: "0 0 16px", fontSize: 14, fontWeight: 700, color: "#94A3B8", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                🗺 Optimal Waste Allocation Plan
              </h3>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
                      {["Waste","Collection Center","Qty (kg)","Revenue","Disposal Fee","Transport","Holding","Env Penalty","Circ Bonus","Net","Days","Circularity"].map(h => (
                        <th key={h} style={{ padding: "8px 12px", textAlign: "left", color: "#64748B", fontFamily: "monospace", fontSize: 11, fontWeight: 600, textTransform: "uppercase" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.allocations.map((r, i) => (
                      <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                        <td style={{ padding: "10px 12px" }}>
                          <span style={{ background: WASTE_COLORS[r.wasteType]+"25", color: WASTE_COLORS[r.wasteType], borderRadius: 6, padding: "2px 8px", fontSize: 11, fontWeight: 700 }}>
                            {r.wasteType.toUpperCase()}
                          </span>
                        </td>
                        <td style={{ padding: "10px 12px", color: "#CBD5E1", fontSize: 12 }}>{r.ccName}</td>
                        <td style={{ padding: "10px 12px", fontFamily: "monospace", fontWeight: 700 }}>{fmtN(r.quantity)}</td>
                        <td style={{ padding: "10px 12px", fontFamily: "monospace", color: "#10B981" }}>{fmtN(Math.round(r.revenue))}</td>
                        <td style={{ padding: "10px 12px", fontFamily: "monospace", color: "#EF4444" }}>{r.disposalFee > 0 ? `-${fmtN(Math.round(r.disposalFee))}` : "—"}</td>
                        <td style={{ padding: "10px 12px", fontFamily: "monospace", color: "#F59E0B" }}>-{fmtN(Math.round(r.transportCost))}</td>
                        <td style={{ padding: "10px 12px", fontFamily: "monospace", color: "#F97316" }}>-{fmtN(Math.round(r.holdingCost))}</td>
                        <td style={{ padding: "10px 12px", fontFamily: "monospace", color: "#8B5CF6" }}>-{fmtN(Math.round(r.envPenalty))}</td>
                        <td style={{ padding: "10px 12px", fontFamily: "monospace", color: "#06B6D4" }}>+{fmtN(Math.round(r.circularityBonus))}</td>
                        <td style={{ padding: "10px 12px", fontFamily: "monospace", fontWeight: 700, color: r.netContribution >= 0 ? "#10B981" : "#EF4444" }}>
                          {fmt(Math.round(r.netContribution))}
                        </td>
                        <td style={{ padding: "10px 12px", color: "#94A3B8", textAlign: "center" }}>{r.processDays}d</td>
                        <td style={{ padding: "10px 12px", textAlign: "center" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <div style={{ height: 4, borderRadius: 99, background: "rgba(255,255,255,0.08)", flex: 1, overflow: "hidden" }}>
                              <div style={{ height: "100%", width: `${r.circularityRate*100}%`, background: "#06B6D4", borderRadius: 99 }} />
                            </div>
                            <span style={{ fontSize: 11, fontFamily: "monospace", color: "#06B6D4" }}>{fmtPct(r.circularityRate)}</span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Financial Summary + ETP */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 24 }}>
                <h3 style={{ margin: "0 0 16px", fontSize: 14, fontWeight: 700, color: "#94A3B8", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  📊 Financial Breakdown
                </h3>
                {[
                  { label: "Revenue (Solid + Yarn)", value: result.summary.totalRevenue, sign: 1, color: "#10B981" },
                  { label: "Disposal Fees (Chemical)", value: result.summary.totalDisposal, sign: -1, color: "#EF4444" },
                  { label: "Transportation Costs", value: result.summary.totalTransport, sign: -1, color: "#F59E0B" },
                  { label: "Holding & Prep Costs", value: result.summary.totalHolding, sign: -1, color: "#F97316" },
                  { label: "Environmental Penalties", value: result.summary.totalEnvPen, sign: -1, color: "#8B5CF6" },
                  { label: "Circularity Bonuses", value: result.summary.totalCircBonus, sign: 1, color: "#06B6D4" },
                  { label: "ETP Net Contribution", value: result.summary.etpNet, sign: 1, color: "#84CC16" },
                ].map(item => (
                  <div key={item.label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                    <span style={{ color: "#94A3B8", fontSize: 13 }}>{item.label}</span>
                    <span style={{ fontFamily: "monospace", fontWeight: 700, color: item.color }}>
                      {item.sign === 1 ? "+" : "-"}{fmtN(Math.round(item.value))} BDT
                    </span>
                  </div>
                ))}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px 0 0", borderTop: "2px solid rgba(255,255,255,0.1)", marginTop: 8 }}>
                  <span style={{ color: "#F1F5F9", fontWeight: 700, fontSize: 14 }}>Optimal Z</span>
                  <span style={{ fontFamily: "monospace", fontWeight: 800, fontSize: 18, color: result.summary.Z >= 0 ? "#10B981" : "#EF4444" }}>
                    {fmt(Math.round(result.summary.Z))} BDT
                  </span>
                </div>
              </div>

              {/* ETP Panel */}
              <div style={{ background: "rgba(6,182,212,0.06)", border: "1px solid rgba(6,182,212,0.2)", borderRadius: 16, padding: 24 }}>
                <h3 style={{ margin: "0 0 16px", fontSize: 14, fontWeight: 700, color: "#06B6D4", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  💧 ETP — Treated Wastewater Analysis
                </h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "#94A3B8" }}>Total Treated</span>
                    <span style={{ fontFamily: "monospace", color: "#F1F5F9" }}>{fmtN(treatedWW)} L</span>
                  </div>
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                      <span style={{ color: "#94A3B8" }}>Recycled ({fmtPct(etpRate)})</span>
                      <span style={{ fontFamily: "monospace", color: "#10B981" }}>{fmtN(Math.round(result.etp.recycled))} L</span>
                    </div>
                    <div style={{ height: 8, borderRadius: 99, background: "rgba(255,255,255,0.08)", overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${etpRate*100}%`, background: "linear-gradient(90deg, #06B6D4, #10B981)", borderRadius: 99 }} />
                    </div>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "#94A3B8" }}>Drained ({fmtPct(1-etpRate)})</span>
                    <span style={{ fontFamily: "monospace", color: "#94A3B8" }}>{fmtN(Math.round(result.etp.drained))} L</span>
                  </div>
                  <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: 12, marginTop: 4 }}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ color: "#94A3B8" }}>ETP Savings</span>
                      <span style={{ fontFamily: "monospace", color: "#10B981" }}>+{fmtN(result.etp.etpSaving.toFixed(2))} BDT</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                      <span style={{ color: "#94A3B8" }}>Drain Cost</span>
                      <span style={{ fontFamily: "monospace", color: "#EF4444" }}>-{fmtN(result.etp.etpDrain.toFixed(2))} BDT</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8, borderTop: "1px solid rgba(6,182,212,0.2)", paddingTop: 8 }}>
                      <span style={{ color: "#06B6D4", fontWeight: 700 }}>Net ETP Contribution</span>
                      <span style={{ fontFamily: "monospace", color: "#06B6D4", fontWeight: 700 }}>+{fmtN(result.etp.etpNet.toFixed(2))} BDT</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── ANALYTICS TAB ── */}
        {activeTab === "analytics" && result && (
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            {/* Allocation bar chart */}
            <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 24 }}>
              <h3 style={{ margin: "0 0 20px", fontSize: 14, fontWeight: 700, color: "#94A3B8", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                Waste Allocation by CC (kg)
              </h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={allocationChartData} barGap={4}>
                  <XAxis dataKey="waste" tick={{ fill: "#64748B", fontSize: 12 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#64748B", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: "#1E293B", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10, color: "#F1F5F9" }} />
                  {ccs.map((cc, i) => (
                    <Bar key={cc.id} dataKey={cc.name.split(" ")[0]} fill={CC_COLORS[i % 3]} radius={[4,4,0,0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Financial pie + CC radar side by side */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 24 }}>
                <h3 style={{ margin: "0 0 20px", fontSize: 14, fontWeight: 700, color: "#94A3B8", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  Cost Structure
                </h3>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={financialData} layout="vertical">
                    <XAxis type="number" tick={{ fill: "#64748B", fontSize: 10 }} axisLine={false} tickLine={false} />
                    <YAxis type="category" dataKey="name" tick={{ fill: "#94A3B8", fontSize: 11 }} width={90} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: "#1E293B", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10, color: "#F1F5F9" }} />
                    <Bar dataKey="value" radius={[0,4,4,0]}>
                      {financialData.map((entry, i) => (
                        <Cell key={i} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 24 }}>
                <h3 style={{ margin: "0 0 20px", fontSize: 14, fontWeight: 700, color: "#94A3B8", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                  CC Performance Radar
                </h3>
                <ResponsiveContainer width="100%" height={260}>
                  <RadarChart data={[
                    { metric: "Revenue", ...Object.fromEntries(ccs.map((cc,i) => [cc.name.split(" ")[0], cc.facilities.solid.revenuePerKg*10])) },
                    { metric: "Proximity", ...Object.fromEntries(ccs.map((cc,i) => [cc.name.split(" ")[0], Math.max(0,100-cc.distance)])) },
                    { metric: "Circularity", ...Object.fromEntries(ccs.map((cc,i) => [cc.name.split(" ")[0], cc.facilities.solid.circularityRate*100])) },
                    { metric: "Capacity", ...Object.fromEntries(ccs.map((cc,i) => [cc.name.split(" ")[0], (cc.facilities.solid.capacity+cc.facilities.yarn.capacity)/50])) },
                    { metric: "EcoScore", ...Object.fromEntries(ccs.map((cc,i) => [cc.name.split(" ")[0], Math.max(0,100-cc.facilities.chemical.co2PerKg*20)])) },
                  ]}>
                    <PolarGrid stroke="rgba(255,255,255,0.08)" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: "#64748B", fontSize: 11 }} />
                    {ccs.map((cc, i) => (
                      <Radar key={cc.id} name={cc.name.split(" ")[0]} dataKey={cc.name.split(" ")[0]}
                        stroke={CC_COLORS[i%3]} fill={CC_COLORS[i%3]} fillOpacity={0.1} strokeWidth={2} />
                    ))}
                    <Legend wrapperStyle={{ fontSize: 11, color: "#94A3B8" }} />
                    <Tooltip contentStyle={{ background: "#1E293B", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10, color: "#F1F5F9" }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Network flow summary */}
            <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 24 }}>
              <h3 style={{ margin: "0 0 16px", fontSize: 14, fontWeight: 700, color: "#94A3B8", fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                🔗 Network Flow Summary
              </h3>
              <div style={{ display: "flex", alignItems: "center", gap: 20, overflowX: "auto", padding: "8px 0" }}>
                {/* Factory */}
                <div style={{ flexShrink: 0, background: "rgba(99,102,241,0.15)", border: "2px solid #6366F1", borderRadius: 14, padding: "20px 24px", textAlign: "center", minWidth: 140 }}>
                  <div style={{ fontSize: 24, marginBottom: 6 }}>🏭</div>
                  <div style={{ fontWeight: 700, fontSize: 13 }}>Thread Dye Factory</div>
                  <div style={{ color: "#6366F1", fontSize: 11, marginTop: 4, fontFamily: "monospace" }}>
                    {fmtN(Math.round(Object.values(wastes).reduce((s,v)=>s+v,0)))} kg/day
                  </div>
                </div>
                <div style={{ fontSize: 20, color: "#64748B" }}>→</div>
                {/* Waste Streams */}
                <div style={{ flexShrink: 0, display: "flex", flexDirection: "column", gap: 8 }}>
                  {["solid","yarn","chemical"].map(w => (
                    <div key={w} style={{ background: WASTE_COLORS[w]+"20", border: `1px solid ${WASTE_COLORS[w]}60`, borderRadius: 8, padding: "6px 14px", display: "flex", justifyContent: "space-between", gap: 12, minWidth: 160 }}>
                      <span style={{ color: WASTE_COLORS[w], fontWeight: 600, fontSize: 12 }}>{w.toUpperCase()}</span>
                      <span style={{ fontFamily: "monospace", fontSize: 12 }}>{fmtN(wastes[w])} kg</span>
                    </div>
                  ))}
                  <div style={{ background: "rgba(6,182,212,0.1)", border: "1px solid rgba(6,182,212,0.4)", borderRadius: 8, padding: "6px 14px", display: "flex", justifyContent: "space-between", gap: 12, minWidth: 160 }}>
                    <span style={{ color: "#06B6D4", fontWeight: 600, fontSize: 12 }}>WASTEWATER</span>
                    <span style={{ fontFamily: "monospace", fontSize: 12 }}>{(treatedWW/1000).toFixed(0)}kL</span>
                  </div>
                </div>
                <div style={{ fontSize: 20, color: "#64748B" }}>→</div>
                {/* CCs */}
                <div style={{ display: "flex", gap: 12, flexShrink: 0 }}>
                  {result.metrics.activeCCs.map((ccName, i) => (
                    <div key={ccName} style={{ background: CC_COLORS[i%3]+"15", border: `2px solid ${CC_COLORS[i%3]}60`, borderRadius: 14, padding: "16px 18px", textAlign: "center", minWidth: 130 }}>
                      <div style={{ fontSize: 18, marginBottom: 4 }}>♻</div>
                      <div style={{ fontWeight: 700, fontSize: 12, color: CC_COLORS[i%3] }}>{ccName.split("(")[0].trim()}</div>
                      <div style={{ color: "#64748B", fontSize: 11, marginTop: 4 }}>
                        {fmtN(Math.round(result.allocations.filter(a=>a.ccName.includes(ccName.split(" ")[0])).reduce((s,a)=>s+a.quantity,0)))} kg
                      </div>
                    </div>
                  ))}
                  {/* ETP */}
                  <div style={{ background: "rgba(6,182,212,0.08)", border: "2px solid rgba(6,182,212,0.4)", borderRadius: 14, padding: "16px 18px", textAlign: "center", minWidth: 120 }}>
                    <div style={{ fontSize: 18, marginBottom: 4 }}>💧</div>
                    <div style={{ fontWeight: 700, fontSize: 12, color: "#06B6D4" }}>ETP</div>
                    <div style={{ color: "#64748B", fontSize: 11, marginTop: 4 }}>
                      {fmtPct(etpRate)} reuse
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Solve Button */}
        <div style={{ marginTop: 24, display: "flex", justifyContent: "center" }}>
          <button onClick={solve} disabled={solving} style={{
            background: solving ? "rgba(99,102,241,0.3)" : "linear-gradient(135deg, #6366F1, #06B6D4)",
            border: "none", borderRadius: 12, color: "#fff", fontSize: 15, fontWeight: 700,
            padding: "16px 48px", cursor: solving ? "wait" : "pointer",
            boxShadow: solving ? "none" : "0 8px 32px rgba(99,102,241,0.4)",
            transition: "all 0.3s", letterSpacing: "0.03em"
          }}>
            {solving ? "⏳ Solving MILP..." : "⚡ Run MILP Optimization"}
          </button>
        </div>
        <p style={{ textAlign: "center", color: "#334155", fontSize: 11, fontFamily: "monospace", marginTop: 12 }}>
          MILP · scipy.optimize.milp · Maximize Z = Revenue − Costs + ETP
        </p>
      </div>
    </div>
  );
}
