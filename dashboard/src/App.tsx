import { useState, useEffect, useLayoutEffect, useRef } from 'react'
import * as echarts from 'echarts'
import './index.css'

const API = '/api'

// ============ Types ============
interface TSStructure {
  symbol: string; date: string; structure: string;
  near_price: number; far_price: number; total_spread_pct: number;
  curve: { symbol: string; price: number; month: number; year: number }[];
}
interface TSHistory { date: string; spread_pct: number; structure: string }
interface IVSummary {
  product: string; date: string; n_with_iv: number;
  underlying_price: number | null; atm_iv: number | null;
  otm_put_iv: number | null; otm_call_iv: number | null;
  skew: number | null; hv_20: number | null; iv_hv_ratio: number | null;
}
interface OptRecord {
  symbol: string; product: string; option_type: string; strike: number;
  underlying_price: number; moneyness: number; days_to_expiry: number;
  market_price: number; implied_vol: number;
  delta: number; gamma: number; theta: number; vega: number; rho: number;
  volume: number; open_interest: number; price_source: string;
}
interface FutSymbol {
  symbol: string; close: number; date: string; ret_5d: number; ret_20d: number;
  vol_20d: number; volume: number; oi: number;
}
interface Overview {
  ts_symbols: number; ts_records: number; ts_latest_date: string;
  opt_symbols: number; opt_contracts: number; fut_symbols: number;
  backwardation_count: number; contango_count: number;
}

// ============ Hooks ============
function useApi<T>(url: string) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    if (!url) { setLoading(false); return }
    setLoading(true)
    fetch(url).then(r => r.json()).then(d => { setData(d); setLoading(false) }).catch(() => setLoading(false))
  }, [url])
  return { data, loading }
}

// ============ Chart Component ============
// Uses useLayoutEffect to guarantee DOM dimensions exist before echarts.init
function Chart({ option, height }: { option: object; height?: number }) {
  const elRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<echarts.ECharts | null>(null)
  const roRef = useRef<ResizeObserver | null>(null)

  useLayoutEffect(() => {
    const el = elRef.current
    if (!el) return
    if (!chartRef.current) {
      chartRef.current = echarts.init(el, undefined, { renderer: 'canvas' })
      roRef.current = new ResizeObserver(() => chartRef.current?.resize())
      roRef.current.observe(el)
    }
    try { chartRef.current.setOption(option, true) } catch {}
  })

  useEffect(() => () => {
    roRef.current?.disconnect()
    chartRef.current?.dispose()
    chartRef.current = null
  }, [])

  return <div ref={elRef} style={{ width: '100%', height: height || 400 }} />
}

function Loading() {
  return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200, color: '#78909c' }}>加载中...</div>
}

// ============ Overview ============
function OverviewPage() {
  const { data, loading } = useApi<Overview>(`${API}/overview`)
  const { data: futData } = useApi<FutSymbol[]>(`${API}/futures/symbols`)
  if (loading || !data) return <Loading />
  const movers = futData ? [...futData].sort((a, b) => Math.abs(b.ret_5d) - Math.abs(a.ret_5d)).slice(0, 12) : []

  return (
    <div>
      <div className="grid-4">
        <div className="stat-card"><div className="stat-label">期限结构</div><div className="stat-value">{data.ts_symbols}</div><div className="stat-sub">{data.ts_records.toLocaleString()} 条</div></div>
        <div className="stat-card"><div className="stat-label">期权</div><div className="stat-value">{data.opt_symbols}</div><div className="stat-sub">{data.opt_contracts.toLocaleString()} 合约</div></div>
        <div className="stat-card"><div className="stat-label">期货</div><div className="stat-value">{data.fut_symbols}</div><div className="stat-sub">日线</div></div>
        <div className="stat-card"><div className="stat-label">最新</div><div className="stat-value">{data.ts_latest_date}</div><div className="stat-sub">数据更新</div></div>
      </div>
      <div className="grid-2" style={{ marginTop: 16 }}>
        <div className="card"><Chart height={280} option={{
          title: { text: '结构分布', left: 'center', top: 10, textStyle: { color: '#4fc3f7', fontSize: 14 } },
          tooltip: { trigger: 'item' },
          series: [{ type: 'pie', radius: ['35%', '65%'], center: ['50%', '55%'], label: { color: '#e0e0e0' },
            data: [
              { value: data.backwardation_count, name: `Back (${data.backwardation_count})`, itemStyle: { color: '#26a69a' } },
              { value: data.contango_count, name: `Cont (${data.contango_count})`, itemStyle: { color: '#ef5350' } },
            ] }]
        }} /></div>
        <div className="card">{futData && futData.length > 0 && <Chart height={280} option={{
          title: { text: '5日涨跌TOP20', left: 'center', top: 10, textStyle: { color: '#4fc3f7', fontSize: 14 } },
          tooltip: { trigger: 'axis' },
          xAxis: { type: 'category', data: [...futData].sort((a, b) => b.ret_5d - a.ret_5d).slice(0, 20).map(s => s.symbol), axisLabel: { color: '#78909c', rotate: 45, fontSize: 10 } },
          yAxis: { type: 'value', axisLabel: { color: '#78909c', formatter: '{value}%' }, splitLine: { lineStyle: { color: '#1e3a5f' } } },
          series: [{ type: 'bar', data: [...futData].sort((a, b) => b.ret_5d - a.ret_5d).slice(0, 20).map(s => ({ value: s.ret_5d, itemStyle: { color: s.ret_5d >= 0 ? '#ef5350' : '#26a69a' } })) }],
          grid: { left: 50, right: 10, top: 40, bottom: 60 },
        }} />}</div>
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <div style={{ padding: '10px 0 6px', fontSize: 13, color: '#4fc3f7', fontWeight: 600 }}>异动品种</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {movers.map(m => (
            <div key={m.symbol} className="symbol-chip" style={{ background: 'rgba(79,195,247,0.08)', borderColor: m.ret_5d >= 0 ? '#ef5350' : '#26a69a' }}>
              <span style={{ fontWeight: 600, marginRight: 4 }}>{m.symbol}</span>
              <span style={{ color: m.ret_5d >= 0 ? '#ef5350' : '#26a69a', fontSize: 11 }}>{m.ret_5d >= 0 ? '+' : ''}{m.ret_5d.toFixed(2)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ============ Term Structure ============
function TermStructurePage() {
  const { data, loading } = useApi<TSStructure[]>(`${API}/ts/structure`)
  const [selected, setSelected] = useState('')
  const historyUrl = selected ? `${API}/ts/history/${selected}` : ''
  const { data: history } = useApi<TSHistory[]>(historyUrl)

  useEffect(() => { if (data && data.length > 0 && !selected) setSelected(data[0].symbol) }, [data, selected])
  if (loading || !data) return <Loading />

  const item = data.find(d => d.symbol === selected)
  const pts = item?.curve?.filter(c => c.price > 0).map(c => ({ ...c, year: c.year < 100 ? c.year + 2000 : c.year }))
    .sort((a, b) => (a.year * 12 + a.month) - (b.year * 12 + b.month)) || []
  const isB = item?.structure === 'backwardation'
  const clr = isB ? '#26a69a' : '#ef5350'

  return (
    <div style={{ display: 'flex', gap: 12 }}>
      <div style={{ width: 140, flexShrink: 0, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 6, maxHeight: 'calc(100vh - 180px)', overflowY: 'auto' }}>
        <div style={{ padding: '8px 10px', borderBottom: '1px solid var(--border)', fontSize: 12, color: '#4fc3f7', fontWeight: 600 }}>品种 ({data.length})</div>
        {data.map(r => (
          <div key={r.symbol} onClick={() => setSelected(r.symbol)}
            style={{ padding: '6px 10px', cursor: 'pointer', fontSize: 12, transition: 'background 0.1s',
              background: selected === r.symbol ? 'rgba(79,195,247,0.15)' : 'transparent',
              borderLeft: selected === r.symbol ? '3px solid #4fc3f7' : '3px solid transparent',
              color: selected === r.symbol ? '#4fc3f7' : '#e0e0e0',
              fontWeight: selected === r.symbol ? 600 : 400,
            }}>
            {r.symbol}
            <span style={{ float: 'right', fontSize: 9, color: r.structure === 'backwardation' ? '#26a69a' : '#ef5350' }}>
              {r.structure === 'backwardation' ? 'B' : 'C'}
            </span>
          </div>
        ))}
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
        {/* Curve chart */}
        <div className="card">
          <div style={{ marginBottom: 8, fontSize: 13, color: '#4fc3f7', fontWeight: 600 }}>
            {selected} 期限结构 <span style={{ color: isB ? '#26a69a' : '#ef5350', fontSize: 11 }}>({isB ? '贴水' : '升水'})</span>
          </div>
          {pts.length > 0 ? (
            <Chart height={280} option={{
              tooltip: { trigger: 'axis' },
              xAxis: { type: 'category', data: pts.map(c => c.symbol || `${c.year}/${c.month}`), axisLabel: { color: '#78909c', fontSize: 9, rotate: 30 } },
              yAxis: { type: 'value', axisLabel: { color: '#78909c' }, splitLine: { lineStyle: { color: '#1e3a5f' } } },
              series: [{ type: 'line', data: pts.map(c => c.price), smooth: true, lineStyle: { color: clr, width: 3 },
                areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: isB ? 'rgba(38,166,154,0.25)' : 'rgba(239,83,80,0.25)' }, { offset: 1, color: 'rgba(0,0,0,0)' }] } },
                symbol: 'circle', symbolSize: 6, itemStyle: { color: clr } }],
              grid: { left: 60, right: 16, top: 10, bottom: 50 },
            }} />
          ) : <Loading />}
        </div>
        {/* History chart */}
        <div className="card">
          <div style={{ marginBottom: 8, fontSize: 13, color: '#4fc3f7', fontWeight: 600 }}>{selected} 价差历史</div>
          {history && history.length > 0 ? (
            <Chart height={280} option={{
              tooltip: { trigger: 'axis' },
              xAxis: { type: 'category', data: history.map(h => h.date), axisLabel: { color: '#78909c', fontSize: 9 } },
              yAxis: { type: 'value', axisLabel: { color: '#78909c', formatter: '{value}%' }, splitLine: { lineStyle: { color: '#1e3a5f' } } },
              visualMap: { show: false, pieces: [{ lt: 0, color: '#26a69a' }, { gte: 0, color: '#ef5350' }] },
              series: [{ type: 'line', data: history.map(h => h.spread_pct), smooth: true, lineStyle: { width: 1.5 },
                markLine: { data: [{ yAxis: 0, lineStyle: { color: '#78909c', type: 'dashed' } }] } }],
              dataZoom: [{ type: 'slider', height: 18, bottom: 4 }],
              grid: { left: 56, right: 16, top: 10, bottom: 40 },
            }} />
          ) : <Loading />}
        </div>
      </div>
    </div>
  )
}

// ============ Options ============
function OptionsPage() {
  const { data: symData } = useApi<{ symbols: string[]; dates: string[] }>(`${API}/options/symbols`)
  const { data: ivData } = useApi<IVSummary[]>(`${API}/options/iv_summary`)
  const [sym, setSym] = useState('')
  const [dt, setDt] = useState('')

  useEffect(() => {
    if (symData && !sym && symData.symbols.length > 0) setSym(symData.symbols[0])
    if (symData && !dt && symData.dates.length > 0) setDt(symData.dates[symData.dates.length - 1])
  }, [symData])

  const surfaceUrl = sym && dt ? `${API}/options/surface?symbol=${sym}&date=${dt}` : ''
  const { data: surface } = useApi<OptRecord[]>(surfaceUrl)

  if (!symData || !ivData) return <Loading />

  // Prepare chart data
  const calls = surface?.filter(s => s.option_type === 'CALL') || []
  const puts = surface?.filter(s => s.option_type === 'PUT') || []
  const allExps = [...new Set((surface || []).map(s => s.days_to_expiry))].sort((a, b) => a - b)
  const nearExp = allExps.find(e => e >= 14) || allExps[0]

  // IV Smile data: near-expiry call+put by moneyness
  const smileCalls = calls.filter(s => s.days_to_expiry === nearExp).sort((a, b) => a.moneyness - b.moneyness)
  const smilePuts = puts.filter(s => s.days_to_expiry === nearExp).sort((a, b) => a.moneyness - b.moneyness)
  const smileMK = [...new Set([...smileCalls, ...smilePuts].map(s => Math.round(s.moneyness * 100) / 100))].sort((a, b) => a - b)

  // Heatmap: all calls, moneyness x expiry
  const hmMks = [...new Set(calls.map(s => Math.round(s.moneyness * 100) / 100))].sort((a, b) => a - b)
  const hmExps = [...new Set(calls.map(s => s.days_to_expiry))].sort((a, b) => a - b)
  const hmData: number[][] = []
  for (const e of hmExps) {
    const row: number[] = []
    for (const m of hmMks) {
      const f = calls.find(c => Math.abs(Math.round(c.moneyness * 100) / 100 - m) < 0.005 && c.days_to_expiry === e)
      row.push(f?.implied_vol ? +(f.implied_vol * 100).toFixed(1) : 0)
    }
    hmData.push(row)
  }
  const hmVals = hmData.flat().filter(v => v > 0)

  return (
    <div style={{ display: 'flex', gap: 12 }}>
      {/* Sidebar: IV summary list */}
      <div style={{ width: 180, flexShrink: 0, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 6, maxHeight: 'calc(100vh - 180px)', overflowY: 'auto' }}>
        <div style={{ padding: '8px 10px', borderBottom: '1px solid var(--border)', fontSize: 12, color: '#4fc3f7', fontWeight: 600 }}>
          IV品种 ({symData.symbols.length})
          <div style={{ marginTop: 4 }}>
            <select value={dt} onChange={e => setDt(e.target.value)} style={{ width: '100%', fontSize: 11, padding: '3px 6px', background: 'var(--bg-card)', color: '#e0e0e0', border: '1px solid var(--border)', borderRadius: 3 }}>
              {symData.dates.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
        </div>
        {ivData.filter(r => r.date === dt).map(r => (
          <div key={r.product} onClick={() => setSym(r.product)}
            style={{ padding: '5px 10px', cursor: 'pointer', fontSize: 11,
              background: sym === r.product ? 'rgba(79,195,247,0.15)' : 'transparent',
              borderLeft: sym === r.product ? '3px solid #4fc3f7' : '3px solid transparent',
            }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: sym === r.product ? 600 : 400, color: sym === r.product ? '#4fc3f7' : '#e0e0e0' }}>{r.product}</span>
              <span style={{ color: '#4fc3f7' }}>{r.atm_iv !== null ? (r.atm_iv * 100).toFixed(1) + '%' : '-'}</span>
            </div>
          </div>
        ))}
      </div>
      {/* Charts */}
      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div className="grid-2">
          {/* IV Smile */}
          <div className="card">
            <div style={{ marginBottom: 6, fontSize: 13, color: '#4fc3f7', fontWeight: 600 }}>
              {sym} IV微笑曲线 ({nearExp}天到期)
              <span style={{ fontSize: 11, color: '#78909c', marginLeft: 8 }}>{calls.length}C + {puts.length}P</span>
            </div>
            {surface && surface.length > 0 && smileMK.length > 0 ? <Chart height={300} option={{
              tooltip: { trigger: 'axis', valueFormatter: (v: number) => v ? (v * 100).toFixed(1) + '%' : '-' },
              legend: { data: ['Call', 'Put'], top: 26, textStyle: { color: '#78909c', fontSize: 11 } },
              xAxis: { type: 'category', data: smileMK.map(m => m.toFixed(2)), name: 'K/S', nameTextStyle: { color: '#78909c' }, axisLabel: { color: '#78909c', fontSize: 10 } },
              yAxis: { type: 'value', axisLabel: { color: '#78909c', formatter: (v: number) => (v * 100).toFixed(0) + '%' }, splitLine: { lineStyle: { color: '#1e3a5f' } } },
              series: [
                { name: 'Call', type: 'line', smooth: true, data: smileMK.map(m => smileCalls.find(c => Math.abs(c.moneyness - m) < 0.015)?.implied_vol ?? null), lineStyle: { color: '#ef5350' }, itemStyle: { color: '#ef5350' } },
                { name: 'Put', type: 'line', smooth: true, data: smileMK.map(m => smilePuts.find(p => Math.abs(p.moneyness - m) < 0.015)?.implied_vol ?? null), lineStyle: { color: '#26a69a' }, itemStyle: { color: '#26a69a' } },
              ],
              grid: { left: 60, right: 16, top: 56, bottom: 30 },
            }} /> : <Loading />}
          </div>
          {/* IV Heatmap */}
          <div className="card">
            <div style={{ marginBottom: 6, fontSize: 13, color: '#4fc3f7', fontWeight: 600 }}>
              {sym} IV热力图 (Call)
            </div>
            {hmVals.length > 2 ? <Chart height={300} option={{
              tooltip: { formatter: (p: any) => `K/S=${hmMks[p.value[0]]?.toFixed(2)}, ${hmExps[p.value[1]]}d到期: ${p.value[2]}%` },
              xAxis: { type: 'category', data: hmMks.map(m => m.toFixed(2)), axisLabel: { color: '#78909c', fontSize: 9 }, splitArea: { show: true } },
              yAxis: { type: 'category', data: hmExps.map(e => e + 'd'), axisLabel: { color: '#78909c', fontSize: 9 }, splitArea: { show: true } },
              visualMap: { min: Math.min(...hmVals), max: Math.max(...hmVals), calculable: true, orient: 'horizontal', left: 'center', bottom: 0,
                inRange: { color: ['#0d47a1', '#26a69a', '#ffc107', '#ef5350'] }, textStyle: { color: '#78909c' } },
              series: [{ type: 'heatmap', data: hmData.flatMap((row, j) => row.map((v, i) => [i, j, v])),
                label: { show: hmMks.length * hmExps.length < 60, color: '#e0e0e0', fontSize: 9, formatter: (p: any) => p.value[2] } }],
              grid: { left: 50, right: 16, top: 10, bottom: 56 },
            }} /> : <div style={{ padding: 20, color: '#78909c', textAlign: 'center' }}>数据点不足，无法生成热力图 ({surface?.length || 0}条记录)</div>}
          </div>
        </div>
        {/* IV Term Structure */}
        <div className="card">
          <div style={{ marginBottom: 6, fontSize: 13, color: '#4fc3f7', fontWeight: 600 }}>
            {sym} IV期限结构 (不同K/S)
          </div>
          {surface && surface.length > 0 ? (() => {
            const mkLvls = [0.90, 0.95, 1.0, 1.05, 1.10]
            const colors = ['#ef5350', '#ff9800', '#4fc3f7', '#26a69a', '#7c4dff']
            return <Chart height={260} option={{
              tooltip: { trigger: 'axis', valueFormatter: (v: number) => v ? v.toFixed(1) + '%' : '-' },
              legend: { data: mkLvls.map(m => `K/S=${m}`), top: 26, textStyle: { color: '#78909c', fontSize: 10 } },
              xAxis: { type: 'category', data: allExps.map(e => e + '天'), axisLabel: { color: '#78909c' } },
              yAxis: { type: 'value', axisLabel: { color: '#78909c', formatter: '{value}%' }, splitLine: { lineStyle: { color: '#1e3a5f' } } },
              series: mkLvls.map((m, i) => ({
                name: `K/S=${m}`, type: 'line', smooth: true,
                data: allExps.map(e => { const f = calls.find(c => Math.abs(c.moneyness - m) < 0.03 && c.days_to_expiry === e); return f ? +(f.implied_vol * 100).toFixed(1) : null }),
                lineStyle: { color: colors[i] }, itemStyle: { color: colors[i] },
              })),
              grid: { left: 56, right: 16, top: 56, bottom: 24 },
            }} />
          })() : <Loading />}
        </div>
      </div>
    </div>
  )
}

// ============ Greeks ============
function GreeksPage() {
  const { data: symData } = useApi<{ symbols: string[]; dates: string[] }>(`${API}/options/symbols`)
  const [sym, setSym] = useState('')
  const [dt, setDt] = useState('')
  useEffect(() => {
    if (symData && !sym) setSym(symData.symbols[0])
    if (symData && !dt && symData.dates.length > 0) setDt(symData.dates[symData.dates.length - 1])
  }, [symData])

  const surfaceUrl = sym && dt ? `${API}/options/surface?symbol=${sym}&date=${dt}` : ''
  const { data: surface } = useApi<OptRecord[]>(surfaceUrl)

  if (!symData) return <Loading />

  const allExps = [...new Set((surface || []).map(s => s.days_to_expiry))].sort((a, b) => a - b)
  const nearExp = allExps.find(e => e >= 14) || allExps[0]
  const calls = (surface || []).filter(s => s.option_type === 'CALL')
  const nearCalls = calls.filter(s => s.days_to_expiry === nearExp).sort((a, b) => a.moneyness - b.moneyness)
  const atmCalls = calls.filter(s => Math.abs(s.moneyness - 1.0) < 0.05)

  // Greeks chart builder
  function greekChart(title: string, field: 'delta' | 'gamma' | 'theta' | 'vega', color: string, fmt: (v: number) => string) {
    return nearCalls.length > 0 ? <Chart height={320} option={{
      title: { text: `${sym} ${title}`, left: 'center', top: 8, textStyle: { color: '#4fc3f7', fontSize: 14 } },
      tooltip: { trigger: 'axis', valueFormatter: (v: number) => v != null ? fmt(v) : '-' },
      xAxis: { type: 'category', data: nearCalls.map(s => s.moneyness.toFixed(2)), name: 'K/S (行权价/标的价格)', nameLocation: 'center', nameGap: 28, nameTextStyle: { color: '#78909c', fontSize: 11 },
        axisLabel: { color: '#78909c', fontSize: 10 } },
      yAxis: { type: 'value', axisLabel: { color: '#78909c' }, splitLine: { lineStyle: { color: '#1e3a5f' } } },
      series: [{ type: 'line', smooth: true, data: nearCalls.map(s => s[field]),
        lineStyle: { color, width: 2 }, itemStyle: { color },
        areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: color.replace(')', ',0.2)').replace('rgb', 'rgba') }, { offset: 1, color: 'rgba(0,0,0,0)' }] } } }],
      grid: { left: 60, right: 16, top: 44, bottom: 48 },
    }} /> : <div style={{ padding: 20, color: '#78909c', textAlign: 'center' }}>无数据</div>
  }

  return (
    <div>
      <div className="card" style={{ marginBottom: 12, padding: '10px 16px', display: 'flex', gap: 12, alignItems: 'center' }}>
        <span style={{ fontSize: 12, color: '#78909c' }}>品种</span>
        <select value={sym} onChange={e => setSym(e.target.value)} style={{ background: 'var(--bg-card)', color: '#e0e0e0', border: '1px solid var(--border)', padding: '4px 8px', borderRadius: 3 }}>
          <option value="">选择</option>
          {symData.symbols.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <span style={{ fontSize: 12, color: '#78909c' }}>日期</span>
        <select value={dt} onChange={e => setDt(e.target.value)} style={{ background: 'var(--bg-card)', color: '#e0e0e0', border: '1px solid var(--border)', padding: '4px 8px', borderRadius: 3 }}>
          {symData.dates.map(d => <option key={d} value={d}>{d}</option>)}
        </select>
        {surface && <span style={{ fontSize: 11, color: '#78909c', marginLeft: 8 }}>{surface.length}条记录 · {nearExp}天到期</span>}
      </div>
      <div className="grid-2">
        <div className="card">{greekChart('Delta (方向敏感度)', 'delta', '#4fc3f7', v => v.toFixed(4))}</div>
        <div className="card">{greekChart('Gamma (Δ的敏感度)', 'gamma', '#ffc107', v => v.toFixed(5))}</div>
      </div>
      <div className="grid-2" style={{ marginTop: 12 }}>
        <div className="card">{greekChart('Theta (时间衰减/天)', 'theta', '#ef5350', v => v.toFixed(4))}</div>
        <div className="card">{greekChart('Vega (波动率敏感度)', 'vega', '#7c4dff', v => v.toFixed(4))}</div>
      </div>
      {/* ATM Greeks Table */}
      <div className="card" style={{ marginTop: 12 }}>
        <div style={{ marginBottom: 8, fontSize: 13, color: '#4fc3f7', fontWeight: 600 }}>平值期权Greeks明细</div>
        <div className="scroll-table" style={{ maxHeight: 300 }}>
          <table><thead><tr><th>合约</th><th>类型</th><th>K/S</th><th>到期天数</th><th>IV</th><th>Delta</th><th>Gamma</th><th>Theta</th><th>Vega</th><th>价格</th></tr></thead>
          <tbody>{atmCalls.sort((a, b) => a.days_to_expiry - b.days_to_expiry).slice(0, 40).map((s, i) => (
            <tr key={i}>
              <td style={{ fontSize: 10 }}>{s.symbol}</td>
              <td><span className={`tag ${s.option_type === 'CALL' ? 'tag-cont' : 'tag-back'}`}>C</span></td>
              <td>{s.moneyness.toFixed(3)}</td>
              <td>{s.days_to_expiry}d</td>
              <td style={{ color: '#4fc3f7' }}>{(s.implied_vol * 100).toFixed(1)}%</td>
              <td>{s.delta?.toFixed(4)}</td>
              <td>{s.gamma?.toFixed(5)}</td>
              <td>{s.theta?.toFixed(4)}</td>
              <td>{s.vega?.toFixed(4)}</td>
              <td>{s.market_price?.toFixed(2)}</td>
            </tr>
          ))}</tbody></table>
        </div>
      </div>
    </div>
  )
}

// ============ Futures ============
function FuturesPage() {
  const { data, loading } = useApi<FutSymbol[]>(`${API}/futures/symbols`)
  if (loading || !data) return <Loading />
  const sorted = [...data].sort((a, b) => b.ret_5d - a.ret_5d)
  return (
    <div className="card">
      <div className="card-header"><span className="card-title">期货品种</span><span className="card-badge">{data.length}</span></div>
      <div className="scroll-table"><table><thead><tr><th>品种</th><th>收盘</th><th>5日%</th><th>20日%</th><th>波动率</th><th>成交量</th><th>持仓量</th></tr></thead>
      <tbody>{sorted.map(r => (
        <tr key={r.symbol}><td style={{ fontWeight: 600 }}>{r.symbol}</td><td>{r.close.toFixed(2)}</td>
        <td className={r.ret_5d >= 0 ? 'up' : 'down'}>{r.ret_5d >= 0 ? '+' : ''}{r.ret_5d.toFixed(2)}%</td>
        <td className={r.ret_20d >= 0 ? 'up' : 'down'}>{r.ret_20d >= 0 ? '+' : ''}{r.ret_20d.toFixed(2)}%</td>
        <td>{r.vol_20d.toFixed(1)}%</td><td>{r.volume?.toLocaleString()}</td><td>{r.oi?.toLocaleString()}</td></tr>
      ))}</tbody></table></div>
    </div>
  )
}

// ============ App ============
type Page = 'overview' | 'ts' | 'options' | 'greeks' | 'futures'

function App() {
  const [page, setPage] = useState<Page>('overview')
  const tabs: { key: Page; label: string }[] = [
    { key: 'overview', label: '总览' }, { key: 'ts', label: '期限结构' },
    { key: 'options', label: '波动率' }, { key: 'greeks', label: 'Greeks' },
    { key: 'futures', label: '期货' },
  ]
  return (
    <>
      <div className="header"><div><h1>Futures Analytics</h1><div className="subtitle">期限结构 | 波动率 | Greeks | 期货行情</div></div>
        <div className="header-right"><span><span className="status-dot" />Live</span></div></div>
      <div className="nav">{tabs.map(t => (
        <div key={t.key} className={`nav-item ${page === t.key ? 'active' : ''}`} onClick={() => setPage(t.key)}>{t.label}</div>
      ))}</div>
      <div className="main">
        {page === 'overview' && <OverviewPage />}
        {page === 'ts' && <TermStructurePage />}
        {page === 'options' && <OptionsPage />}
        {page === 'greeks' && <GreeksPage />}
        {page === 'futures' && <FuturesPage />}
      </div>
    </>
  )
}

export default App
