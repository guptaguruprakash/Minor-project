import { useEffect, useState } from 'react'

const emptyStatus = {
  online: false,
  mode: 'unknown',
  exercise: 'unknown',
  posture: 'unknown',
  label: 'unknown',
  confidence: 0,
  samples: 0,
  received_at: null,
}

function formatWords(value) {
  return String(value || 'unknown').replaceAll('_', ' ')
}

function postureLabel(value) {
  const normalized = String(value || '').toLowerCase()
  if (normalized === 'bad' || normalized === 'wrong') return 'BAD POSTURE'
  if (normalized === 'good') return 'GOOD POSTURE'
  return normalized ? normalized.toUpperCase() : '--'
}

function timeSince(value) {
  if (!value) return 'No signal received'
  const seconds = Math.max(0, Math.round((Date.now() - new Date(value).getTime()) / 1000))
  return seconds < 2 ? 'Updated just now' : `Updated ${seconds}s ago`
}

function App() {
  const [status, setStatus] = useState(emptyStatus)
  const [summary, setSummary] = useState({ total: 0, by_label: {} })
  const [apiError, setApiError] = useState(false)
  const [lastRefresh, setLastRefresh] = useState(Date.now())

  useEffect(() => {
    let active = true

    async function refresh() {
      try {
        const [statusResponse, summaryResponse] = await Promise.all([
          fetch('/api/live-status', { cache: 'no-store' }),
          fetch('/api/sensor-data/summary', { cache: 'no-store' }),
        ])
        if (!statusResponse.ok || !summaryResponse.ok) throw new Error('API request failed')
        const [nextStatus, nextSummary] = await Promise.all([
          statusResponse.json(),
          summaryResponse.json(),
        ])
        if (active) {
          setStatus(nextStatus)
          setSummary(nextSummary)
          setApiError(false)
          setLastRefresh(Date.now())
        }
      } catch {
        if (active) setApiError(true)
      }
    }

    refresh()
    const interval = window.setInterval(refresh, 1000)
    return () => {
      active = false
      window.clearInterval(interval)
    }
  }, [])

  const confidence = Math.round((Number(status.confidence) || 0) * 100)
  const posture = status.online ? postureLabel(status.posture) : '--'
  const labels = Object.entries(summary.by_label || {})
  const maxCount = Math.max(...labels.map(([, count]) => count), 1)

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="eyebrow">ESP32 / DUAL MPU6050</span>
          <h1>Form<br /><em>check.</em></h1>
        </div>
        <div className="status-stack">
          <div className={`connection ${status.online ? 'is-online' : ''}`}>
            <span className="status-dot" />
            <strong>{apiError ? 'API unavailable' : status.online ? 'ESP32 live' : 'Waiting for ESP32'}</strong>
          </div>
          <span className="mode">Mode / {formatWords(status.mode)}</span>
        </div>
      </header>

      <section className="hero-grid">
        <article className="surface exercise-card">
          <div className="card-heading"><span className="section-label">Detected exercise</span><span className="pulse-mark" /></div>
          <h2>{status.online ? formatWords(status.exercise) : 'Waiting'}</h2>
          <div className="card-footer"><span>{status.online ? formatWords(status.label) : 'No prediction yet'}</span><span>{status.samples || 0} samples</span></div>
        </article>

        <article className={`surface posture-card ${posture === 'BAD POSTURE' ? 'is-bad' : ''}`}>
          <span className="section-label">Posture signal</span>
          <h2>{posture}</h2>
          <div className="confidence-block">
            <div className="card-footer"><span>Confidence</span><strong>{confidence}%</strong></div>
            <div className="confidence-track"><span style={{ width: `${confidence}%` }} /></div>
          </div>
        </article>
      </section>

      <section className="lower-grid">
        <article className="surface signal-card">
          <div className="card-heading"><span className="section-label">Connection telemetry</span><span className="live-caption">{timeSince(status.received_at)}</span></div>
          <div className="telemetry-row"><span>Last packet</span><strong>{status.received_at ? new Date(status.received_at).toLocaleTimeString() : '--:--:--'}</strong></div>
          <div className="telemetry-row"><span>Refresh cadence</span><strong>1 second</strong></div>
          <div className="telemetry-row"><span>Frontend</span><strong>React + Vite</strong></div>
        </article>

        <article className="surface dataset-card">
          <div className="card-heading"><span className="section-label">Training dataset</span><strong>{summary.total || 0} rows</strong></div>
          {labels.length ? labels.map(([label, count]) => (
            <div className="dataset-row" key={label}>
              <div><span>{formatWords(label)}</span><strong>{count}</strong></div>
              <div className="dataset-track"><span style={{ width: `${(count / maxCount) * 100}%` }} /></div>
            </div>
          )) : <p className="empty-state">No labeled sensor data stored yet.</p>}
        </article>
      </section>

      <footer className="footer-note">Live status is supplied by FastAPI. KNN inference runs locally on the ESP32 before the prediction reaches this screen.</footer>
      <span className="refresh-indicator">{new Date(lastRefresh).toLocaleTimeString()}</span>
    </main>
  )
}

export default App
