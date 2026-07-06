import type { ModeInfo, TapeFormat, Turn } from '../types'
import { buildTapeCells, fellOffBefore, windowSpan } from '../tapeUtils'

const DISPLAY: Record<string, string> = { ' ': '␣', '\n': '⏎', '\t': '⇥' }
const show = (ch: string) => DISPLAY[ch] ?? ch

export function TapePanel(props: {
  turns: Turn[]
  format: TapeFormat
  tape: string
  mode: ModeInfo | null
  modes: ModeInfo[]
  K: number | null
  focusPos: number | null
  onSwitchMode: (m: ModeInfo) => void
  onToggleTurn: (id: string) => void
  onEditTurn: (id: string, text: string) => void
}) {
  const { turns, format, tape, mode, modes, K, focusPos } = props
  const cells = buildTapeCells(turns, format)
  const span = focusPos !== null && K ? windowSpan(focusPos, K) : null
  const inWindow = (i: number) => span !== null && i >= span.start && i < span.end
  const fellOff = fellOffBefore(tape.length, K)
  return (
    <aside className="panel tape-panel">
      <h2>The tape</h2>
      <div className="mode-switch">
        {modes.map(m => (
          <button key={m.id} disabled={!m.available}
            title={m.available ? m.blurb : `not trained -- ${m.train_hint}`}
            className={m.id === mode?.id ? 'active' : ''}
            onClick={() => props.onSwitchMode(m)}>
            {m.label}{m.available ? '' : ' (not trained)'}
          </button>
        ))}
      </div>
      {mode && (
        <p className="muted">
          {mode.blurb} — K={mode.K ?? '?'}, {mode.params?.toLocaleString() ?? '?'} params.{' '}
          <a href="/api/scorecard" target="_blank" rel="noreferrer">openness scorecard</a>
        </p>
      )}
      <p className="muted">
        This is the ONLY thing the model ever sees: one string, rebuilt from
        your turns on every request. The server keeps no state at all.
      </p>
      <div className="budget">
        tape {tape.length} chars · window K={K ?? '?'}
        {fellOff > 0 && (
          <span className="fell-off"> · {fellOff} chars beyond the newest
            prediction's reach — struck out above: the model can no longer see them</span>
        )}
      </div>
      <div className="tape">
        {cells.map((c, i) => (
          <span key={i}
            className={`tape-cell ${c.role}${inWindow(i) ? ' in-window' : ''}${i < fellOff ? ' fell-off-cell' : ''}`}
            title={c.role === 'marker' ? 'format scaffold (role marker)' : c.role}>
            {show(c.char)}
          </span>
        ))}
        {!cells.length && <span className="muted">send a turn to begin</span>}
      </div>
      {span !== null && (
        <p className="muted">The outlined band is the K-char window the focused
          character's prediction saw (as generated -- edits below aren't re-run
          until you regenerate).</p>
      )}
      <h3>Context surgery</h3>
      <p className="muted">Exclude or edit a turn, then send again — deleting a
        turn deletes the “memory”.</p>
      {turns.map(t => (
        <div key={t.id} className={`turn-row ${t.excluded ? 'excluded' : ''}`}>
          <label title={t.excluded ? 'excluded from the tape' : 'included in the tape'}>
            <input type="checkbox" checked={!t.excluded}
              onChange={() => props.onToggleTurn(t.id)} />
            {t.role}
          </label>
          <textarea value={t.text} rows={1}
            onChange={e => props.onEditTurn(t.id, e.target.value)} />
        </div>
      ))}
    </aside>
  )
}
