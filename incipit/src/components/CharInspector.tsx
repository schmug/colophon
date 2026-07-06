import type { CharRecord, ModeInfo, SaliencyCell, Sampling, TurnResponse } from '../types'
import { sessionStats } from '../tapeUtils'

// top_k / context_window arrive as DISPLAY glyphs (server-side _display_char);
// map back to the real character when feeding the ban list. Backslash-u
// escapes keep PAD ('∅' = U+0000) out of the source bytes; banning PAD is
// legitimate logit surgery like any other id.
const UNDISPLAY: Record<string, string> = { '␣': ' ', '⏎': '\n', '⇥': '\t', '∅': '\u0000' }
const real = (ch: string) => UNDISPLAY[ch] ?? ch
const glyph = (ch: string) =>
  ch === ' ' ? '␣' : ch === '\n' ? '⏎' : ch === '\t' ? '⇥' : ch === '\u0000' ? '∅' : ch

export function CharInspector(props: {
  record: CharRecord | null
  saliency: SaliencyCell[] | null
  response: TurnResponse | null
  recordSets: CharRecord[][]
  sampling: Sampling
  onSampling: (s: Sampling) => void
  mode: ModeInfo | null
}) {
  const { record, saliency, response, mode } = props
  const stats = sessionStats(props.recordSets)
  const maxP = record ? Math.max(...record.top_k.map(([, p]) => p), 1e-9) : 1
  const maxD = saliency ? Math.max(...saliency.map(c => c.delta), 1e-9) : 1
  const ban = (ch: string) => {
    if (!props.sampling.banned_chars.includes(ch)) {
      props.onSampling({ ...props.sampling,
        banned_chars: [...props.sampling.banned_chars, ch] })
    }
  }
  const unban = (ch: string) =>
    props.onSampling({ ...props.sampling,
      banned_chars: props.sampling.banned_chars.filter(c => c !== ch) })

  return (
    <aside className="panel inspector">
      <h2>Char inspector</h2>
      {!record && <p className="muted">Click any generated character to see
        the probability contest it won.</p>}
      {record && (
        <>
          <div className="focus-head">
            ‘{record.display}’ · entropy {record.entropy.toFixed(3)}
            {record.truth_prob !== null && (
              <> · p={record.truth_prob.toFixed(3)} · rank #{record.truth_rank}</>
            )}
            {record.off_map && <span className="off"> · OFF-MAP</span>}
          </div>
          <h3>What it considered next (top-k, read from real logits — not a summary)</h3>
          {record.top_k.map(([ch, p], i) => (
            <div key={i} className="barrow">
              <span className="lab">{ch}</span>
              <span className="bar" style={{ width: `${(100 * p) / maxP}px` }} />
              <span className="muted">{p.toFixed(3)}</span>
              <button className="tiny"
                title="mask this character's logit to −∞ and regenerate"
                onClick={() => ban(real(ch))}>ban</button>
            </div>
          ))}
          <h3>The window it saw (literal, K chars)</h3>
          <div className="window">
            {record.context_window.map((c, i) => (
              <span key={i} className={`cell ${record.context_types[i]}`}
                title={record.context_types[i]}>{c}</span>
            ))}
          </div>
          <h3>Which remembered characters mattered (hide each one, measure how far the prediction moves)</h3>
          {!saliency && <p className="muted">measuring…</p>}
          {saliency && (
            <div className="window">
              {saliency.map((c, i) => (
                <span key={i} className={`cell ${c.is_pad ? 'pad' : ''}`}
                  title={`delta ${c.delta.toFixed(3)}`}>
                  {c.display}
                  <span className="salbar"
                    style={{ width: `${Math.round((100 * c.delta) / maxD)}%` }} />
                </span>
              ))}
            </div>
          )}
        </>
      )}

      <h3>Logit surgery</h3>
      {props.sampling.banned_chars.length === 0 ? (
        <p className="muted">No banned characters. “Ban” one above: its logit
          is set to −∞ before sampling and generation reroutes. “The model
          chose X” means “X won a probability contest you can rig.”</p>
      ) : (
        props.sampling.banned_chars.map(ch => (
          <button key={ch} className="banned-chip" title="click to unban"
            onClick={() => unban(ch)}>
            {glyph(ch)} ✕
          </button>
        ))
      )}
      {props.sampling.banned_chars.length > 0 && response && (
        <p className="muted">applied on the last turn:{' '}
          {response.banned_applied.length
            ? response.banned_applied.map(glyph).join(' ')
            : 'none (those characters are not in this model’s vocabulary — a logit that doesn’t exist can’t be masked)'}
        </p>
      )}

      <h3>Session</h3>
      <p className="muted">
        {stats.turnCount} model turns · {stats.charCount} generated chars ·
        median entropy {stats.medianEntropy.toFixed(3)} · off-map{' '}
        {stats.offMapCount}
      </p>
      {response && (
        <>
          <h3>Turn verdict (plain English)</h3>
          <p>{response.verdict}</p>
          {response.off_map && (
            <p>never-seen characters this turn:{' '}
              {response.unknown_chars.map(glyph).join(' ')}</p>
          )}
        </>
      )}
      {response?.source.matched && (
        <>
          <h3>Source echo (ground truth)</h3>
          <p className="muted">
            {mode ? (
              <a
                href={`/source?mode=${encodeURIComponent(mode.id)}&file=${encodeURIComponent(response.source.file ?? '')}&line=${response.source.line}#L${response.source.line}`}
                target="_blank"
                rel="noopener"
              >
                {response.source.file}:{response.source.line}
              </a>
            ) : (
              <>{response.source.file}:{response.source.line}</>
            )}
            {' — '}“…{response.source.pre}<b>{response.source.match}</b>
            {response.source.post}…”
          </p>
          {mode && (
            <p className="muted">
              <a href={`/corpus?mode=${encodeURIComponent(mode.id)}`} target="_blank" rel="noopener">
                browse the full corpus →
              </a>{' '}— read every training file, no prompt needed
            </p>
          )}
        </>
      )}
    </aside>
  )
}
