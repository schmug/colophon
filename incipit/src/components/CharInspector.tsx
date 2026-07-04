import type { CharRecord, SaliencyCell, Sampling, TurnResponse } from '../types'
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
}) {
  const { record, saliency, response } = props
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
          <h3>Top-k next-char distribution (real logits)</h3>
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
          <h3>Context saliency (occlusion — real attribution)</h3>
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

      <h3>Session</h3>
      <p className="muted">
        {stats.turnCount} model turns · {stats.charCount} generated chars ·
        median entropy {stats.medianEntropy.toFixed(3)} · off-map{' '}
        {stats.offMapCount}
      </p>
      {response?.source.matched && (
        <>
          <h3>Source echo (ground truth)</h3>
          <p className="muted">{response.source.file}:{response.source.line} —
            “…{response.source.pre}<b>{response.source.match}</b>
            {response.source.post}…”</p>
        </>
      )}
    </aside>
  )
}
