import { useState } from 'react'
import type { Act } from '../acts'
import type { ModelTurnData } from '../App'
import type { Sampling, TapeFormat, Turn, TurnResponse } from '../types'

function entColor(e: number) {
  const h = Math.round(140 * (1 - Math.max(0, Math.min(1, e))))
  return `hsl(${h} 60% 50% / 0.35)`
}

const CAND_LABELS = ['cooler', 'as set', 'hotter']

export function ConversationPanel(props: {
  turns: Turn[]
  modelData: Record<string, ModelTurnData>
  busy: boolean
  error: string
  candidates: TurnResponse[] | null
  act: number | null
  acts: Act[]
  format: TapeFormat
  sampling: Sampling
  focus: { turnId: string; index: number } | null
  onSend: (text: string) => void
  onCommit: (r: TurnResponse) => void
  onStartAct: (n: number) => void
  onFreePlay: () => void
  onSampling: (s: Sampling) => void
  onFormat: (f: TapeFormat) => void
  onFocusChar: (turnId: string, index: number) => void
}) {
  const [draft, setDraft] = useState('')
  const activeAct = props.acts.find(a => a.n === props.act) ?? null
  return (
    <main className="panel conversation">
      <h1><span className="glyph">❧</span> Incipit</h1>
      <p className="banner">Every number on this page is read from the model's
        own weights, on your machine. The 2025 glassboxllm mock had to simulate
        these signals — Colophon computes them for real.</p>

      <div className="acts">
        {props.acts.map(a => (
          <button key={a.n} className={a.n === props.act ? 'active' : ''}
            onClick={() => props.onStartAct(a.n)}>
            Act {a.n}: {a.title}
          </button>
        ))}
        <button className={props.act === null ? 'active' : ''}
          onClick={props.onFreePlay}>
          Free play
        </button>
      </div>
      {activeAct && (
        <div className="act-copy">
          <p>{activeAct.copy}</p>
          <button onClick={() => setDraft(activeAct.prompt)}>
            Use the suggested prompt
          </button>
        </div>
      )}

      <div className="bubbles">
        {props.turns.map(t => t.role === 'user' ? (
          <div key={t.id} className={`bubble user ${t.excluded ? 'excluded' : ''}`}>
            {t.text}
          </div>
        ) : (
          <div key={t.id} className={`bubble model ${t.excluded ? 'excluded' : ''}`}>
            {props.modelData[t.id]
              ? props.modelData[t.id].records
                  .filter(r => r.is_continuation)
                  .map((r, i) => (
                    <span key={i} style={{ background: entColor(r.entropy) }}
                      className={props.focus?.turnId === t.id &&
                        props.focus.index === i ? 'focus' : ''}
                      title={`entropy ${r.entropy.toFixed(3)}` +
                        (r.truth_prob !== null
                          ? ` · p=${r.truth_prob.toFixed(3)}` : '')}
                      onClick={() => props.onFocusChar(t.id, i)}>
                      {r.char}
                    </span>
                  ))
              : t.text}
          </div>
        ))}
        {props.busy && <div className="muted">sampling three candidates…</div>}
      </div>

      {props.candidates && (
        <div className="candidates">
          <h3>Pick a reply — three honest samples, not one “answer”</h3>
          {props.candidates.map((c, i) => (
            <div key={i} className="candidate">
              <div className="cand-head">
                {CAND_LABELS[i]} · {c.off_map
                  ? 'OFF-MAP — ignore the %'
                  : `${c.confidence_pct}% sure`}
              </div>
              <div className="cand-verdict muted">{c.verdict}</div>
              {c.off_map && c.unknown_chars.length > 0 && (
                <div className="cand-verdict muted">never seen: {c.unknown_chars.join(' ')}</div>
              )}
              <div className="cand-text">
                {c.records.filter(r => r.is_continuation).map((r, j) => (
                  <span key={j} style={{ background: entColor(r.entropy) }}>
                    {r.char}
                  </span>
                ))}
              </div>
              <button onClick={() => props.onCommit(c)}>Commit this reply</button>
            </div>
          ))}
        </div>
      )}

      {props.error && <div className="error">{props.error}</div>}

      <div className="composer">
        <textarea value={draft} onChange={e => setDraft(e.target.value)}
          placeholder={props.format === 'chat'
            ? 'Ask something…'
            : 'Text for the model to continue…'} />
        <button disabled={props.busy || !draft}
          onClick={() => { props.onSend(draft); setDraft('') }}>
          Send
        </button>
      </div>

      <div className="controls">
        <label>format{' '}
          <select value={props.format}
            onChange={e => props.onFormat(e.target.value as TapeFormat)}>
            <option value="raw">raw completion</option>
            <option value="chat">chat (user:/model:)</option>
          </select>
        </label>
        <label title="how adventurous the sampling is: 0 = always pick the single most likely next character; higher = give lower-ranked characters a real chance">
          temperature {props.sampling.temperature.toFixed(1)}
          <input type="range" min={0} max={2} step={0.1}
            value={props.sampling.temperature}
            onChange={e => props.onSampling({ ...props.sampling,
              temperature: Number(e.target.value) })} />
        </label>
        <label title="restrict sampling to the k most likely next characters; 'off' means the whole vocabulary stays in play">
          top-k {props.sampling.top_k || 'off'}
          <input type="range" min={0} max={20} step={1}
            value={props.sampling.top_k}
            onChange={e => props.onSampling({ ...props.sampling,
              top_k: Number(e.target.value) })} />
        </label>
        <label title="every chat API cuts the model off at a stop sequence; here is the knob itself">
          stop at line end{' '}
          <input type="checkbox" checked={props.sampling.stop === '\n'}
            onChange={e => props.onSampling({ ...props.sampling,
              stop: e.target.checked ? '\n' : null })} />
        </label>
      </div>
    </main>
  )
}
