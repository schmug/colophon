import { useEffect, useMemo, useState } from 'react'
import { getModes, postSaliency, postTurn } from './api'
import { ACTS } from './acts'
import { buildTape } from './tapeUtils'
import { CharInspector } from './components/CharInspector'
import { ConversationPanel } from './components/ConversationPanel'
import { TapePanel } from './components/TapePanel'
import type { CharRecord, ModeInfo, SaliencyCell, Sampling, TapeFormat, Turn, TurnResponse } from './types'

let nextId = 1
const newId = () => String(nextId++)

export interface ModelTurnData {
  records: CharRecord[]
  response: TurnResponse
}

/** Tape position of the focused generated char, in the coordinates of the
 *  response it belongs to (its tape + continuation). */
function focusedTapePos(
  focus: { turnId: string; index: number } | null,
  modelData: Record<string, ModelTurnData>,
): number | null {
  if (!focus) return null
  const d = modelData[focus.turnId]
  return d ? d.response.tape.length + focus.index : null
}

export default function App() {
  const [modes, setModes] = useState<ModeInfo[]>([])
  const [modeId, setModeId] = useState<string | null>(null)
  const [format, setFormat] = useState<TapeFormat>('raw')
  const [turns, setTurns] = useState<Turn[]>([])
  const [modelData, setModelData] = useState<Record<string, ModelTurnData>>({})
  const [sampling, setSampling] = useState<Sampling>({
    temperature: 0.8, top_k: 0, seed: 0, max_chars: 160,
    stop: null, banned_chars: [],
  })
  const [candidates, setCandidates] = useState<TurnResponse[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [act, setAct] = useState<number | null>(null)
  const [focus, setFocus] = useState<{ turnId: string; index: number } | null>(null)
  const [saliency, setSaliency] = useState<SaliencyCell[] | null>(null)

  const mode = useMemo(() => modes.find(m => m.id === modeId) ?? null,
    [modes, modeId])

  useEffect(() => {
    getModes().then(d => {
      setModes(d.modes)
      const start = d.modes.find(m => m.id === d.default && m.available)
        ?? d.modes.find(m => m.available) ?? null
      if (start) { setModeId(start.id); setFormat(start.format_default) }
      else setError('No trained model found — see the mode buttons for the training commands.')
    }).catch(e => setError(String(e)))
  }, [])

  const reset = () => {
    setTurns([]); setModelData({}); setCandidates(null)
    setFocus(null); setSaliency(null); setError('')
  }

  const switchMode = (m: ModeInfo) => {
    setModeId(m.id); setFormat(m.format_default); setAct(null); reset()
  }

  const startAct = (n: number) => {
    const a = ACTS.find(x => x.n === n)!
    const m = modes.find(x => x.id === a.mode)
    if (!m?.available) {
      setError(`Act ${n} needs the '${a.mode}' model — ${m?.train_hint ?? 'train it first'}`)
      return
    }
    setModeId(a.mode); setFormat(a.format); setAct(n); reset()
  }

  const send = async (text: string) => {
    if (!modeId) return
    const userTurn: Turn = { id: newId(), role: 'user', text, excluded: false }
    const sent = [...turns, userTurn]
    setTurns(sent); setBusy(true); setError(''); setCandidates(null)
    // Three candidates: cooler / as-set / hotter. Each is a separate honest
    // sample -- three real draws from the distribution, not one "answer".
    const temps = [Math.max(0.1, sampling.temperature - 0.5),
      sampling.temperature, sampling.temperature + 0.5]
    try {
      const results = await Promise.all(temps.map((temperature, i) =>
        postTurn(modeId, sent, format,
          { ...sampling, temperature, seed: sampling.seed + i })))
      setCandidates(results)
    } catch (e) {
      setError(String(e))
      setTurns(turns) // roll the user turn back so a failed send isn't stuck in history
    } finally {
      setBusy(false)
    }
  }

  const commitCandidate = (r: TurnResponse) => {
    const id = newId()
    setTurns(ts => [...ts, { id, role: 'model', text: r.continuation, excluded: false }])
    setModelData(md => ({ ...md, [id]: { records: r.records, response: r } }))
    setCandidates(null); setFocus(null); setSaliency(null)
  }

  const toggleTurn = (id: string) =>
    setTurns(ts => ts.map(t => t.id === id ? { ...t, excluded: !t.excluded } : t))

  const editTurn = (id: string, text: string) =>
    setTurns(ts => ts.map(t => t.id === id ? { ...t, text } : t))

  const focusChar = async (turnId: string, index: number) => {
    setFocus({ turnId, index }); setSaliency(null)
    const data = modelData[turnId]
    if (!data || !modeId) return
    // Saliency runs over the exact text of THAT response (tape+continuation)
    // at the record's position -- nothing is re-generated server-side.
    const fullText = data.response.tape + data.response.continuation
    try {
      const s = await postSaliency(modeId, fullText,
        data.response.tape.length + index)
      setSaliency(s.window)
    } catch { /* saliency is best-effort; the record is already on screen */ }
  }

  const tape = useMemo(() => buildTape(turns, format), [turns, format])
  const lastModelTurn = [...turns].reverse().find(t => t.role === 'model')
  const focusedResponse = focus
    ? modelData[focus.turnId]?.response ?? null
    : lastModelTurn ? modelData[lastModelTurn.id]?.response ?? null : null
  const focusedRecord = focus
    ? modelData[focus.turnId]?.records.filter(r => r.is_continuation)[focus.index] ?? null
    : null

  return (
    <div className="app">
      <TapePanel
        turns={turns} format={format} tape={tape} mode={mode} modes={modes}
        K={mode?.K ?? focusedResponse?.K ?? null}
        focusPos={focusedTapePos(focus, modelData)}
        onSwitchMode={switchMode} onToggleTurn={toggleTurn}
        onEditTurn={editTurn}
      />
      <ConversationPanel
        turns={turns} modelData={modelData} busy={busy} error={error}
        candidates={candidates} act={act} acts={ACTS} format={format}
        sampling={sampling} focus={focus}
        onSend={send} onCommit={commitCandidate} onStartAct={startAct}
        onFreePlay={() => setAct(null)}
        onSampling={setSampling} onFormat={setFormat} onFocusChar={focusChar}
      />
      <CharInspector
        record={focusedRecord} saliency={saliency} response={focusedResponse}
        recordSets={Object.values(modelData).map(d => d.records)}
        sampling={sampling} onSampling={setSampling} mode={mode}
      />
    </div>
  )
}
