import type { CharRecord, TapeFormat, Turn } from './types'

export interface TapeCell {
  char: string
  turnId: string | null  // null for format scaffold ('role: ' markers, generation prefix)
  role: 'user' | 'model' | 'marker'
}

/** Mirror of the server's build_tape(): the exact string the model sees.
 *  Kept in lockstep with incipit.py -- the server's `tape` response field is
 *  the source of truth; this exists so the tape panel can render before a
 *  response arrives and color chars by provenance. */
export function buildTape(turns: Turn[], format: TapeFormat): string {
  const active = turns.filter(t => !t.excluded)
  if (format === 'chat') {
    return active.map(t => `${t.role}: ${t.text}\n`).join('') + 'model: '
  }
  return active.map(t => t.text).join('')
}

/** Per-char provenance for the tape: which turn (or format scaffold) each
 *  character came from. Invariant (vitest-pinned): cells spell buildTape(). */
export function buildTapeCells(turns: Turn[], format: TapeFormat): TapeCell[] {
  const cells: TapeCell[] = []
  const active = turns.filter(t => !t.excluded)
  for (const t of active) {
    if (format === 'chat') {
      for (const ch of `${t.role}: `) cells.push({ char: ch, turnId: null, role: 'marker' })
    }
    for (const ch of t.text) cells.push({ char: ch, turnId: t.id, role: t.role })
    if (format === 'chat') cells.push({ char: '\n', turnId: null, role: 'marker' })
  }
  if (format === 'chat') {
    for (const ch of 'model: ') cells.push({ char: ch, turnId: null, role: 'marker' })
  }
  return cells
}

/** Tape positions [start, end) inside the model's K-char window when it
 *  predicted the character at tape position `pos` (record i's context is the
 *  K characters ending just before position i; left-pad clamps at 0). */
export function windowSpan(pos: number, K: number): { start: number; end: number } {
  return { start: Math.max(0, pos - K), end: pos }
}

export function median(xs: number[]): number {
  if (!xs.length) return 0
  const s = [...xs].sort((a, b) => a - b)
  const mid = Math.floor(s.length / 2)
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2
}

export interface SessionStats {
  turnCount: number
  charCount: number
  medianEntropy: number
  offMapCount: number
}

/** Aggregates over every committed model turn's per-char records. */
export function sessionStats(recordSets: CharRecord[][]): SessionStats {
  const all = recordSets.flat()
  return {
    turnCount: recordSets.length,
    charCount: all.length,
    medianEntropy: median(all.map(r => r.entropy)),
    offMapCount: all.filter(r => r.off_map).length,
  }
}
