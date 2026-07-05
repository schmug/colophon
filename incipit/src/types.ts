// Mirrors incipit.py's JSON contract exactly. If a field changes there,
// it changes here -- the server is the source of truth.

export type Role = 'user' | 'model'
export type TapeFormat = 'raw' | 'chat'

export interface Turn {
  id: string
  role: Role
  text: string
  excluded: boolean
}

export interface CharRecord {
  char: string
  display: string
  is_continuation: boolean
  entropy: number
  top_k: [string, number][]
  context_window: string[]
  context_types: string[]
  truth_rank: number | null
  truth_prob: number | null
  off_map: boolean
}

export interface Sampling {
  temperature: number
  top_k: number
  seed: number
  max_chars: number
  stop: string | null
  banned_chars: string[]
}

export interface SourceEcho {
  matched: boolean
  file?: string
  line?: number
  pre?: string
  match?: string
  post?: string
}

export interface TurnResponse {
  tape: string
  continuation: string
  records: CharRecord[]
  K: number
  entropy: number
  unknown_chars: string[]
  off_map: boolean
  banned_applied: string[]
  confidence_pct: number | null
  verdict_level: string
  verdict: string
  source: SourceEcho
  format: TapeFormat
}

export interface ModeInfo {
  id: string
  label: string
  blurb: string
  acts: number[]
  format_default: TapeFormat
  available: boolean
  train_hint: string
  K: number | null
  params: number | null
}

export interface SaliencyCell {
  char: string
  display: string
  delta: number
  is_pad: boolean
}
