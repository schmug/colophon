import type { ModeInfo, SaliencyCell, Sampling, TapeFormat, Turn, TurnResponse } from './types'

async function jsonOrThrow(res: Response) {
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`)
  return data
}

export async function getModes(): Promise<{ default: string; modes: ModeInfo[] }> {
  return jsonOrThrow(await fetch('/api/modes'))
}

export async function postTurn(
  mode: string, turns: Turn[], format: TapeFormat, sampling: Sampling,
): Promise<TurnResponse> {
  return jsonOrThrow(await fetch('/api/turn', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, turns, format, sampling }),
  }))
}

export async function postSaliency(
  mode: string, text: string, pos: number,
): Promise<{ pos: number; window: SaliencyCell[] }> {
  return jsonOrThrow(await fetch('/api/saliency', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, text, pos }),
  }))
}
