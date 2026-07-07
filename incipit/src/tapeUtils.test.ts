import { describe, expect, it } from 'vitest'
import { buildTape, buildTapeCells, fellOffBefore, median, sessionStats, windowSpan } from './tapeUtils'
import type { CharRecord, Turn } from './types'

const t = (id: string, role: 'user' | 'model', text: string, excluded = false): Turn =>
  ({ id, role, text, excluded })

describe('buildTape', () => {
  it('raw format concatenates active turn texts directly', () => {
    expect(buildTape([t('1', 'user', 'number: 26\n'), t('2', 'model', 'symbol: Fe\n')], 'raw'))
      .toBe('number: 26\nsymbol: Fe\n')
  })
  it('chat format adds role markers and a generation prefix', () => {
    expect(buildTape([t('1', 'user', 'what is element 26?')], 'chat'))
      .toBe('user: what is element 26?\nmodel: ')
  })
  it('excluded turns vanish from the tape -- the model has no other memory', () => {
    expect(buildTape([t('1', 'user', 'AAA', true), t('2', 'user', 'BBB')], 'raw')).toBe('BBB')
  })
})

describe('buildTapeCells', () => {
  it('cells spell exactly the tape string, char for char', () => {
    const turns = [t('1', 'user', 'hi?'), t('2', 'model', 'hello.')]
    for (const fmt of ['raw', 'chat'] as const) {
      expect(buildTapeCells(turns, fmt).map(c => c.char).join('')).toBe(buildTape(turns, fmt))
    }
  })
  it('marker chars carry no turnId; text chars carry their turn', () => {
    const cells = buildTapeCells([t('7', 'user', 'x')], 'chat')
    expect(cells.filter(c => c.role === 'marker').every(c => c.turnId === null)).toBe(true)
    expect(cells.find(c => c.char === 'x')!.turnId).toBe('7')
  })
})

describe('windowSpan', () => {
  it('covers the K chars before pos, clamped at the tape start', () => {
    expect(windowSpan(10, 4)).toEqual({ start: 6, end: 10 })
    expect(windowSpan(2, 12)).toEqual({ start: 0, end: 2 })
  })
})

describe('fellOffBefore', () => {
  it('counts the cells the newest prediction can no longer see', () => {
    expect(fellOffBefore(100, 64)).toBe(36)
  })
  it('is 0 while the whole tape still fits in the window', () => {
    expect(fellOffBefore(10, 64)).toBe(0)
    expect(fellOffBefore(64, 64)).toBe(0)
  })
  it('is 0 when K is unknown', () => {
    expect(fellOffBefore(100, null)).toBe(0)
  })
})

describe('sessionStats', () => {
  const rec = (entropy: number, off = false): CharRecord => ({
    char: 'a', display: 'a', is_continuation: true, entropy, top_k: [],
    context_window: [], context_types: [], truth_rank: 1, truth_prob: 1, off_map: off,
  })
  it('aggregates entropy and off-map counts across turns', () => {
    const stats = sessionStats([[rec(0.2), rec(0.4, true)], [rec(0.6)]])
    expect(stats.turnCount).toBe(2)
    expect(stats.charCount).toBe(3)
    expect(stats.medianEntropy).toBeCloseTo(0.4)
    expect(stats.offMapCount).toBe(1)
  })
  it('median of empty is 0', () => { expect(median([])).toBe(0) })
})
