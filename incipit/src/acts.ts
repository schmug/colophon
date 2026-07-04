import type { TapeFormat } from './types'

export interface Act {
  n: number
  title: string
  mode: string
  format: TapeFormat
  prompt: string
  copy: string
}

export const ACTS: Act[] = [
  {
    n: 1,
    title: "It's just a tape",
    mode: 'elements64',
    format: 'raw',
    prompt: 'number: 26\n',
    copy: 'Send this and watch the model continue the YAML it was trained ' +
      'on. Then look left: your "conversation" is one growing string, and ' +
      'the highlighted band is the ONLY part the model can see. Keep going ' +
      'until early turns slide out of the window -- the model has no memory ' +
      'except that band.',
  },
  {
    n: 2,
    title: 'Chat is a format it never saw',
    mode: 'elements64',
    format: 'chat',
    prompt: 'what is element 26?',
    copy: 'Same model, same K=64 window -- but your turn is now wrapped as ' +
      '"user: ... / model: ...". It was trained on YAML, not conversations, ' +
      'so it completes YAML-ish noise. The "?" is not even in its ' +
      'vocabulary -- watch the off-map flag. The failure is the training ' +
      "data's format, not the window: the next act uses the identical " +
      'architecture.',
  },
  {
    n: 3,
    title: 'Train on dialogue, get a chatbot',
    mode: 'dialogue',
    format: 'chat',
    prompt: 'what is element 26?',
    copy: 'Identical architecture (K=64, ~2M params), same 118 facts -- but ' +
      'the training data is user:/model: dialogues. Now chat format ' +
      'completes correctly. Try the follow-up "which period is it in?" and ' +
      'check the saliency panel: the answer depends on the earlier turn ' +
      'still being inside the window. That is the whole trick: a chatbot ' +
      'is a completion model whose training data contains conversations.',
  },
]
