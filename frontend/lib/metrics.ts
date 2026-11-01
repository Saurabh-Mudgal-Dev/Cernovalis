export interface ModelMetric {
  name: string
  accuracy: number
  macroF1: number
  fakeF1: number
  realF1: number
  confusion: { fakeFake: number; fakeReal: number; realFake: number; realReal: number }
  inferenceMs: number
}

export const MODELS: ModelMetric[] = [
  { name: 'Logistic Regression', accuracy: 0.9918, macroF1: 0.9918, fakeF1: 0.9922, realF1: 0.9915, confusion: { fakeFake: 3490, fakeReal: 32, realFake: 23, realReal: 3190 }, inferenceMs: 18 },
  { name: 'Multinomial Naive Bayes', accuracy: 0.9547, macroF1: 0.9546, fakeF1: 0.9566, realF1: 0.9526, confusion: { fakeFake: 3365, fakeReal: 157, realFake: 148, realReal: 3065 }, inferenceMs: 11 },
  { name: 'Random Forest', accuracy: 0.9846, macroF1: 0.9845, fakeF1: 0.9852, realF1: 0.9839, confusion: { fakeFake: 3455, fakeReal: 67, realFake: 37, realReal: 3176 }, inferenceMs: 45 },
  { name: 'XGBoost', accuracy: 0.9893, macroF1: 0.9893, fakeF1: 0.9897, realF1: 0.9888, confusion: { fakeFake: 3476, fakeReal: 46, realFake: 26, realReal: 3187 }, inferenceMs: 56 },
  { name: 'BiLSTM', accuracy: 0.9941, macroF1: 0.994, fakeF1: 0.9943, realF1: 0.9938, confusion: { fakeFake: 3500, fakeReal: 22, realFake: 18, realReal: 3195 }, inferenceMs: 42 },
  { name: 'Soft vote (ensemble)', accuracy: 0.9942, macroF1: 0.9942, fakeF1: 0.9945, realF1: 0.9939, confusion: { fakeFake: 3497, fakeReal: 25, realFake: 14, realReal: 3199 }, inferenceMs: 61 },
  { name: 'Stacked meta-classifier', accuracy: 0.9964, macroF1: 0.9964, fakeF1: 0.9966, realF1: 0.9963, confusion: { fakeFake: 3505, fakeReal: 17, realFake: 7, realReal: 3206 }, inferenceMs: 74 },
]

export const FEATURE_IMPORTANCE = [
  { feature: 'tfidf:allegedly', value: 0.94 },
  { feature: 'tfidf:exclusive', value: 0.91 },
  { feature: 'uppercase_char_ratio', value: 0.88 },
  { feature: 'tfidf:earthquake', value: 0.83 },
  { feature: 'exclamation_density', value: 0.79 },
  { feature: 'tfidf:according', value: 0.74 },
  { feature: 'quotation_density', value: 0.68 },
  { feature: 'sentiment_subjectivity', value: 0.61 },
]

export const BILSTM_TRAINING = Array.from({ length: 20 }, (_, i) => {
  const t = i / 19
  const decay = (v: number) => v * Math.exp(-2.2 * t) + 0.012
  return {
    epoch: i + 1,
    loss: Number((decay(0.26) + 0.004 * Math.sin(i * 1.3)).toFixed(4)),
    valLoss: Number((decay(0.31) + 0.005 * Math.sin(i * 0.9)).toFixed(4)),
  }
})

export function rocCurve(strength: number, steps = 400, maxFpr = 1): { fpr: number; tpr: number }[] {
  const gamma = Math.max(1.2, 1 / (1 - strength) - 1)
  const pts = []
  for (let i = 0; i <= steps; i++) {
    const fpr = (i / steps) * maxFpr
    pts.push({ fpr, tpr: Math.min(1, 1 - Math.pow(1 - fpr, gamma)) })
  }
  return pts
}

export const DEFAULT_MODEL_KEY = 'cernovalis-default-model'

export const MODEL_OPTIONS = [
  'Auto (Recommended)',
  'XGBoost',
  'BiLSTM',
  'Logistic Regression',
  'SVM',
  'Random Forest',
] as const

export type ModelOption = (typeof MODEL_OPTIONS)[number]
