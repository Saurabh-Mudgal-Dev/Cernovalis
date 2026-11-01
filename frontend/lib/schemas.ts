import { z } from 'zod'

export const featureContributionSchema = z.object({
  feature: z.string(),
  value: z.number(),
  impact: z.enum(['low', 'medium', 'high']),
})

export const modelContributionSchema = z.object({
  prediction: z.enum(['REAL', 'FAKE']),
  confidence: z.number(),
})

export const sourceCredibilitySchema = z.object({
  domain: z.string().nullable(),
  score: z.number().nullable(),
  label: z.enum(['RELIABLE', 'MIXED', 'UNRELIABLE', 'UNKNOWN']),
  known_flags: z.array(z.string()),
})

export const explanationSchema = z.object({
  top_features: z.array(featureContributionSchema),
  linguistic_flags: z.array(z.string()),
  model_contributions: z.record(modelContributionSchema),
})

export const analyzeResponseSchema = z.object({
  id: z.string().uuid(),
  timestamp: z.string(),
  input_type: z.enum(['text', 'url']),
  original_input: z.string(),
  extracted_text: z.string().nullable(),
  verdict: z.enum(['REAL', 'FAKE']),
  confidence: z.number(),
  risk_level: z.enum(['LOW', 'MEDIUM', 'HIGH']),
  source_credibility: sourceCredibilitySchema,
  explanation: explanationSchema,
})

export const batchAnalysisItemSchema = z.object({
  row: z.number(),
  result: analyzeResponseSchema.nullable(),
  error: z.string().nullable(),
})

export const batchResponseSchema = z.object({
  total: z.number(),
  succeeded: z.number(),
  failed: z.number(),
  results: z.array(batchAnalysisItemSchema),
})

export const healthSchema = z.object({
  status: z.string(),
  models: z
    .array(z.object({ name: z.string(), loaded: z.boolean(), detail: z.string().nullable() }))
    .optional(),
  last_loaded_at: z.string().nullable().optional(),
})

export type FeatureContribution = z.infer<typeof featureContributionSchema>
export type ModelContribution = z.infer<typeof modelContributionSchema>
export type SourceCredibility = z.infer<typeof sourceCredibilitySchema>
export type AnalysisExplanation = z.infer<typeof explanationSchema>
export type AnalyzeResult = z.infer<typeof analyzeResponseSchema>
export type BatchAnalysisItem = z.infer<typeof batchAnalysisItemSchema>
export type BatchResult = z.infer<typeof batchResponseSchema>
export type Health = z.infer<typeof healthSchema>
