export interface Audit { risk_result: string; tax_result: string; control_result: string; cfo_memo: string }
export interface UploadResult { filename: string; status: string; reason?: string; pages?: number; chunks?: number }
export interface SystemStatus { mode: string; backend: { status: string; uptime: string; python_version: string; platform: string }; models: { llm: string; embedding: string; llm_framework: string }; storage: { location: string }; vector_search: { backend: string }; ingestion: { uploads_this_session: number } }
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, init)
  const payload: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = payload && typeof payload === 'object' && 'detail' in payload ? payload.detail : null
    const message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((item: { msg?: string }) => item.msg || 'Invalid input').join('; ') : `Request failed (${response.status}).`
    throw new Error(message)
  }
  return payload as T
}
export const jsonPost = (value: unknown): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) })
export function report(raw: string): Record<string, unknown> | null { try { const value = JSON.parse(raw); return value && typeof value === 'object' && !Array.isArray(value) ? value : null } catch { return null } }
