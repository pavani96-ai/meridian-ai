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

export type AuditEvent =
  | { type: 'progress'; message: string }
  | { type: 'specialist'; key: 'risk_result' | 'tax_result' | 'control_result'; report: string }
  | { type: 'memo_delta'; text: string }
  | { type: 'complete'; audit: Audit }
  | { type: 'error'; code: string; message: string }

export async function streamAudit(request: string, onEvent: (event: AuditEvent) => void, signal?: AbortSignal): Promise<void> {
  const response = await fetch('/api/agent/audit/stream', { ...jsonPost({ request_text: request }), signal })
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new Error(typeof payload?.detail === 'string' ? payload.detail : `Request failed (${response.status}).`)
  }
  if (!response.body) throw new Error('Streaming is unavailable in this browser.')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let pending = ''
  let complete = false
  const consume = (line: string) => {
    if (!line.trim()) return
    if (complete) throw new Error('Unexpected event after audit completion.')
    const event = JSON.parse(line) as AuditEvent
    if (event.type === 'error') throw new Error(event.message)
    onEvent(event)
    if (event.type === 'complete') complete = true
  }
  try {
    while (true) {
      const { value, done } = await reader.read()
      pending += decoder.decode(value, { stream: !done })
      let end: number
      while ((end = pending.indexOf('\n')) >= 0) {
        consume(pending.slice(0, end))
        pending = pending.slice(end + 1)
      }
      if (done) break
    }
    consume(pending)
    if (!complete) throw new Error('The connection ended before the audit completed. Please try again.')
  } finally {
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}
