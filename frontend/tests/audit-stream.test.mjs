import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import ts from 'typescript'

const source = await readFile(new URL('../src/api.ts', import.meta.url), 'utf8')
const js = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText
const { streamAudit } = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`)
const encoder = new TextEncoder()
function response(parts) {
  return new Response(new ReadableStream({ start(controller) {
    for (const part of parts) controller.enqueue(part)
    controller.close()
  } }))
}

test('delivers progress before completion and handles fragmented Unicode/JSON', async () => {
  const original = globalThis.fetch
  let controller
  globalThis.fetch = async () => new Response(new ReadableStream({ start(value) { controller = value } }))
  const received = []
  try {
    const run = streamAudit('Review', event => received.push(event))
    await new Promise(resolve => setTimeout(resolve, 0))
    const line = encoder.encode(JSON.stringify({ type: 'progress', message: 'Reviewing…' }) + '\n')
    for (const byte of line) controller.enqueue(Uint8Array.of(byte))
    await new Promise(resolve => setTimeout(resolve, 0))
    assert.deepEqual(received, [{ type: 'progress', message: 'Reviewing…' }])
    controller.enqueue(encoder.encode('{"type":"memo_delta","text":"CFO"}\n{"type":"complete","audit":{}}'))
    controller.close()
    await run
    assert.equal(received.at(-1).type, 'complete')
  } finally { globalThis.fetch = original }
})

test('refusals and truncated streams never count as completed audits', async () => {
  const original = globalThis.fetch
  try {
    globalThis.fetch = async () => response([encoder.encode('{"type":"error","message":"Please rephrase."}\n')])
    await assert.rejects(streamAudit('Review', () => {}), /Please rephrase/)
    globalThis.fetch = async () => response([encoder.encode('{"type":"memo_delta","text":"Partial"}\n')])
    await assert.rejects(streamAudit('Review', () => {}), /before the audit completed/)
  } finally { globalThis.fetch = original }
})
