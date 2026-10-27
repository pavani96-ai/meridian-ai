import { test, expect } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({ json: { status: 'ok', mode: 'local' } }))
  await page.route('**/api/rag/uploads', route => route.fulfill({ json: { uploads: [] } }))
})

test('audit request, structured reports, and CFO memo', async ({ page }) => {
  let requestText = ''
  await page.route('**/api/agent/audit', async route => {
    requestText = route.request().postDataJSON().request_text
    await route.fulfill({ json: { risk_result: JSON.stringify({ overall_risk_verdict: 'CONDITIONAL APPROVAL', credit_score: 60 }), tax_result: JSON.stringify({ treasury_verdict: 'MANUAL APPROVAL REQUIRED' }), control_result: JSON.stringify({ control_verdict: 'FLAGGED FOR CONTROLLING REVIEW' }), cfo_memo: 'ALDERMOOR INDUSTRIES — CFO AUDIT MEMORANDUM\nFINAL DECISION: CONDITIONAL HOLD' } })
  })
  await page.goto('/')
  await expect(page.getByRole('button', { name: 'Run procurement audit' })).toBeDisabled()
  await page.getByRole('button', { name: 'Use an example' }).click()
  await page.getByRole('button', { name: 'Run procurement audit' }).click()
  await expect(page.getByText('FINAL DECISION: CONDITIONAL HOLD', { exact: false })).toBeVisible()
  expect(requestText).toContain('38,000 EUR')
  await page.getByRole('tab', { name: 'Risk', exact: true }).click()
  await expect(page.getByText('credit score', { exact: true })).toBeVisible()
  await page.getByRole('tab', { name: 'CFO memorandum' }).click()
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download memo' }).click()
  expect((await download).suggestedFilename()).toBe('aldermoor-cfo-memorandum.txt')
})

test('audit error is visible and can be retried', async ({ page }) => {
  await page.route('**/api/agent/audit', route => route.fulfill({ status: 503, json: { detail: 'Gemini service unavailable' } }))
  await page.goto('/')
  await page.getByLabel('Purchase request', { exact: true }).fill('Review a server purchase')
  await page.getByRole('button', { name: 'Run procurement audit' }).click()
  await expect(page.getByRole('alert')).toContainText('Gemini service unavailable')
  await expect(page.getByRole('button', { name: 'Run procurement audit' })).toBeEnabled()
})

test('PDF upload, RAG retrieval, and stored document ingestion', async ({ page }) => {
  await page.route('**/api/rag/upload', route => route.fulfill({ json: { results: [{ filename: 'policy.pdf', status: 'ingested', pages: 2, chunks: 4 }] } }))
  await page.route('**/api/rag/ask', async route => {
    expect(route.request().postDataJSON()).toEqual({ query: 'What approvals are required?', retriever_type: 'multiquery' })
    await route.fulfill({ json: { answer: 'Department approval is required. Source: policy.pdf.' } })
  })
  await page.route('**/api/rag/ingest', route => route.fulfill({ json: { status: 'Ingestion complete (local mode)' } }))
  await page.goto('/')
  await page.getByRole('button', { name: 'Document intelligence', exact: true }).click()
  await page.getByLabel('Select PDF documents').setInputFiles({ name: 'policy.pdf', mimeType: 'application/pdf', buffer: Buffer.from('%PDF-1.4 mocked PDF') })
  await page.getByRole('button', { name: 'Upload & index 1 file(s)' }).click()
  await expect(page.getByRole('cell', { name: 'policy.pdf' })).toBeVisible()
  await page.getByLabel('Your question').fill('What approvals are required?')
  await page.getByLabel('Retrieval strategy').selectOption('multiquery')
  await page.getByRole('button', { name: 'Ask a question' }).click()
  await expect(page.getByText('Department approval is required. Source: policy.pdf.')).toBeVisible()
  await page.getByRole('button', { name: 'Index stored PDFs' }).click()
  await expect(page.getByText('Ingestion complete (local mode)')).toBeVisible()
})

test('mobile layout, navigation, and system status', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.route('**/api/status', route => route.fulfill({ json: { mode: 'local', backend: { status: 'healthy', uptime: '00h 01m 10s', python_version: '3.12', platform: 'Linux' }, models: { llm: 'gemini-3.8-flash', embedding: 'gemini-embedding-001', llm_framework: 'Gemini via LangChain' }, storage: { location: '/data/uploads' }, vector_search: { backend: 'Chroma (local disk)' }, ingestion: { uploads_this_session: 0 } } }))
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Clarity before commitment.' })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await page.getByRole('button', { name: 'System status', exact: true }).click()
  await expect(page.getByText('gemini-3.8-flash', { exact: true })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
})
