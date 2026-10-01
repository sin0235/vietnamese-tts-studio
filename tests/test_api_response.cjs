// Run: node tests/test_api_response.cjs
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');

function readFunction(source, name) {
  const start = source.indexOf(`  async function ${name}(`);
  assert.ok(start >= 0, `Missing ${name}`);
  const end = source.indexOf('\n  }', start) + 4;
  return source.slice(start, end);
}

async function main() {
  for (const template of ['home.html', 'studio_engine.html']) {
    const source = readFileSync(resolve(__dirname, '../webapp/templates', template), 'utf8');
    const parseApiResponse = new Function('MAX_UPLOAD_MB',
      `${readFunction(source, 'parseApiResponse')}; return parseApiResponse;`)(32);
    for (const [body, status, contentType, expected] of [
      ['{"ok":true,"done":false,"status":"queued","queue_position":1}', 200, 'application/json',
        { ok: true, done: false, status: 'queued', queue_position: 1 }],
      ['{"ok":false,"error":"Worker lỗi"}', 503, 'application/json',
        { ok: false, error: 'Worker lỗi' }],
      ['Bad gateway', 502, 'application/json', { ok: false, error: 'Bad gateway' }],
      ['<html>Gateway error</html>', 502, 'application/json', { ok: false, error: 'API trả về lỗi 502.' }],
      ['', 200, 'application/json', { ok: false, error: 'API trả về dữ liệu không hợp lệ.' }],
      ['<html>Too large</html>', 413, 'text/html',
        { ok: false, error: 'File upload quá lớn. Giới hạn hiện tại là 32 MB.' }],
    ]) {
      const response = new Response(body, { status, headers: { 'content-type': contentType } });
      assert.deepEqual(await parseApiResponse(response), expected, template);
    }

    if (template === 'studio_engine.html') {
      const pollGenerationJob = new Function('parseApiResponse', 'fetch', 'delayWithAbort',
        'updatePendingHistoryStatus', `${readFunction(source, 'pollGenerationJob')}; return pollGenerationJob;`)(
        parseApiResponse,
        async () => new Response('<html>Unexpected page</html>', { headers: { 'content-type': 'text/html' } }),
        async () => { throw new Error('Polling continued after an invalid response'); },
        () => {},
      );
      await assert.rejects(pollGenerationJob('job-1'), /API trả về dữ liệu không hợp lệ/);

      const replies = [
        { ok: true, done: false, status: 'queued', queue_position: 1 },
        { ok: true, done: false, status: 'running', gpu_index: 0 },
        { ok: true, done: true, status: 'completed', result: { ok: true, audio_url: '/outputs/test.wav' } },
      ];
      const labels = [];
      const pollUntilComplete = new Function('parseApiResponse', 'fetch', 'delayWithAbort',
        'updatePendingHistoryStatus', `${readFunction(source, 'pollGenerationJob')}; return pollGenerationJob;`)(
        parseApiResponse,
        async () => {
          assert.ok(replies.length, 'Polling continued after completion');
          return new Response(JSON.stringify(replies.shift()), { headers: { 'content-type': 'application/json' } });
        },
        async () => {},
        label => labels.push(label),
      );
      assert.deepEqual(await pollUntilComplete('job-1'), { ok: true, audio_url: '/outputs/test.wav' });
      assert.deepEqual(labels, ['Đang chờ, vị trí 1...', 'Đang sinh trên GPU 0...']);
    }
  }
  console.log('API response and polling regression checks passed.');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
