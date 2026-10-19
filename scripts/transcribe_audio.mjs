// Node.js 20+: upload, enqueue, poll and save Qwen3-ASR results using HTTP only.
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const [audioPath, destination, title = ''] = process.argv.slice(2);
if (!audioPath) {
  console.error('Usage: node scripts/transcribe_audio.mjs <audio> [output-directory] [song-title]');
  process.exit(1);
}

const base = new URL(process.env.H3_BASE_URL || 'http://127.0.0.1:8000');
const headers = process.env.H3_API_KEY ? { 'X-API-Key': process.env.H3_API_KEY } : {};
const out = path.resolve(destination || `outputs/transcriptions/${path.parse(audioPath).name}-${Date.now()}`);
const controller = new AbortController();
const timer = setTimeout(() => controller.abort(new Error('Transcription client timed out after 45 minutes')), 45 * 60_000);
const interrupt = () => controller.abort(new Error('Interrupted'));
process.once('SIGINT', interrupt);
process.once('SIGTERM', interrupt);

async function request(resource, options = {}, signal = controller.signal) {
  const url = new URL(resource, base);
  if (url.origin !== base.origin || url.username || url.password) {
    throw new Error('Refusing to send API credentials to another origin');
  }
  const response = await fetch(url, {
    ...options, headers: { ...headers, ...options.headers }, redirect: 'error',
    signal: AbortSignal.any([signal, AbortSignal.timeout(60_000)]),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${(await response.text()).slice(0, 2000)}`);
  return response;
}

let taskId;
let terminal = false;
try {
  const form = new FormData();
  form.set('kind', 'audio');
  form.set('file', new Blob([await readFile(audioPath)]), path.basename(audioPath));
  const uploaded = await (await request('/v1/files', { method: 'POST', body: form })).json();
  const submitted = await (await request('/v1/audio/transcriptions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_id: uploaded.file_id, language: 'zh', title }),
  })).json();
  taskId = submitted.id;
  console.log(`Task ${taskId}; queue ${submitted.queue_number}`);
  let previousStatus;
  while (true) {
    const task = await (await request(`/v1/tasks/${encodeURIComponent(taskId)}`)).json();
    if (task.status !== previousStatus) console.log(`Status: ${task.status}`);
    previousStatus = task.status;
    terminal = ['succeeded', 'failed', 'canceled'].includes(task.status);
    if (task.status === 'failed' || task.status === 'canceled') throw new Error(task.error || task.status);
    if (task.status === 'succeeded') {
      const jsonOutput = task.outputs?.find(item => item.filename.endsWith('.json'));
      const textOutput = task.outputs?.find(item => item.filename.endsWith('.txt'));
      if (!jsonOutput || !textOutput) throw new Error('Task did not return JSON and TXT outputs');
      const result = await (await request(jsonOutput.download_url)).json();
      const text = await (await request(textOutput.download_url)).text();
      if (result.timing !== 'chunk_boundaries_only' || typeof result.text !== 'string') {
        throw new Error('Unexpected transcription schema');
      }
      await mkdir(out, { recursive: true });
      await writeFile(path.join(out, 'qwen3-asr.json'), JSON.stringify(result, null, 2) + '\n', 'utf8');
      await writeFile(path.join(out, 'qwen3-asr.txt'), text, 'utf8');
      console.log(`Saved: ${out}\nAudio: ${result.duration_seconds}s; inference: ${result.elapsed_seconds}s\n${result.text}`);
      break;
    }
    await delay(1500, undefined, { signal: controller.signal });
  }
} catch (error) {
  console.error(error.message);
  if (taskId && !terminal) {
    try {
      await request(`/v1/tasks/${encodeURIComponent(taskId)}`, { method: 'DELETE' }, AbortSignal.timeout(10_000));
    } catch (cancelError) {
      console.error(`Could not cancel ${taskId}: ${cancelError.message}`);
    }
  }
  process.exitCode = 1;
} finally {
  clearTimeout(timer);
  process.removeListener('SIGINT', interrupt);
  process.removeListener('SIGTERM', interrupt);
}
