let busy = false;
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const readPermissions = {origins: ['https://*/*', 'http://*/*']};
// 本机 MiniMax H3 服务的采集协议地址（默认端口 8000，与 H3_PORT 一致）。安装后无需配对。
const API = 'http://127.0.0.1:8000/api/collector';
async function serviceRequest(suffix = '', init = {}) {
  let response;
  try {response = await fetch(API + suffix, {...init, headers: {'X-Collector-Version': chrome.runtime.getManifest().version, 'X-Collector-Capture': 'true', ...init.headers}, signal: AbortSignal.timeout(suffix === '?mode=media' ? 30000 : 10000)});}
  catch {throw Error('采集服务没有运行。请先启动 MiniMax H3 本地服务（python -m mediaforge_service 或 start.cmd），然后重试。');}
  let payload;
  try {payload = await response.json();} catch {throw Error('端口 8000 上没有返回采集服务数据，请确认启动的是 MiniMax H3 本地服务。');}
  if (!response.ok) throw Error(payload.error || '采集服务拒绝了请求。');
  return payload;
}
async function connectBrowser(quiet = false) {
  try {
    if (!await chrome.permissions.contains(readPermissions)) throw Error('尚未获得网页读取权限，请点击“开启”，并在浏览器提示中允许。');
    await serviceRequest('?mode=ping');
    const message = '连接成功：采集服务在线，网页读取权限已授予。';
    await chrome.storage.local.set({connection: 'connected', lastConnectedAt: Date.now(), ...(!quiet ? {status: message} : {})});
    return {ok: true, message};
  } catch (error) {
    await chrome.storage.local.set({connection: 'error', status: error.message});
    return {ok: false, error: error.message};
  }
}
function publicPage(value) {
  const url = new URL(value), host = url.hostname.toLowerCase();
  return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password &&
    !['localhost', '0.0.0.0', '::1', '[::1]'].includes(host) && !host.endsWith('.local') && !host.endsWith('.localhost') &&
    !/^(127\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.)/.test(host);
}

// Observe only responses already loaded by this task's tab. Never fetch a media URL.
async function capturePageResponses(tabId) {
  const target = {tabId}, requests = new Map(), resources = new Map(), pending = new Set();
  let total = 0, stopped = false, attached = false, error = '';
  const header = (headers, key) => String(Object.entries(headers || {}).find(([name]) => name.toLowerCase() === key)?.[1] || '');
  const onEvent = (source, method, params) => {
    if (source.tabId !== tabId || stopped) return;
    if (method === 'Network.responseReceived') {
      const response = params.response;
      if (requests.size >= 40 || !['Image', 'Media'].includes(params.type) || !publicPage(response.url) || !/^(image|video)\//.test(response.mimeType)) return;
      if (![200, 206].includes(response.status)) return;
      const kind = response.mimeType.startsWith('image/') ? 'image' : 'video', limit = (kind === 'image' ? 12 : 32) * 1024 * 1024;
      const length = Number(header(response.headers, 'content-length'));
      if (length > limit) return;
      requests.set(params.requestId, {url: response.url, mime: response.mimeType, kind, limit, responseStatus: response.status, contentRange: header(response.headers, 'content-range')});
    }
    if (method === 'Network.loadingFailed') requests.delete(params.requestId);
    if (method !== 'Network.loadingFinished') return;
    const response = requests.get(params.requestId); if (!response) return;
    requests.delete(params.requestId);
    if (params.encodedDataLength > response.limit || total >= 64 * 1024 * 1024) return;
    const task = (async () => {
      try {
        const body = await chrome.debugger.sendCommand(target, 'Network.getResponseBody', {requestId: params.requestId});
        if (stopped) return;
        if (body.body.length > Math.ceil(response.limit * 4 / 3) + 4) return;
        const bytes = body.base64Encoded ? Uint8Array.from(atob(body.body), character => character.charCodeAt(0)) : new TextEncoder().encode(body.body);
        if (!bytes.length || bytes.length > response.limit || total + bytes.length > 64 * 1024 * 1024) return;
        if (response.responseStatus === 206) {
          const range = /^bytes 0-(\d+)\/(\d+)$/.exec(response.contentRange);
          if (!range || Number(range[1]) + 1 !== Number(range[2]) || Number(range[2]) !== bytes.length) return;
        }
        total += bytes.length; resources.set(response.url, {...response, bytes});
      } catch {error = '部分响应已被浏览器释放或未完整加载，未重新请求资源链接';}
    })();
    pending.add(task); void task.finally(() => pending.delete(task));
  };
  const onDetach = source => {if (source.tabId === tabId) {stopped = true; error = '页面资源捕获已中断，请重新读取';}};
  const stop = async () => {
    stopped = true;
    chrome.debugger.onEvent.removeListener(onEvent); chrome.debugger.onDetach.removeListener(onDetach);
    if (attached) {try {await chrome.debugger.detach(target);} catch { /* User or browser may have detached already. */ }}
    requests.clear(); resources.clear();
  };
  try {
    await chrome.debugger.attach(target, '1.3'); attached = true;
    chrome.debugger.onEvent.addListener(onEvent); chrome.debugger.onDetach.addListener(onDetach);
    await chrome.debugger.sendCommand(target, 'Network.enable', {maxTotalBufferSize: 64 * 1024 * 1024, maxResourceBufferSize: 32 * 1024 * 1024});
  } catch {await stop(); error = '未获得页面资源捕获权限，请重新加载扩展并在扩展中启用捕获';}
  return {get error() {return error;}, resources, finish: async () => {let timer; try {await Promise.race([Promise.allSettled([...pending]), new Promise(resolve => {timer = setTimeout(resolve, 10000);})]);} finally {clearTimeout(timer);}}, stop};
}

// Runs in the requested page only. No cookies, hidden state, form fields, or unrelated tabs.
function extractVisiblePage() {
  const visible = el => !!el && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
  const textOf = el => visible(el) ? (el.innerText || '').trim() : '';
  const main = document.querySelector('article, [itemprop="articleBody"], .post_content, #paragraph, .article-content, main') || document.body;
  const text = textOf(main).slice(0, 28000);
  const blocking = [...document.querySelectorAll('[role="dialog"], [class*="login"], [class*="captcha"], [id*="captcha"]')].filter(visible).some(el => /扫码登录|请先登录|登录后查看|安全验证|完成验证|验证码/.test(textOf(el)));
  if (blocking || /just a moment|access denied|attention required/i.test(document.title) || text.length < 180 && /登录|验证|sign in|captcha/i.test(document.body.innerText)) return {status: 'waiting_login', error: '请在这个网页完成登录或验证，然后在 API 中重新提交同一网址'};
  if (text.length < 80) return {status: 'failed', error: '没有读取到正文，请等待页面加载或补充正文后重试'};
  const comments = [], seen = new Set();
  const selectors = ['.CommentItem', '.CommentItemV2', '[class*="CommentItem"]', '.comment-item', '.comment_item', '.reply-item', '.root-reply-container', '.commentlist li', '[data-testid="comment"]'];
  for (const el of document.querySelectorAll(selectors.join(','))) {
    if (!visible(el)) continue;
    const body = el.querySelector('.comment-content, .reply-content, .CommentItem-content, [class*="CommentContent"], .RichText, .text, .con') || el;
    const content = textOf(body).slice(0, 4000);
    if (!content || seen.has(content)) continue; seen.add(content);
    const author = textOf(el.querySelector('[class*="user-name"], [class*="UserLink"], .name, .author')).slice(0, 200);
    const likeText = textOf(el.querySelector('[class*="like"], [class*="Like"], [aria-label*="赞"]'));
    const match = likeText.replace(/,/g, '').match(/(\d+(?:\.\d+)?)\s*(万|千|[kK])?/);
    const likes = match ? Math.round(Number(match[1]) * (match[2] === '万' ? 10000 : match[2] ? 1000 : 1)) : undefined;
    comments.push({text: content, ...(author ? {author} : {}), ...(likes !== undefined ? {likes} : {})});
    if (comments.length >= 60) break;
  }
  comments.sort((a, b) => (b.likes ?? -1) - (a.likes ?? -1));
  const media = [...main.querySelectorAll('img, video')].filter(visible).flatMap(el => {
    const url = el.tagName === 'VIDEO' && /^blob:/.test(el.currentSrc || el.src || '') ? location.href : el.currentSrc || el.src;
    if (!/^https?:\/\//i.test(url || '') || el.tagName === 'IMG' && el.naturalWidth < 250) return [];
    const caption = textOf(el.closest('figure')?.querySelector('figcaption'));
    const context = [el.getAttribute('alt'), el.getAttribute('title'), caption, textOf(el.parentElement).slice(0, 700)].filter(Boolean).join(' · ').slice(0, 2000);
    return [{url, kind: el.tagName === 'VIDEO' ? 'video' : 'image', context}];
  }).slice(0, 12);
  return {status: 'completed', title: document.title.slice(0, 300), url: location.href, text, comments, media};
}
async function tick() {
  if (busy) return;
  const {enabled} = await chrome.storage.local.get('enabled');
  if (!enabled) return;
  busy = true;
  let task, capture, ownedTab, keepTab = false, reachedServer = false;
  try {
    if (!await chrome.permissions.contains(readPermissions)) throw Error('网页读取权限未授予，请点击“开启”并允许授权。');
    const payload = await serviceRequest();
    reachedServer = true;
    task = payload.task;
    await chrome.storage.local.set({connection: 'connected', lastConnectedAt: Date.now(), status: task ? '已连接，正在采集：' + task.title : '已连接，等待采集任务'});
    if (!task) return;
    if (!publicPage(task.url)) throw Error('不允许采集本机或内网页面');
    const url = new URL(task.url);
    // A fresh task tab shares the user's login but never reloads their open tabs.
    let tab = await chrome.tabs.create({url: 'about:blank', active: false}); ownedTab = tab.id;
    capture = await capturePageResponses(tab.id);
    tab = await chrome.tabs.update(tab.id, {url: task.url});
    for (let i = 0; i < 20; i++) {tab = await chrome.tabs.get(tab.id); if (tab.status === 'complete') break; await delay(700);}
    await delay(2000);
    tab = await chrome.tabs.get(tab.id);
    if (!publicPage(tab.url) || new URL(tab.url).hostname !== url.hostname) throw Error('页面跳转到了其他站点，请用实际链接重新提交');
    const [{result}] = await chrome.scripting.executeScript({target: {tabId: tab.id}, func: extractVisiblePage});
    if (result.status === 'completed') {
      await capture.finish();
      for (const media of result.media) {
        const resource = capture.resources.get(media.url);
        if (!resource || resource.kind !== media.kind) {media.captureError = capture.error || '未捕获到完整响应（可能尚未加载、为缓存或视频分片），保留线索'; continue;}
        try {
          await serviceRequest('?mode=media', {method: 'POST', headers: {'Content-Type': resource.mime, 'X-Collector-Media': encodeURIComponent(JSON.stringify({id: task.id, lease: task.lease, url: media.url, pageUrl: result.url, kind: media.kind, context: media.context, responseStatus: resource.responseStatus, contentRange: resource.contentRange}))}, body: resource.bytes});
        } catch (error) {media.captureError = String(error.message).slice(0, 1000);}
      }
    }
    const saved = await fetch(API, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Collector-Version': chrome.runtime.getManifest().version}, body: JSON.stringify({...result, id: task.id, lease: task.lease}), signal: AbortSignal.timeout(10000)});
    if (!saved.ok) throw Error((await saved.json()).error || '保存失败');
    if (result.status === 'waiting_login') {keepTab = true; await chrome.tabs.update(tab.id, {active: true}); await chrome.storage.local.set({status: result.error});}
  } catch (e) {
    await chrome.storage.local.set({connection: reachedServer ? 'connected' : 'error', status: (reachedServer ? '采集服务已连接，但当前网页读取失败：' : '') + (e.message || '连接中断，请检查采集服务')});
    if (task) {
      try {await fetch(API, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Collector-Version': chrome.runtime.getManifest().version}, body: JSON.stringify({id: task.id, lease: task.lease, status: 'failed', error: String(e.message).slice(0, 1000)}), signal: AbortSignal.timeout(10000)});} catch {}
    }
  } finally {await capture?.stop(); if (ownedTab && !keepTab) {try {await chrome.tabs.remove(ownedTab);} catch {}} busy = false;}
}
chrome.alarms.onAlarm.addListener(alarm => {if (alarm.name === 'collector-poll') {if (busy) void connectBrowser(true); else void tick();}});
// 只有用户在弹窗里点击过“开启”才会轮询；安装或浏览器启动后会继续保持开启状态。
async function startPolling() {
  const {enabled} = await chrome.storage.local.get('enabled');
  if (!enabled) return;
  await chrome.alarms.create('collector-poll', {periodInMinutes: 0.5});
  void tick();
}
chrome.runtime.onStartup.addListener(() => void startPolling());
chrome.runtime.onInstalled.addListener(() => void startPolling());
chrome.runtime.onMessage.addListener((message, _sender, respond) => {
  if (!['connect', 'check'].includes(message.action)) return;
  // A handshake never waits for page loading or claims a collection task.
  connectBrowser().then(async result => {
    respond(result);
    if (result.ok && message.action === 'connect') void tick();
  }).catch(e => respond({ok: false, error: e.message}));
  return true;
});
