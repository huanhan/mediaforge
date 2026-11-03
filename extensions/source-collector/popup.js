const status = document.querySelector('#status');
const check = document.querySelector('#check'), toggle = document.querySelector('#toggle');
const readPermissions = {origins: ['https://*/*', 'http://*/*']};
const DEFAULT_STATUS = '点击“测试连接”检查本机 MiniMax H3 服务（默认端口 8000）。';
let enabled = false;

function renderToggle(value) {
  enabled = value === true;
  toggle.textContent = enabled ? '关闭' : '开启';
}
function showStatus(text, kind = '') {
  status.textContent = text || DEFAULT_STATUS;
  status.classList.toggle('ok', kind === 'ok');
  status.classList.toggle('err', kind === 'err');
}
async function refresh() {
  const value = await chrome.storage.local.get(['enabled', 'status', 'connection']);
  renderToggle(value.enabled);
  showStatus(value.status, value.connection === 'connected' ? 'ok' : value.connection === 'error' ? 'err' : '');
}
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== 'local') return;
  if (changes.enabled) renderToggle(changes.enabled.newValue);
  if (changes.status || changes.connection) void refresh();
});
async function send(action) {
  check.disabled = true; toggle.disabled = true;
  showStatus('正在检查…');
  try {
    const result = await chrome.runtime.sendMessage({action});
    if (!result?.ok) throw Error(result?.error || '扩展没有回应，请在扩展管理页重新加载后再试。');
    await refresh();
    showStatus(result.message, 'ok');
  } catch (e) {
    showStatus('连接失败：' + e.message, 'err');
  } finally {
    check.disabled = false; toggle.disabled = false;
  }
}
check.onclick = () => send('check');
toggle.onclick = async () => {
  if (enabled) {
    await chrome.alarms.clear('collector-poll');
    await chrome.storage.local.set({enabled: false, status: '已关闭采集，不再领取新任务。'});
    renderToggle(false);
    return;
  }
  // The permission prompt must run directly from the click, before any other await.
  const allowed = await chrome.permissions.request(readPermissions);
  if (!allowed) {
    showStatus('未获得网页读取权限，请再次点击“开启”并在浏览器提示中选择允许。', 'err');
    return;
  }
  await chrome.storage.local.set({enabled: true});
  await chrome.alarms.create('collector-poll', {periodInMinutes: 0.5});
  await send('connect');
};
refresh();
