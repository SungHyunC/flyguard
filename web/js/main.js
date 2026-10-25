import { Brain } from './brain.js';
import { Circuit } from './circuit.js';
import { Raster } from './hud.js';
import { Maze } from './maze.js';
import { Stage } from './stage.js';

const $ = (sel) => document.querySelector(sel);
const pct = (p) => `${Math.round(p * 100)}%`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const LABEL = { benign: '정상', injection: '인젝션' };
const TAGS = {
  'hard-negative': '헷갈리는 정상', benign: '정상', override: '지시 무시', roleplay: '역할 탈옥',
  extraction: '프롬프트 탈취', obfuscation: '난독화', urgency: '긴급 압박', hypothetical: '가정 상황',
};

const state = {
  brain: null, samples: [], batch: null, offline: false, batchRun: null,
  noses: [], nose: 'frozen', last: null, item: null, selected: 0,
};
// The live stream: prompts are classified one by one and put on the conveyor, at most three at a time.
const stream = { running: false, busy: false, pool: [], custom: [], stats: { n: 0, pass: 0, block: 0, right: 0, labelled: 0 } };
let stage, maze, brain, circuit, raster;

async function api(path, body) {
  const opts = body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {};
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${path} 응답 ${res.status}`);
  return res.json();
}

function setStatus(text, kind = '') {
  const el = $('#status');
  el.textContent = text;
  el.className = kind;
}

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// --- theme -------------------------------------------------------------------------------------
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('flyguard-theme', theme); } catch { /* storage may be blocked */ }
  $('#theme').textContent = theme === 'dark' ? '밝은 화면' : '어두운 화면';
  maze?.applyTheme();
  brain?.applyTheme();
  circuit?.draw(1);
}

function initialTheme() {
  try {
    const saved = localStorage.getItem('flyguard-theme');
    if (saved) return saved;
  } catch { /* ignore */ }
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

// --- noses -------------------------------------------------------------------------------------
const noseInfo = () => state.noses.find((n) => n.id === state.nose) || { label: '고정 MiniLM 코', flies: 1 };

function renderNoses() {
  const box = $('#noses');
  box.replaceChildren();
  for (const n of state.noses) {
    const b = document.createElement('button');
    b.className = n.id === state.nose ? 'on' : '';
    b.title = `validation 정확도 ${pct(n.val_acc)}`;
    b.textContent = n.flies > 1 ? `${n.label} (폴드 ${n.flies}개 투표)` : n.label;
    b.addEventListener('click', () => {
      state.nose = n.id;
      renderNoses();
      describeNose();
    });
    box.append(b);
  }
  describeNose();
}

function describeNose() {
  const n = noseInfo();
  $('#hud-nose').textContent = `초파리 1마리 · ${n.label}${n.flies > 1 ? ` · 폴드 ${n.flies}개 투표` : ''}`;
}

// --- live stream -------------------------------------------------------------------------------
function buildPool() {
  const samples = shuffle(state.samples.map((s) => ({ text: s.text, label: s.label, tag: TAGS[s.tag] || s.tag, sample: s })));
  if (state.offline) return samples;
  const split = state.batch?.split || 'test';
  const rest = shuffle((state.batch?.items || []).map((it) => ({ text: it.text, label: it.label, tag: split })));
  return $('#stream-source').value === 'samples' ? [...samples, ...rest] : rest;
}

function offlineResult(item) {
  const fly = item.sample?.fly;
  if (!fly) return null;
  const inj = fly.decision === 'injection';
  return { text: item.text, flies: [fly], rep: 0, fly, comparator: item.sample.comparator,
           vote: { p: fly.p, decision: fly.decision, injection: +inj, benign: +!inj } };
}

async function feed() {
  if (stream.busy || !stream.running) return;
  stream.busy = true;
  try {
    while (stream.running && stage.load < 3) {
      if (!stream.custom.length && !stream.pool.length) stream.pool = buildPool();
      const item = stream.custom.shift() || stream.pool.shift();
      if (!item) break;
      const res = state.offline ? offlineResult(item) : await api('/api/classify', { text: item.text, nose: state.nose });
      if (res) stage.add(item, res);
    }
    setStatus('');
  } catch (err) {
    setStatus(err.message, 'error');
  } finally {
    stream.busy = false;
  }
}

function setRunning(on) {
  stream.running = on;
  $('#stream-go').textContent = on ? '❚❚ 일시정지' : '▶ 스트림 시작';
  if (on) feed();
}

function inject(item) {
  stream.custom.push(item);
  if (!stream.running) setRunning(true);
  else feed();
}

function setVerdict(vote) {
  const el = $('#hud-verdict');
  if (!vote) {
    el.className = 'chip wait';
    el.textContent = '냄새 맡는 중…';
    return;
  }
  const inj = vote.decision === 'injection';
  el.className = `chip ${inj ? 'block' : 'pass'}`;
  el.textContent = `${inj ? '격리 · 인젝션' : 'LLM으로 전달 · 정상'} ${pct(vote.p)}`;
}

function onInspect(item, res) {
  Object.assign(state, { last: res, item, selected: res.rep });
  brain.show(res.fly);
  circuit.show(res.fly);
  raster.burst(res.fly.kc_active);
  $('#hud-active').textContent = res.fly.kc_active.length;
  $('#hud-text').textContent = item.text;
  $('#hud-tag').textContent = item.tag || '';
  setVerdict(null);
}

function onSorted(item, res) {
  const s = stream.stats;
  const inj = res.vote.decision === 'injection';
  s.n++;
  s[inj ? 'block' : 'pass']++;
  if (item.label != null) {
    s.labelled++;
    s.right += (item.label === 1) === inj;
  }
  $('#hud-stats').textContent = `처리 ${s.n} · 전달 ${s.pass} · 격리 ${s.block}${s.labelled ? ` · 정답률 ${pct(s.right / s.labelled)}` : ''}`;
  const row = document.createElement('li');
  const ok = item.label == null ? null : (item.label === 1) === inj;
  row.className = ok == null ? '' : ok ? 'ok' : 'bad';
  row.textContent = `${inj ? '격리' : '전달'} ${pct(res.vote.p)}${ok == null ? '' : ok ? ' ✓' : ' ✗'} · ${item.text}`;
  const ticker = $('#live-ticker');
  ticker.prepend(row);
  if (ticker.childElementCount > 7) ticker.lastElementChild.remove();
}

function selectFly(i) {
  state.selected = i;
  brain.show(state.last.flies[i]);
  circuit.show(state.last.flies[i]);
  showResult();
}

function showResult() {
  const { flies, vote, comparator } = state.last;
  const item = state.item;
  const fly = flies[state.selected];
  const inj = vote.decision === 'injection';
  const many = flies.length > 1;
  const agree = comparator.decision === vote.decision;
  let truth = '';
  if (item?.label != null) {
    const correct = (item.label === 1) === inj;
    truth = `<p class="truth ${correct ? 'ok' : 'bad'}">정답 ${LABEL[item.label ? 'injection' : 'benign']} · 초파리 ${correct ? '맞힘' : '틀림'}</p>`;
  }
  const votes = many ? `<div class="votes" role="group" aria-label="폴드별 판정">${flies.map((f, i) => `
      <button data-fly="${i}" class="${f.decision === 'injection' ? 'block' : 'pass'}${i === state.selected ? ' on' : ''}">
        <b>폴드 ${i + 1}</b>${f.decision === 'injection' ? '격리' : '전달'} ${pct(f.p)}</button>`).join('')}</div>` : '';
  const block = fly.kc_vote.filter((v) => v > 0).length;
  $('#result').innerHTML = `
    <p class="muted small">방금 판단한 프롬프트</p>
    <p class="now-text">${esc(item.text)}</p>
    <div class="verdict ${inj ? 'block' : 'pass'}">${inj ? '격리' : 'LLM으로 전달'}
      <small>${many ? `폴드 ${flies.length}개 투표 · 격리 ${vote.injection} / 전달 ${vote.benign} · 평균 인젝션 확률 ${pct(vote.p)}`
                    : `인젝션 확률 ${pct(vote.p)}`}</small></div>
    ${votes}
    <dl class="facts">
      ${many ? `<div class="wide"><dt>뇌 패널에 보이는 회로</dt><dd>폴드 ${state.selected + 1} (번호를 누르면 바뀌어요)</dd></div>` : ''}
      <div><dt>판정 여유</dt><dd>${fly.margin >= 0 ? '+' : ''}${fly.margin.toFixed(2)}σ</dd></div>
      <div><dt>MBON 접근 / 회피</dt><dd>${fly.approach.toFixed(3)} / ${fly.avoid.toFixed(3)}</dd></div>
      <div><dt>격리에 투표한 KC</dt><dd>${block} / ${fly.kc_active.length}</dd></div>
      <div><dt>${esc(comparator.name)}</dt><dd class="${agree ? 'ok' : 'bad'}">${LABEL[comparator.decision]} ${pct(comparator.p)}</dd></div>
    </dl>
    ${truth}`;
  $('#result').querySelectorAll('.votes button').forEach((b) => b.addEventListener('click', () => selectFly(Number(b.dataset.fly))));
}

function renderChips() {
  const filter = $('#sample-filter').value;
  const box = $('#chips');
  box.replaceChildren();
  for (const s of state.samples) {
    const group = s.tag === 'hard-negative' ? 'hard' : s.label ? 'attack' : 'benign';
    if (filter !== 'all' && filter !== group) continue;
    const b = document.createElement('button');
    b.className = `chip ${group}`;
    b.title = s.text;
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = TAGS[s.tag] || s.tag;
    b.append(tag, s.text.length > 70 ? `${s.text.slice(0, 68)}…` : s.text);
    b.addEventListener('click', () => inject({ text: s.text, label: s.label, tag: TAGS[s.tag] || s.tag, sample: s }));
    box.append(b);
  }
}

// --- batch mode (T-maze) -----------------------------------------------------------------------
async function startBatch() {
  if (state.batchRun) return stopBatch();
  try {
    state.batch ??= await api('/api/batch');
  } catch (err) {
    setStatus(err.message, 'error');
    return;
  }
  const items = state.batch.items;
  const key = state.nose === 'hybrid' && state.batch.hybrid ? 'hyb' : 'fly';
  maze.clearPile();
  const s = { i: 0, fly: 0, comp: 0, tp: 0, fp: 0, tn: 0, fn: 0, last: performance.now() };
  state.batchRun = s;
  $('#batch-go').textContent = '멈추기';
  $('#batch-split').textContent = state.batch.split === 'test' ? 'test' : 'validation';
  $('#batch-fly-name').textContent = key === 'hyb' ? `초파리 (${noseInfo().label}, 폴드 ${noseInfo().flies}개 투표)` : '초파리 (고정 MiniLM 코)';
  $('#batch-comp-name').textContent = state.batch.comparator;
  const ticker = $('#ticker');
  ticker.replaceChildren();
  const step = (now) => {
    if (state.batchRun !== s) return;
    const rate = Number($('#batch-speed').value);
    let todo = Math.max(1, Math.floor(((now - s.last) / 1000) * rate));
    s.last = now;
    while (todo-- > 0 && s.i < items.length) {
      const it = items[s.i++];
      const pred = it[key];
      maze.drop(pred ? 'quarantine' : 'pass', it.label);
      s.fly += pred === it.label;
      s.comp += it.comp === it.label;
      s[it.label ? (pred ? 'tp' : 'fn') : (pred ? 'fp' : 'tn')]++;
      const row = document.createElement('li');
      row.className = pred === it.label ? 'ok' : 'bad';
      row.textContent = `${pred === it.label ? '✓' : '✗'} ${it.text}`;
      ticker.prepend(row);
      if (ticker.childElementCount > 8) ticker.lastElementChild.remove();
    }
    $('#batch-n').textContent = `${s.i} / ${items.length}`;
    $('#batch-fly').textContent = pct(s.fly / s.i);
    $('#batch-comp').textContent = pct(s.comp / s.i);
    for (const k of ['tp', 'fp', 'tn', 'fn']) $(`#cm-${k}`).textContent = s[k];
    if (s.i < items.length) requestAnimationFrame(step);
    else stopBatch();
  };
  requestAnimationFrame(step);
}

function stopBatch() {
  state.batchRun = null;
  $('#batch-go').textContent = '다시 시작';
}

// --- physics videos ----------------------------------------------------------------------------
async function loadVideos() {
  const box = $('#videos');
  let list = [];
  try { list = state.offline ? [] : await api('/api/videos'); } catch { /* no videos */ }
  box.replaceChildren();
  if (!list.length) {
    box.innerHTML = '<p class="muted">아직 영상이 없어요. <code>sim/README.md</code>를 따라 NeuroMechFly 영상을 렌더해 <code>web/videos/</code>에 넣으면 여기에 나타나요.</p>';
    return;
  }
  const REACHED = { left: '왼쪽 냄새원으로 갔어요', right: '오른쪽 냄새원으로 갔어요', neither: '어느 쪽에도 가지 않았어요' };
  const side = (name, p) => `
    <div class="vside ${p.decision === 'injection' ? 'block' : 'pass'}">
      <span class="muted small">${name} 냄새원 · ${p.decision === 'injection' ? '인젝션 (빨강, 피하는 냄새)' : '정상 (초록, 끌리는 냄새)'} ${pct(p.p)}</span>
      <span>${esc(p.text)}</span></div>`;
  for (const m of list) {
    const card = document.createElement('figure');
    card.className = 'vcard';
    const v = document.createElement('video');
    Object.assign(v, { src: m.file, controls: true, muted: true, loop: true, playsInline: true, autoplay: true });
    card.append(v);
    if (m.left) {
      const cap = document.createElement('figcaption');
      cap.innerHTML = `${side('왼쪽', m.left)}${side('오른쪽', m.right)}
        <p class="vreach">→ ${REACHED[m.reached] || m.reached} <span class="muted small">(시뮬레이션 ${m.seconds}초, ${m.play_speed}배속)</span></p>`;
      card.append(cap);
    }
    box.append(card);
  }
}

function setTab(tab) {
  document.body.dataset.tab = tab;
  document.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === tab));
  stage.setActive(tab === 'live');
  brain.setActive(tab === 'live');
  maze.setActive(tab === 'batch');
  if (tab !== 'live') setRunning(false);
  if (tab !== 'batch') stopBatch();
  if (tab === 'physics') loadVideos();
}

async function boot() {
  applyTheme(initialTheme());
  try {
    [state.brain, state.samples, state.noses, state.batch] = await Promise.all(
      [api('/api/brain'), api('/api/samples'), api('/api/noses'), api('/api/batch')]);
    state.nose = state.noses[0].id;
  } catch {
    const fb = window.FLYGUARD_FALLBACK;
    if (!fb) {
      setStatus('서버에 연결할 수 없어요. README의 실행 방법을 확인해 주세요.', 'error');
      return;
    }
    Object.assign(state, { offline: true, brain: fb.brain, samples: fb.samples, batch: fb.batch, noses: [], nose: 'frozen' });
    setStatus('오프라인 모드: 예시 문장만 재생합니다.', 'warn');
  }
  stage = new Stage($('#stage'));
  brain = new Brain($('#brain'), state.brain, { labels: false, frame: 'brain' });
  circuit = new Circuit($('#circuit'), state.brain);
  maze = new Maze($('#maze'));
  raster = new Raster($('#raster'));
  stage.on = {
    inspect: onInspect, sorted: onSorted,
    decide: (item, res) => {   // reveal the verdict only when the fly acts on it
      setVerdict(res.vote);
      showResult();
    },
    frame: (dt) => {
      raster.tick(dt);
      $('#hud-clock').textContent = `${stage.time.toFixed(2)} S`;
    },
  };
  window.flyguard = { stage, maze, brain, circuit, state, stream };   // handle for debugging in the console
  const layout = state.brain;
  $('#hud-kc').textContent = layout.kc.length.toLocaleString();
  $('#hud-wiring').textContent = `Kenyon cells · 사구체 ${layout.glom.length}개`;
  $('#hud-source').textContent = layout.source.startsWith('flywire') ? '실제 FlyWire 배선 · 오른쪽 반구' : '무작위 배선 · 모식도 좌표';
  $('#hud-total').textContent = layout.kc.length.toLocaleString();
  renderNoses();
  renderChips();

  $('#stream-go').addEventListener('click', () => setRunning(!stream.running));
  $('#stream-speed').addEventListener('change', (e) => { stage.speed = Number(e.target.value); });
  $('#stream-source').addEventListener('change', () => { stream.pool = buildPool(); });
  $('#go').addEventListener('click', () => {
    const text = $('#prompt').value.trim();
    if (text) inject({ text, label: null, tag: '직접 입력' });
  });
  $('#prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') $('#go').click();
  });
  $('#sample-filter').addEventListener('change', renderChips);
  $('#batch-go').addEventListener('click', startBatch);
  $('#theme').addEventListener('click', () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
  document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => setTab(b.dataset.tab)));
  setInterval(feed, 400);
  setTab('live');
}

boot();
