import { Brain } from './brain.js';
import { Circuit } from './circuit.js';
import { Raster } from './hud.js';
import { Maze } from './maze.js';
import { Stage } from './stage.js';

const $ = (sel) => document.querySelector(sel);
const pct = (p) => `${Math.round(p * 100)}%`;
// Small probabilities (an LLM's calibrated threshold can be 1e-8) stay readable instead of showing "0%".
const prob = (p) => (p >= 0.01 || p === 0 ? pct(p) : p >= 1e-4 ? `${(p * 100).toFixed(2)}%` : `${p.toExponential(1)}`);
const finalVote = (res) => res.final || res.vote;   // the cascade's verdict when the server sends one
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
  el.textContent = vote.by === 'llm'
    ? `${inj ? '격리 · 인젝션' : 'LLM으로 전달 · 정상'} · 큰 모델 판정`
    : `${inj ? '격리 · 인젝션' : 'LLM으로 전달 · 정상'} ${pct(vote.p)}`;
}

function routeLine(r) {
  if (!r) return '';
  const dist = `가장 가까운 학습 문장까지 거리 ${r.distance.toFixed(2)}, 기준 ${r.tau.toFixed(2)}`;
  if (r.kind === 'fly') return `<p class="route fly"><b>익숙한 냄새</b> (${dist}) → 초파리가 판정했어요.</p>`;
  if (r.kind === 'unfamiliar') {
    return `<p class="route unfamiliar"><b>낯선 냄새</b> (${dist}) → 큰 모델이 꺼져 있어 초파리가 판정했어요.</p>`;
  }
  const q = r.llm;
  return `<p class="route llm"><b>낯선 냄새</b> (${dist}) → ${esc(q.name)}에게 물어봤어요: <b>${LABEL[q.decision]}</b>
    (Yes 확률 ${prob(q.p)}, S-Labs validation으로 정한 기준 ${prob(q.threshold)}). 초파리 대신 이 판정을 따랐어요.</p>`;
}

function onInspect(item, res) {
  Object.assign(state, { last: res, item, selected: res.rep });
  brain.show(res.fly);
  circuit.show(res.fly);
  raster.burst(res.fly.kc_active);
  $('#hud-active').textContent = res.fly.kc_active.length;
  $('#hud-text').textContent = item.text;
  const asking = res.route?.kind === 'llm' ? ' · 낯선 냄새, 큰 모델에 물어보는 중' : '';
  $('#hud-tag').textContent = `${item.tag || ''}${asking}`;
  setVerdict(null);
}

function onSorted(item, res) {
  const s = stream.stats;
  const final = finalVote(res);
  const inj = final.decision === 'injection';
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
  const who = final.by === 'llm' ? ' (큰 모델)' : ` ${pct(final.p)}`;
  row.textContent = `${inj ? '격리' : '전달'}${who}${ok == null ? '' : ok ? ' ✓' : ' ✗'} · ${item.text}`;
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
  const { flies, vote, comparator, route } = state.last;
  const item = state.item;
  const fly = flies[state.selected];
  const final = finalVote(state.last);
  const inj = final.decision === 'injection';
  const many = flies.length > 1;
  const agree = comparator.decision === final.decision;
  const flyInj = vote.decision === 'injection';
  const flySays = final.by === 'llm'
    ? `초파리 판단은 ${flyInj ? '격리' : '전달'} ${pct(vote.p)}였어요 · `
    : '';
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
      <small>${flySays}${many ? `폴드 ${flies.length}개 투표 · 격리 ${vote.injection} / 전달 ${vote.benign} · 평균 인젝션 확률 ${pct(vote.p)}`
                    : `초파리 인젝션 확률 ${pct(vote.p)}`}</small></div>
    ${routeLine(route)}
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

// --- shortcut lab ------------------------------------------------------------------------------
function renderLabExamples() {
  const box = $('#lab-examples');
  box.replaceChildren();
  const benign = state.samples.filter((s) => s.label === 0 && s.tag !== 'hard-negative').slice(0, 4);
  const attack = state.samples.filter((s) => s.label === 1).slice(0, 1);
  for (const s of [...benign, ...attack]) {
    const b = document.createElement('button');
    b.className = 'ghost';
    b.title = s.text;
    b.textContent = `${s.label ? '공격' : '정상'} · ${s.text.length > 48 ? `${s.text.slice(0, 46)}…` : s.text}`;
    b.addEventListener('click', () => {
      $('#lab-text').value = s.text;
      runLab();
    });
    box.append(b);
  }
}

async function runLab() {
  const text = $('#lab-text').value.trim();
  const out = $('#lab-result');
  if (!text) return;
  if (state.offline) {
    out.innerHTML = '<p class="warn">지름길 실험실은 서버가 켜져 있을 때만 동작해요.</p>';
    return;
  }
  out.classList.add('pending');
  try {
    const res = await api('/api/lab', { text });
    const base = res.rows[0].scores;
    const cell = (s, orig) => {
      const inj = s.decision === 'injection';
      const flip = s.decision !== orig.decision;
      return `<div class="lab-cell ${inj ? 'block' : 'pass'}"><b>${inj ? '막음' : '통과'} ${pct(s.p)}</b>
        <div class="lab-bar"><i style="width:${Math.max(2, Math.round(s.p * 100))}%"></i></div>
        ${flip ? '<span class="flip">원문과 판정이 달라졌어요</span>' : ''}</div>`;
    };
    const rows = res.rows.map((r) => `<tr><td class="variant">${esc(r.variant)}<small>${esc(r.text)}</small></td>${
      res.models.map((m) => `<td>${cell(r.scores[m.id], base[m.id])}</td>`).join('')}</tr>`).join('');
    out.innerHTML = `<table class="lab-table"><thead><tr><th>글자만 바꾼 변형</th>${
      res.models.map((m) => `<th>${esc(m.label)}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table>
      <p class="muted lab-note">"막음"은 인젝션으로 판정했다는 뜻이에요. 내용은 그대로이니, 정상 문장이 막히면 글자 모양에 속은 거예요.
      키릴 글자로 바꾼 변형은 S-Labs train의 공격 기법 그대로라 막는 게 배운 대로예요.</p>`;
  } catch (err) {
    out.innerHTML = `<p class="warn">${esc(err.message)}</p>`;
  } finally {
    out.classList.remove('pending');
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
      setVerdict(finalVote(res));
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
  renderLabExamples();
  $('#lab-go').addEventListener('click', runLab);
  $('#lab-text').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runLab();
  });
  $('#theme').addEventListener('click', () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
  document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => setTab(b.dataset.tab)));
  setInterval(feed, 400);
  setTab('live');
}

boot();
