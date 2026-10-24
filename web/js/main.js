import { Brain } from './brain.js';
import { Circuit } from './circuit.js';
import { Maze } from './maze.js';

const $ = (sel) => document.querySelector(sel);
const pct = (p) => `${Math.round(p * 100)}%`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const LABEL = { benign: '정상', injection: '인젝션' };
const TAGS = {
  'hard-negative': '헷갈리는 정상', benign: '정상', override: '지시 무시', roleplay: '역할 탈옥',
  extraction: '프롬프트 탈취', obfuscation: '난독화', urgency: '긴급 압박', hypothetical: '가정 상황',
};

const state = {
  brain: null, samples: [], batch: null, offline: false, busy: false, batchRun: null,
  noses: [], nose: 'frozen', last: null, sample: null, selected: 0,
};
let maze, brain, circuit;

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

// One dot per fly on the evidence track (left = pass, right = quarantine).
function setDots(n) {
  const track = $('#evidence-track');
  track.replaceChildren();
  for (let i = 0; i < n; i++) {
    const d = document.createElement('div');
    d.className = 'edot';
    d.textContent = n > 1 ? i + 1 : '';
    d.style.top = `calc(50% + ${(i - (n - 1) / 2) * 7}px)`;   // stack the flies' dots around the track
    track.append(d);
  }
}

function setEvidence(i, e) {
  const d = $('#evidence-track').children[i];
  if (d) d.style.left = `${50 + 50 * e}%`;
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
function renderNoses() {
  const box = $('#noses');
  box.replaceChildren();
  for (const n of state.noses) {
    const b = document.createElement('button');
    b.className = n.id === state.nose ? 'on' : '';
    b.title = `validation 정확도 ${pct(n.val_acc)}`;
    b.textContent = `${n.label} · 초파리 ${n.flies}마리`;
    b.addEventListener('click', () => {
      if (state.busy) return;
      state.nose = n.id;
      renderNoses();
      setDots(n.flies);
      maze.setCount(n.flies);
      maze.reset();
    });
    box.append(b);
  }
}

const noseInfo = () => state.noses.find((n) => n.id === state.nose) || { label: '고정 MiniLM 코', flies: 1 };

// --- live mode ---------------------------------------------------------------------------------
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
    b.addEventListener('click', () => {
      $('#prompt').value = s.text;
      run(s.text, s);
    });
    box.append(b);
  }
}

async function run(text, sample) {
  text = text.trim();
  if (!text || state.busy) return;
  state.busy = true;
  $('#go').disabled = true;
  setStatus('냄새를 맡는 중…');
  $('#result').classList.add('pending');
  try {
    let res;
    if (state.offline) {
      const s = sample || state.samples.find((x) => x.text === text);
      if (!s) throw new Error('오프라인 모드에서는 예시 문장만 쓸 수 있어요.');
      res = {
        text: s.text, flies: [s.fly], rep: 0, fly: s.fly, comparator: s.comparator,
        vote: { p: s.fly.p, decision: s.fly.decision, injection: +(s.fly.decision === 'injection'), benign: +(s.fly.decision === 'benign') },
      };
    } else {
      res = await api('/api/classify', { text, nose: state.nose });
    }
    sample = sample || state.samples.find((x) => x.text === res.text);
    Object.assign(state, { last: res, sample, selected: res.rep });
    setDots(res.flies.length);
    brain.show(res.fly);
    circuit.show(res.fly);
    await maze.run(res.text, res.flies, res.vote.decision);
    showResult();
    setStatus('');
  } catch (err) {
    setStatus(err.message, 'error');
  } finally {
    state.busy = false;
    $('#go').disabled = false;
    $('#result').classList.remove('pending');
  }
}

function selectFly(i) {
  state.selected = i;
  brain.show(state.last.flies[i]);
  circuit.show(state.last.flies[i]);
  showResult();
}

function showResult() {
  const { flies, vote, comparator } = state.last;
  const sample = state.sample;
  const fly = flies[state.selected];
  const inj = vote.decision === 'injection';
  const many = flies.length > 1;
  const agree = comparator.decision === vote.decision;
  let truth = '';
  if (sample) {
    const correct = (sample.label === 1) === inj;
    truth = `<p class="truth ${correct ? 'ok' : 'bad'}">정답 ${LABEL[sample.label ? 'injection' : 'benign']} · 초파리 ${correct ? '맞힘' : '틀림'}</p>`;
  }
  const votes = many ? `<div class="votes" role="group" aria-label="초파리별 판정">${flies.map((f, i) => `
      <button data-fly="${i}" class="${f.decision === 'injection' ? 'block' : 'pass'}${i === state.selected ? ' on' : ''}">
        <b>${i + 1}</b>${f.decision === 'injection' ? '격리' : '전달'} ${pct(f.p)}</button>`).join('')}</div>` : '';
  const block = fly.kc_vote.filter((v) => v > 0).length;
  $('#result').innerHTML = `
    <div class="verdict ${inj ? 'block' : 'pass'}">${inj ? '격리' : 'LLM으로 전달'}
      <small>${many ? `초파리 ${flies.length}마리 투표 · 격리 ${vote.injection} / 전달 ${vote.benign} · 평균 인젝션 확률 ${pct(vote.p)}`
                    : `${inj ? '인젝션' : '정상'}으로 판단 · 인젝션 확률 ${pct(vote.p)}`}</small></div>
    ${votes}
    <dl class="facts">
      ${many ? `<div class="wide"><dt>뇌 패널에 보이는 초파리</dt><dd>${state.selected + 1}번 (번호를 누르면 바뀌어요)</dd></div>` : ''}
      <div><dt>인젝션 확률</dt><dd>${pct(fly.p)}</dd></div>
      <div><dt>판정 여유</dt><dd>${fly.margin >= 0 ? '+' : ''}${fly.margin.toFixed(2)}σ</dd></div>
      <div><dt>MBON 접근 / 회피</dt><dd>${fly.approach.toFixed(3)} / ${fly.avoid.toFixed(3)}</dd></div>
      <div><dt>격리에 투표한 KC</dt><dd>${block} / ${fly.kc_active.length}</dd></div>
    </dl>
    ${!many && fly.novelty > 0.3 ? '<p class="warn">낯선 냄새: 학습 때 거의 켜지지 않던 케년세포가 많이 켜졌어요. 영어가 아니거나 학습 데이터와 많이 다른 문장일 수 있어요.</p>' : ''}
    <p class="compare">${esc(comparator.name)}: ${LABEL[comparator.decision]} ${pct(comparator.p)}
      <span class="${agree ? 'ok' : 'bad'}">${agree ? '초파리와 같은 판단' : '초파리와 다른 판단'}</span></p>
    ${truth}`;
  $('#result').querySelectorAll('.votes button').forEach((b) => b.addEventListener('click', () => selectFly(Number(b.dataset.fly))));
}

// --- batch mode --------------------------------------------------------------------------------
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
  setDots(1);
  const s = { i: 0, fly: 0, comp: 0, tp: 0, fp: 0, tn: 0, fn: 0, last: performance.now() };
  state.batchRun = s;
  $('#batch-go').textContent = '멈추기';
  $('#batch-split').textContent = state.batch.split === 'test' ? 'test' : 'validation';
  $('#batch-fly-name').textContent = key === 'hyb' ? `초파리 ${noseInfo().flies}마리 투표 (${noseInfo().label})` : '초파리 1마리 (고정 MiniLM 코)';
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
  if (tab === 'physics') loadVideos();
  if (tab !== 'batch') stopBatch();
  if (tab === 'live') {
    setDots(noseInfo().flies);
    maze.setCount(noseInfo().flies);
    maze.reset();
  }
}

function describeSource(layout) {
  if (layout.source.startsWith('flywire')) return `실제 FlyWire 배선 (오른쪽 반구, KC ${layout.kc.length.toLocaleString()}개, 사구체 ${layout.glom.length}개)`;
  return `무작위 배선 (KC ${layout.kc.length.toLocaleString()}개, 사구체 ${layout.glom.length}개, 모식도 좌표)`;
}

async function boot() {
  applyTheme(initialTheme());
  try {
    [state.brain, state.samples, state.noses] = await Promise.all([api('/api/brain'), api('/api/samples'), api('/api/noses')]);
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
  maze = new Maze($('#maze'), { onEvidence: setEvidence });
  brain = new Brain($('#brain'), state.brain);
  circuit = new Circuit($('#circuit'), state.brain);
  window.flyguard = { maze, brain, circuit, state };   // handle for debugging in the console
  $('#wiring').textContent = describeSource(state.brain);
  renderNoses();
  renderChips();

  $('#go').addEventListener('click', () => run($('#prompt').value));
  $('#prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) run($('#prompt').value);
  });
  $('#sample-filter').addEventListener('change', renderChips);
  $('#batch-go').addEventListener('click', startBatch);
  $('#theme').addEventListener('click', () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
  document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => setTab(b.dataset.tab)));
  setTab('live');
}

boot();
