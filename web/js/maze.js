// T-maze (Tully & Quinn 1985): the prompt's "odour" is released at the junction and each fly walks to
// the left arm (pass to the LLM) or the right arm (quarantine). A walk is a Brownian bridge that
// always ends on the side that fly's mushroom body chose, and a fly deliberates longer when its
// margin is small, as real flies do on hard odour discriminations (DasGupta et al. 2014). With the
// hybrid nose up to five flies smell the same prompt and vote.
import * as THREE from 'three';
import { clamp, css, dotTexture, ease, hash, mulberry32, textSprite } from './util.js';

const W = 1.7;       // corridor width
const STEM = 5.4;    // start arm
const ARM = 5.6;     // half length of the cross bar
const WALL_H = 0.5;
const PILE_MAX = 2400;
const MAX_FLIES = 5;
// Where each fly stands in the start arm when several smell together: [x, distance behind the front fly].
const FORMATION = [[0, 0], [-0.45, 0.6], [0.45, 0.6], [-0.45, 1.25], [0.45, 1.25]];

function evidencePath(seed, steps, target) {
  const rnd = mulberry32(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const w = [0];
  for (let i = 1; i <= steps; i++) w.push(w[i - 1] + gauss() / Math.sqrt(steps));
  return w.map((wi, i) => {
    if (i === steps) return target;
    const t = i / steps;
    return clamp(t * target + 0.6 * (wi - t * w[steps]), -0.96, 0.96);
  });
}

function segment(a, b, radius, mat) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const m = new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.7, radius, dir.length(), 8), mat);
  m.position.copy(a).addScaledVector(dir, 0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  m.castShadow = true;
  return m;
}

function stripeTexture() {
  const c = document.createElement('canvas');
  c.width = 32;
  c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = '#c0925a';
  g.fillRect(0, 0, 32, 256);
  g.fillStyle = '#3a2716';
  for (let i = 0; i < 5; i++) g.fillRect(0, 48 + i * 36, 32, 15);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// A procedural Drosophila: head toward -z, six legs in a tripod gait, red compound eyes.
function makeFly() {
  const g = new THREE.Group();
  const sphere = new THREE.SphereGeometry(1, 28, 18);
  const cuticle = new THREE.MeshStandardMaterial({ color: 0xa47c4a, roughness: 0.55 });
  const legMat = new THREE.MeshStandardMaterial({ color: 0x4d3621, roughness: 0.7 });
  const eyeMat = new THREE.MeshStandardMaterial({ color: 0xc3261b, roughness: 0.3, emissive: 0x300000 });
  const wingMat = new THREE.MeshStandardMaterial({
    color: 0xe6eef7, transparent: true, opacity: 0.38, roughness: 0.15, side: THREE.DoubleSide, depthWrite: false,
  });
  const blob = (mat, scale, pos) => {
    const m = new THREE.Mesh(sphere, mat);
    m.scale.set(...scale);
    m.position.set(...pos);
    m.castShadow = true;
    g.add(m);
    return m;
  };
  blob(cuticle, [0.25, 0.21, 0.29], [0, 0.34, 0]);                       // thorax
  const abdomen = blob(new THREE.MeshStandardMaterial({ map: stripeTexture(), roughness: 0.6 }),
    [0.21, 0.37, 0.18], [0, 0.31, 0.52]);
  abdomen.rotation.x = Math.PI / 2 + 0.1;
  blob(cuticle, [0.16, 0.15, 0.13], [0, 0.38, -0.36]);                   // head
  for (const s of [-1, 1]) blob(eyeMat, [0.095, 0.12, 0.1], [s * 0.12, 0.4, -0.38]);

  const antennae = [-1, 1].map((s) => {
    const a = segment(new THREE.Vector3(s * 0.04, 0.42, -0.45), new THREE.Vector3(s * 0.07, 0.47, -0.58), 0.018, legMat);
    g.add(a);
    return a;
  });

  const wingShape = new THREE.Shape();
  wingShape.absellipse(0, 0.42, 0.15, 0.45, 0, Math.PI * 2);
  const wingGeo = new THREE.ShapeGeometry(wingShape, 24);
  const wings = [-1, 1].map((s) => {
    const pivot = new THREE.Group();
    pivot.position.set(s * 0.05, 0.53, -0.02);
    pivot.rotation.y = s * 0.36;
    const w = new THREE.Mesh(wingGeo, wingMat);
    w.rotation.x = Math.PI / 2;
    pivot.add(w);
    g.add(pivot);
    return pivot;
  });

  const legs = [];
  [[-0.12, -0.34], [0.0, 0.0], [0.12, 0.34]].forEach(([z, footZ], pair) => {
    for (const s of [-1, 1]) {
      const hip = new THREE.Group();
      hip.position.set(s * 0.15, 0.27, z);
      const knee = new THREE.Vector3(s * 0.25, 0.1, footZ * 0.45);
      const foot = new THREE.Vector3(s * 0.45, -0.26, footZ);
      hip.add(segment(new THREE.Vector3(0, 0, 0), knee, 0.024, legMat));
      hip.add(segment(knee, foot, 0.018, legMat));
      g.add(hip);
      legs.push({ hip, side: s, tripod: (pair % 2 === 0) === (s < 0) ? 0 : 1 });
    }
  });

  return {
    group: g,
    phase: 0,
    step(dPhase) {
      this.phase += dPhase;
      for (const l of legs) {
        const ph = this.phase + (l.tripod ? Math.PI : 0);
        l.hip.rotation.y = 0.32 * Math.sin(ph);
        l.hip.rotation.z = l.side * 0.2 * Math.max(0, Math.cos(ph));
      }
    },
    sniff(t) {
      antennae.forEach((a, i) => { a.rotation.z = 0.3 * Math.sin(t * 17 + i * 1.7); });
      wings.forEach((w, i) => { w.rotation.z = 0.04 * Math.sin(t * 9 + i); });
    },
  };
}

export class Maze {
  constructor(canvas, { onEvidence } = {}) {
    this.canvas = canvas;
    this.onEvidence = onEvidence || (() => {});
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x3a4250, 1.5));
    const sun = new THREE.DirectionalLight(0xffffff, 1.7);
    sun.position.set(3, 11, 6);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    Object.assign(sun.shadow.camera, { left: -8, right: 8, top: 8, bottom: -8 });
    this.scene.add(sun);

    this.buildMaze();
    this.flies = Array.from({ length: MAX_FLIES }, () => makeFly());
    this.flies.forEach((f) => this.scene.add(f.group));
    this.fly = this.flies[0];
    this.count = 1;
    this.buildOdor();
    this.buildPile();
    this.applyTheme();

    this.time = 0;
    this.last = 0;
    this.anim = null;
    this.flick = null;
    this.reset();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
    this.setActive(true);
  }

  setActive(on) {
    this.renderer.setAnimationLoop(on ? (now) => this.tick(now) : null);
    this.last = 0;
  }

  buildMaze() {
    this.floorMat = new THREE.MeshStandardMaterial({ roughness: 0.95 });
    const floor = (w, d, x, z) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), this.floorMat);
      m.rotation.x = -Math.PI / 2;
      m.position.set(x, 0, z);
      m.receiveShadow = true;
      this.scene.add(m);
    };
    floor(W, STEM - W / 2, 0, (STEM + W / 2) / 2);
    floor(2 * ARM, W, 0, 0);

    const ground = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.ShadowMaterial({ opacity: 0.25 }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.002;
    ground.receiveShadow = true;
    this.scene.add(ground);

    // Arm tints that brighten toward the arm's end.
    this.armTints = {};
    for (const [side, sign] of [['pass', -1], ['quarantine', 1]]) {
      const c = document.createElement('canvas');
      c.width = 256;
      c.height = 4;
      const g = c.getContext('2d');
      const grd = g.createLinearGradient(0, 0, 256, 0);
      grd.addColorStop(0, 'rgba(255,255,255,0)');
      grd.addColorStop(1, 'rgba(255,255,255,1)');
      g.fillStyle = grd;
      g.fillRect(0, 0, 256, 4);
      const tex = new THREE.CanvasTexture(c);
      const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.35, depthWrite: false });
      const len = ARM - W / 2;
      const m = new THREE.Mesh(new THREE.PlaneGeometry(len, W), mat);
      m.rotation.x = -Math.PI / 2;
      if (sign < 0) m.rotation.z = Math.PI;      // gradient grows toward the far end
      m.position.set(sign * (W / 2 + len / 2), 0.003, 0);
      this.scene.add(m);
      this.armTints[side] = mat;
    }

    this.wallMat = new THREE.MeshStandardMaterial({ transparent: true, opacity: 0.2, roughness: 0.1 });
    this.edgeMat = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.55 });
    const wall = (x1, z1, x2, z2) => {
      const len = Math.hypot(x2 - x1, z2 - z1);
      const geo = new THREE.BoxGeometry(len, WALL_H, 0.05);
      const m = new THREE.Mesh(geo, this.wallMat);
      m.position.set((x1 + x2) / 2, WALL_H / 2, (z1 + z2) / 2);
      m.rotation.y = -Math.atan2(z2 - z1, x2 - x1);
      m.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), this.edgeMat));
      this.scene.add(m);
    };
    wall(-W / 2, W / 2, -W / 2, STEM);
    wall(W / 2, W / 2, W / 2, STEM);
    wall(-W / 2, STEM, W / 2, STEM);
    wall(-ARM, W / 2, -W / 2, W / 2);
    wall(W / 2, W / 2, ARM, W / 2);
    wall(-ARM, -W / 2, ARM, -W / 2);
    wall(-ARM, -W / 2, -ARM, W / 2);
    wall(ARM, -W / 2, ARM, W / 2);

    this.labels = new THREE.Group();
    this.scene.add(this.labels);
  }

  buildOdor() {
    const n = 240;
    this.odor = { n, pos: new Float32Array(n * 3), vel: new Float32Array(n * 3), life: new Float32Array(n), level: 0 };
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.odor.pos, 3));
    this.odorMat = new THREE.PointsMaterial({
      size: 0.16, map: dotTexture, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.odorPoints = new THREE.Points(geo, this.odorMat);
    this.scene.add(this.odorPoints);
    for (let i = 0; i < n; i++) this.respawn(i, Math.random());
  }

  respawn(i, age = 0) {
    const { pos, vel, life } = this.odor;
    const a = Math.random() * Math.PI * 2;
    const r = Math.random() * 0.3;
    pos.set([Math.cos(a) * r, 0.15 + Math.random() * 0.25, -0.1 + Math.sin(a) * r], i * 3);
    vel.set([Math.cos(a) * 0.35, 0.08 + Math.random() * 0.12, Math.sin(a) * 0.12], i * 3);
    life[i] = age;
  }

  buildPile() {
    this.pileMat = new THREE.MeshStandardMaterial({ roughness: 0.5 });
    this.pile = new THREE.InstancedMesh(new THREE.SphereGeometry(0.034, 8, 6), this.pileMat, PILE_MAX);
    this.pile.count = 0;
    this.pile.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.scene.add(this.pile);
    this.pileCount = { pass: 0, quarantine: 0 };
  }

  applyTheme() {
    this.colors = {
      pass: new THREE.Color(css('--pass')),
      block: new THREE.Color(css('--block')),
      accent: new THREE.Color(css('--accent')),
      text: css('--text'),
    };
    this.floorMat.color.set(css('--maze-floor'));
    this.wallMat.color.set(css('--maze-wall'));
    this.edgeMat.color.set(css('--maze-wall'));
    this.armTints.pass.color.copy(this.colors.pass);
    this.armTints.quarantine.color.copy(this.colors.block);
    this.odorMat.color.copy(this.colors.accent);
    this.labels.clear();
    const place = (sprite, x, y, z) => { sprite.position.set(x, y, z); this.labels.add(sprite); };
    place(textSprite('LLM으로 전달', { color: css('--pass'), height: 0.42 }), -ARM + 1.5, 0.95, -0.35);
    place(textSprite('격리', { color: css('--block'), height: 0.42 }), ARM - 1.0, 0.95, -0.35);
    place(textSprite('출발', { color: css('--muted'), height: 0.3 }), 0, 0.7, STEM - 0.3);
    this.flies.forEach((f, i) => {                     // number tags, shown when several flies vote
      if (f.tag) f.group.remove(f.tag);
      f.tag = textSprite(String(i + 1), { color: css('--text'), background: 'rgba(0,0,0,0.35)', height: 0.42 });
      f.tag.position.set(0, 1.05, 0);
      f.tag.visible = this.count > 1;
      f.group.add(f.tag);
    });
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.canvas.parentElement;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    const far = Math.max(1, 1.45 / this.camera.aspect);
    this.camera.position.set(0, 10.8 * far, 8.4 * far);
    this.camera.lookAt(0, 0, 2.2);
    this.camera.updateProjectionMatrix();
  }

  setCount(n) {
    this.count = n;
    this.flies.forEach((f, i) => {
      f.group.visible = i < n;
      f.group.scale.setScalar(n > 1 ? 0.62 : 1);
      f.tag.visible = n > 1;
    });
  }

  // Where fly i starts, and where it stops to smell the prompt.
  home(i) {
    const [x, back] = this.count > 1 ? FORMATION[i] : [0, 0];
    return { x, start: STEM - 0.7 - (this.count > 1 ? 1.25 - back : 0), smell: 0.75 + back };
  }

  // End position in an arm: the first fly to arrive takes the far end, later ones queue behind it.
  armSlot(sign, slot) {
    if (this.count === 1) return new THREE.Vector3(sign * (ARM - 1.0), 0, 0);
    return new THREE.Vector3(sign * (ARM - 0.7 - Math.floor(slot / 2) * 0.6), 0, slot % 2 ? 0.35 : -0.35);
  }

  reset() {
    this.anim = null;
    this.flick = null;
    this.flies.forEach((f, i) => {
      const h = this.home(i);
      this.place(f, h.x, h.start, 0);
    });
    this.odor.level = 0;
    this.armTints.pass.opacity = this.armTints.quarantine.opacity = 0.35;
    if (this.promptSprite) this.scene.remove(this.promptSprite);
    this.promptSprite = null;
    for (let i = 0; i < this.count; i++) this.onEvidence(i, 0);
  }

  place(fly, x, z, heading) {
    fly.group.position.set(x, 0, z);
    fly.group.rotation.y = heading;
  }

  // Walk every fly through its own decision; resolves when all of them have reached an arm.
  // `verdict` is the vote ('injection' or 'benign'); its arm glows at the end.
  run(text, traces, verdict) {
    this.setCount(traces.length);
    this.reset();
    this.promptSprite = textSprite(`“${text}”`, {
      color: this.colors.text, background: 'rgba(242,179,61,0.16)', height: 0.26, maxWidth: 1100, maxLines: 2,
    });
    this.promptSprite.position.set(0, 1.55, -0.45);
    this.scene.add(this.promptSprite);
    return new Promise((resolve) => {
      this.anim = {
        resolve, verdict, t: 0, finished: false, slots: { '-1': 0, 1: 0 },
        flies: traces.map((tr, i) => {
          const target = tr.decision === 'injection' ? 1 : -1;
          return {
            phase: 'walk', t: 0, target, lastE: 0,
            decide: clamp(0.9 / Math.max(Math.abs(tr.margin), 1e-3), 0.4, 3.0),
            path: evidencePath(hash(`${i}:${text}`), 120, target),
          };
        }),
      };
    });
  }

  // Batch mode: drop one prompt into the arm the fly chose, coloured by its true label.
  drop(side, label) {
    const i = this.pile.count;
    if (i >= PILE_MAX) return;
    const k = this.pileCount[side]++;
    const cols = 18, rows = 26, gap = 0.08;
    const layer = Math.floor(k / (cols * rows));
    const r = Math.floor((k % (cols * rows)) / cols);
    const c = k % cols;
    const sign = side === 'pass' ? -1 : 1;
    const m = new THREE.Matrix4().makeTranslation(
      sign * (ARM - 0.2 - r * gap), 0.035 + layer * 0.06, -W / 2 + 0.17 + c * gap,
    );
    this.pile.setMatrixAt(i, m);
    this.pile.setColorAt(i, label ? this.colors.block : this.colors.pass);
    this.pile.count = i + 1;
    this.pile.instanceMatrix.needsUpdate = true;
    this.pile.instanceColor.needsUpdate = true;
    this.flick = { sign, t: 0 };
  }

  clearPile() {
    this.pile.count = 0;
    this.pileCount = { pass: 0, quarantine: 0 };
    this.setCount(1);
    this.reset();
    this.place(this.fly, 0, 0.45, 0);
  }

  tick(now) {
    const dt = this.last ? clamp((now - this.last) / 1000, 0, 0.05) : 0;
    this.last = now;
    this.time += dt;
    for (let i = 0; i < this.count; i++) this.flies[i].sniff(this.time + i);
    this.tickOdor(dt);
    if (this.anim) this.tickAnim(dt);
    if (this.flick) {
      this.flick.t += dt;
      const k = Math.sin(Math.min(this.flick.t / 0.25, 1) * Math.PI);
      this.fly.group.rotation.y = -this.flick.sign * 0.7 * k;
      if (this.flick.t > 0.25) this.flick = null;
    }
    this.renderer.render(this.scene, this.camera);
  }

  tickOdor(dt) {
    const o = this.odor;
    const want = this.anim && this.anim.flies.some((f) => f.phase === 'walk' || f.phase === 'decide') ? 1 : 0;
    o.level += (want - o.level) * Math.min(1, dt * 3);
    this.odorMat.opacity = 0.55 * o.level;
    if (o.level < 0.01) return;
    for (let i = 0; i < o.n; i++) {
      o.life[i] += dt / 2.2;
      if (o.life[i] > 1) this.respawn(i);
      for (let j = 0; j < 3; j++) o.pos[i * 3 + j] += o.vel[i * 3 + j] * dt;
    }
    this.odorPoints.geometry.attributes.position.needsUpdate = true;
  }

  tickAnim(dt) {
    const a = this.anim;
    a.flies.forEach((s, i) => {
      const fly = this.flies[i];
      const h = this.home(i);
      s.t += dt;
      if (s.phase === 'walk') {
        const k = Math.min(s.t / 1.3, 1);
        this.place(fly, h.x, h.start + (h.smell - h.start) * ease(k), 0);
        fly.step(dt * 15);
        if (k >= 1) Object.assign(s, { phase: 'decide', t: 0 });
      } else if (s.phase === 'decide') {
        const k = Math.min(s.t / s.decide, 1);
        const x = k * (s.path.length - 1);
        const j = Math.floor(x);
        const e = s.path[j] + (s.path[Math.min(j + 1, s.path.length - 1)] - s.path[j]) * (x - j);
        this.place(fly, h.x, h.smell - 0.3 * k, -0.65 * e);
        fly.step(dt * 120 * Math.abs(e - s.lastE));
        s.lastE = e;
        this.onEvidence(i, e);
        if (k >= 1) {
          const end = this.armSlot(s.target, a.slots[s.target]++);
          Object.assign(s, { phase: 'commit', t: 0, from: fly.group.position.clone(), end });
        }
      } else if (s.phase === 'commit') {
        const k = ease(Math.min(s.t / 1.6, 1));
        const p0 = s.from, p1 = new THREE.Vector3(h.x * 0.5, 0, -0.05), p2 = s.end;
        const pos = p0.clone().multiplyScalar((1 - k) ** 2).addScaledVector(p1, 2 * (1 - k) * k).addScaledVector(p2, k * k);
        const tan = p1.clone().sub(p0).multiplyScalar(2 * (1 - k)).addScaledVector(p2.clone().sub(p1), 2 * k);
        this.place(fly, pos.x, pos.z, Math.atan2(-tan.x, -tan.z));
        fly.step(dt * 15);
        this.onEvidence(i, s.target);
        if (s.t >= 1.6) s.phase = 'done';
      }
    });
    if (!a.finished && a.flies.every((s) => s.phase === 'done')) {
      a.finished = true;
      a.resolve();
    }
    if (a.finished) {
      a.t += dt;
      const tint = a.verdict === 'injection' ? this.armTints.quarantine : this.armTints.pass;
      tint.opacity = 0.45 + 0.25 * Math.sin(a.t * 5) ** 2;
    }
  }
}
