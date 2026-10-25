// The sorting stage: prompt cards ride a conveyor toward one fly, who catches each card with its front
// legs, smells it, and then either pushes it up into the LLM portal (benign) or slams it down into the
// quarantine grate (injection). Neon city and bloom for the look; callbacks drive the HUD.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { POSES, lerpPose, makeFly } from './fly.js';
import { clamp, dotTexture, ease, mulberry32, textSprite } from './util.js';

const NEON = { cyan: 0x38e8ff, pink: 0xff3d9a, green: 0x3dffa2, red: 0xff4d5e };
const BELT = { start: -16, end: -2.55, y: 0.07, speed: 2.6 };
const CARD = { w: 1.3, h: 0.74, gap: 0.4 };
const HOLD = new THREE.Vector3(-0.3, 1.15, -1.65);
const PORTAL = new THREE.Vector3(0, 2.35, -2.75);
const GRATE = new THREE.Vector3(0, 0.02, -1.72);
const MAX_PARTICLES = 900;

const glow = (hex, k) => new THREE.Color(hex).multiplyScalar(k);   // > 1 so bloom picks it up

function wrapLines(g, text, maxWidth) {
  const lines = [];
  let line = '';
  for (const word of text.split(/\s+/)) {
    const next = line ? `${line} ${word}` : word;
    if (g.measureText(next).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines;
}

function cardTexture(text, label) {
  const c = document.createElement('canvas');
  c.width = 640;
  c.height = 364;
  const g = c.getContext('2d');
  g.fillStyle = 'rgba(6, 12, 26, 0.95)';
  g.beginPath();
  g.roundRect(0, 0, 640, 364, 26);
  g.fill();
  g.fillStyle = '#38e8ff';
  g.font = '700 26px Consolas, "D2Coding", monospace';
  g.fillText(label, 32, 52);
  g.fillStyle = '#e8f4ff';
  g.font = '500 33px Consolas, "D2Coding", "Malgun Gothic", monospace';
  let lines = wrapLines(g, text, 576);
  if (lines.length > 6) lines = [...lines.slice(0, 5), `${lines[5]}…`];
  lines.forEach((l, i) => g.fillText(l, 32, 104 + i * 43));
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

function canvasTexture(w, h, draw, repeat = false) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

export class Stage {
  constructor(canvas) {
    this.canvas = canvas;
    this.on = { inspect: () => {}, decide: () => {}, sorted: () => {}, frame: () => {} };
    this.speed = 1;
    this.belt = [];
    this.current = null;
    this.time = 0;
    this.last = 0;
    this.rand = mulberry32(7);

    const r = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    r.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 0.9;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060914);
    this.scene.fog = new THREE.FogExp2(0x0a0f22, 0.03);
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.1, 200);

    this.composer = new EffectComposer(r);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.7, 0.45, 0.78);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.buildLights();
    this.buildCity();
    this.buildDesk();
    this.buildTargets();
    this.buildParticles();
    this.fly = makeFly();
    this.fly.group.scale.setScalar(2.0);
    this.scene.add(this.fly.group);
    this.fly.setPose(POSES.rest);

    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
  }

  buildLights() {
    const s = this.scene;
    s.add(new THREE.HemisphereLight(0x8a9cff, 0x0a0b18, 0.35));
    const key = new THREE.DirectionalLight(0xffffff, 1.3);
    key.position.set(-4, 6, 1.5);
    s.add(key);
    const cyan = new THREE.PointLight(NEON.cyan, 6, 12, 1.6);
    cyan.position.set(1.5, 2.2, -3.5);
    const pink = new THREE.PointLight(NEON.pink, 5, 10, 1.6);
    pink.position.set(1.2, 1.4, 2.2);
    s.add(cyan, pink);
    this.flash = { red: new THREE.PointLight(NEON.red, 0, 6, 1.5), green: new THREE.PointLight(NEON.green, 0, 6, 1.5) };
    this.flash.red.position.copy(GRATE).add(new THREE.Vector3(-0.4, 0.5, 0));
    this.flash.green.position.copy(PORTAL);
    s.add(this.flash.red, this.flash.green);
  }

  buildCity() {
    const rand = this.rand;
    const windows = canvasTexture(64, 128, (g, w, h) => {
      g.fillStyle = '#05070d';
      g.fillRect(0, 0, w, h);
      for (let y = 4; y < h; y += 10) {
        for (let x = 4; x < w; x += 10) {
          const p = rand();
          if (p < 0.45) continue;
          g.fillStyle = p < 0.75 ? '#ffcf7a' : p < 0.92 ? '#79d8ff' : '#ff7ad1';
          g.globalAlpha = 0.35 + rand() * 0.65;
          g.fillRect(x, y, 5, 6);
        }
      }
      g.globalAlpha = 1;
    }, true);
    const mats = [0.22, 0.4, 0.65].map((k) => new THREE.MeshStandardMaterial({
      color: 0x0a0e1a, roughness: 0.85, metalness: 0.1, emissive: 0xffffff, emissiveMap: windows, emissiveIntensity: k,
    }));
    const strip = [glow(NEON.pink, 3), glow(NEON.cyan, 3)].map((c) => new THREE.MeshBasicMaterial({ color: c, toneMapped: false }));
    for (let i = 0; i < 90; i++) {
      const w = 1 + rand() * 2.8, d = 1 + rand() * 2.8, h = 3 + rand() ** 2 * 17;
      const x = 6 + rand() * 30, z = -38 + rand() * 56;
      const geo = new THREE.BoxGeometry(w, h, d);
      const uv = geo.attributes.uv;
      for (let j = 0; j < uv.count; j++) uv.setXY(j, uv.getX(j) * Math.max(w, d) / 1.4, uv.getY(j) * h / 2.6);
      const b = new THREE.Mesh(geo, mats[Math.floor(rand() * mats.length)]);
      b.position.set(x, -1.2 + h / 2, z);
      this.scene.add(b);
      if (rand() < 0.35) {
        const s = new THREE.Mesh(new THREE.BoxGeometry(0.07, h * (0.4 + rand() * 0.5), 0.07), strip[i % 2]);
        s.position.set(x - w / 2 - 0.04, -1.2 + h * 0.55, z + (rand() - 0.5) * d);
        this.scene.add(s);
      }
    }
    const sky = canvasTexture(4, 256, (g, w, h) => {
      const grd = g.createLinearGradient(0, 0, 0, h);
      grd.addColorStop(0, '#05060f');
      grd.addColorStop(0.55, '#1a0f38');
      grd.addColorStop(1, '#2a1240');
      g.fillStyle = grd;
      g.fillRect(0, 0, w, h);
    });
    const backdrop = new THREE.Mesh(new THREE.PlaneGeometry(260, 70), new THREE.MeshBasicMaterial({ map: sky, fog: false }));
    backdrop.position.set(60, 20, -5);
    backdrop.rotation.y = -Math.PI / 2;
    this.scene.add(backdrop);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshStandardMaterial({ color: 0x05070e, roughness: 0.35, metalness: 0.6 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -1.2;
    this.scene.add(floor);
    const grid = new THREE.GridHelper(120, 120, NEON.pink, 0x1b2a55);
    grid.position.y = -1.19;
    grid.material.transparent = true;
    grid.material.opacity = 0.35;
    this.scene.add(grid);
  }

  buildDesk() {
    const desk = new THREE.Mesh(new THREE.BoxGeometry(2.3, 1.2, 20.5),
      new THREE.MeshStandardMaterial({ color: 0x0c101b, roughness: 0.55, metalness: 0.5 }));
    desk.position.set(0, -0.6, -6.8);
    this.scene.add(desk);
    const rail = new THREE.MeshBasicMaterial({ color: glow(NEON.cyan, 1.5), toneMapped: false });
    for (const x of [-1.15, 1.15]) {
      const r = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.035, 20.5), rail);
      r.position.set(x, 0.0, -6.8);
      this.scene.add(r);
    }
    const beltLen = BELT.end - BELT.start;
    this.beltTex = canvasTexture(64, 64, (g, w, h) => {
      g.fillStyle = '#0a1222';
      g.fillRect(0, 0, w, h);
      g.strokeStyle = 'rgba(56,232,255,0.55)';
      g.lineWidth = 4;
      g.beginPath();
      g.moveTo(8, 40);
      g.lineTo(32, 20);
      g.lineTo(56, 40);
      g.stroke();
    }, true);
    this.beltTex.repeat.set(1, beltLen / 0.9);
    const belt = new THREE.Mesh(new THREE.PlaneGeometry(1.05, beltLen),
      new THREE.MeshBasicMaterial({ map: this.beltTex, color: 0x9fb4d8 }));
    belt.rotation.x = -Math.PI / 2;
    belt.position.set(0, BELT.y - 0.01, (BELT.start + BELT.end) / 2);
    this.scene.add(belt);
  }

  buildTargets() {
    // LLM portal: a glowing ring the benign cards fly into.
    this.portal = new THREE.Group();
    this.portal.position.copy(PORTAL);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.58, 0.045, 16, 72),
      new THREE.MeshBasicMaterial({ color: glow(NEON.green, 1.8), toneMapped: false }));
    ring.rotation.y = Math.PI / 2;
    const disc = new THREE.Mesh(new THREE.CircleGeometry(0.56, 48), new THREE.MeshBasicMaterial({
      map: canvasTexture(128, 128, (g) => {
        const grd = g.createRadialGradient(64, 64, 4, 64, 64, 64);
        grd.addColorStop(0, 'rgba(61,255,162,0.3)');
        grd.addColorStop(1, 'rgba(61,255,162,0.02)');
        g.fillStyle = grd;
        g.fillRect(0, 0, 128, 128);
      }), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    disc.rotation.y = -Math.PI / 2;
    this.portal.add(ring, disc);
    this.portalRing = ring;
    this.scene.add(this.portal);

    // Quarantine grate in the desk between the belt and the fly.
    const grate = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 1.45), new THREE.MeshBasicMaterial({
      map: canvasTexture(128, 128, (g, w, h) => {
        g.fillStyle = '#12030a';
        g.fillRect(0, 0, w, h);
        g.strokeStyle = '#ff4d5e';
        g.lineWidth = 3;
        for (let i = 8; i < w; i += 16) {
          g.beginPath();
          g.moveTo(i, 0);
          g.lineTo(i, h);
          g.stroke();
        }
        g.strokeRect(2, 2, w - 4, h - 4);
      }), color: glow(0xffffff, 1.6), toneMapped: false,
    }));
    grate.rotation.x = -Math.PI / 2;
    grate.position.copy(GRATE);
    this.scene.add(grate);
    this.grate = grate;

    const label = (text, color, pos) => {
      const s = textSprite(text, { color, height: 0.2 });
      s.position.copy(pos);
      this.scene.add(s);
    };
    label('LLM으로 전달', '#3dffa2', PORTAL.clone().add(new THREE.Vector3(0, 0.8, 0)));
    label('격리 · QUARANTINE', '#ff6b78', GRATE.clone().add(new THREE.Vector3(-0.85, 0.12, 0)));
  }

  buildParticles() {
    // Floating neon dust for depth.
    const n = 360, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos.set([-2 + this.rand() * 22, -1 + this.rand() * 9, -22 + this.rand() * 30], i * 3);
      const c = new THREE.Color(this.rand() < 0.5 ? NEON.cyan : NEON.pink);
      col.set([c.r, c.g, c.b], i * 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    this.dust = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 0.07, map: dotTexture, vertexColors: true, transparent: true, opacity: 0.6,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.scene.add(this.dust);

    // Bursts: shattered injections and benign trails share one pool.
    this.fx = { pos: new Float32Array(MAX_PARTICLES * 3), vel: new Float32Array(MAX_PARTICLES * 3),
                col: new Float32Array(MAX_PARTICLES * 3), life: new Float32Array(MAX_PARTICLES), next: 0 };
    for (let i = 0; i < MAX_PARTICLES; i++) this.fx.pos[i * 3 + 1] = -50;   // parked below the floor
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.fx.pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(this.fx.col, 3));
    this.fxPoints = new THREE.Points(g, new THREE.PointsMaterial({
      size: 0.09, map: dotTexture, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.fxPoints.frustumCulled = false;   // particles roam far from the bounds computed at start
    this.scene.add(this.fxPoints);
  }

  emit(at, color, count, speed, up = 0) {
    const f = this.fx, c = new THREE.Color(color);
    for (let i = 0; i < count; i++) {
      const j = f.next;
      f.next = (f.next + 1) % MAX_PARTICLES;
      const a = this.rand() * Math.PI * 2, b = (this.rand() - 0.5) * Math.PI;
      const v = speed * (0.3 + this.rand());
      f.pos.set([at.x + (this.rand() - 0.5) * 0.3, at.y + (this.rand() - 0.5) * 0.3, at.z + (this.rand() - 0.5) * 0.6], j * 3);
      f.vel.set([Math.cos(a) * Math.cos(b) * v * 0.5, Math.sin(b) * v + up, Math.sin(a) * Math.cos(b) * v], j * 3);
      const k = 0.7 + this.rand() * 0.8;
      f.col.set([c.r * k * 2, c.g * k * 2, c.b * k * 2], j * 3);
      f.life[j] = 0.8 + this.rand() * 0.7;
    }
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.canvas.parentElement;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setActive(on) {
    this.renderer.setAnimationLoop(on ? (now) => this.tick(now) : null);
    this.last = 0;
  }

  // Put a classified prompt on the belt. `res` is the /api/classify answer.
  add(item, res) {
    const group = new THREE.Group();
    const label = item.tag ? `PROMPT · ${item.tag}` : 'PROMPT';
    const rim = new THREE.Mesh(new THREE.PlaneGeometry(CARD.w + 0.06, CARD.h + 0.06),
      new THREE.MeshBasicMaterial({ color: glow(NEON.cyan, 2.2), toneMapped: false, transparent: true }));
    rim.position.z = -0.004;
    const face = new THREE.Mesh(new THREE.PlaneGeometry(CARD.w, CARD.h),
      new THREE.MeshBasicMaterial({ map: cardTexture(item.text, label), transparent: true }));
    group.add(rim, face);
    group.rotation.y = -Math.PI / 2;
    const behind = this.belt.length ? this.belt[this.belt.length - 1].group.position.z - (CARD.w + CARD.gap) : BELT.start;
    group.position.set(0, BELT.y + CARD.h / 2, Math.min(BELT.start, behind));
    this.scene.add(group);
    this.belt.push({ item, res, group, rim, face });
  }

  get load() {
    return this.belt.length + (this.current ? 1 : 0);
  }

  clear() {
    for (const c of [...this.belt, this.current].filter(Boolean)) this.scene.remove(c.group);
    this.belt = [];
    this.current = null;
    this.fly.setPose(POSES.rest);
  }

  // One card in the fly's legs: catch -> smell -> push up or slam down -> recover.
  handle(dt) {
    const c = this.current;
    c.t += dt;
    const k = (d) => clamp(c.t / d, 0, 1);
    const fly = this.fly;
    if (c.phase === 'catch') {
      const e = ease(k(0.45));
      fly.setPose(lerpPose(POSES.rest, POSES.hold, e));
      c.group.position.lerpVectors(c.from, HOLD, e);
      c.group.position.y += Math.sin(e * Math.PI) * 0.25;
      if (c.t >= 0.45) {
        Object.assign(c, { phase: 'sniff', t: 0 });
        this.on.inspect(c.item, c.res);
      }
    } else if (c.phase === 'sniff') {
      c.group.position.y = HOLD.y + 0.02 * Math.sin(this.time * 9);
      if (c.t >= c.decide) {
        const benign = c.res.vote.decision === 'benign';
        c.rim.material.color.copy(glow(benign ? NEON.green : NEON.red, 2.6));
        Object.assign(c, { phase: benign ? 'push' : 'windup', t: 0 });
        this.on.decide(c.item, c.res);
      }
    } else if (c.phase === 'push') {
      const e = k(0.8);
      fly.setPose(lerpPose(POSES.hold, POSES.push, ease(Math.min(e * 2, 1))));
      const p0 = HOLD, p1 = new THREE.Vector3(-0.2, 2.5, -1.5), p2 = PORTAL;
      const q = ease(e);
      c.group.position.copy(p0).multiplyScalar((1 - q) ** 2).addScaledVector(p1, 2 * (1 - q) * q).addScaledVector(p2, q * q);
      c.group.scale.setScalar(1 - 0.8 * q);
      c.face.material.opacity = c.rim.material.opacity = 1 - q * q;
      if (this.rand() < 0.6) this.emit(c.group.position, NEON.green, 2, 0.3, -0.2);
      if (c.t >= 0.8) {
        this.flash.green.intensity = 30;
        this.portalPulse = 1;
        this.emit(PORTAL, NEON.green, 50, 1.2);
        this.finish(c);
      }
    } else if (c.phase === 'windup') {
      const e = ease(k(0.32));
      fly.setPose(lerpPose(POSES.hold, POSES.raise, e));
      c.group.position.set(HOLD.x, HOLD.y + 0.25 * e, HOLD.z + 0.05 * e);
      c.group.rotation.z = 0.15 * e;
      if (c.t >= 0.32) Object.assign(c, { phase: 'slam', t: 0 });
    } else if (c.phase === 'slam') {
      const e = k(0.16) ** 2;
      fly.setPose(lerpPose(POSES.raise, POSES.slam, e));
      c.group.position.set(HOLD.x * (1 - e), HOLD.y + 0.25 + (GRATE.y + 0.15 - HOLD.y - 0.25) * e, HOLD.z + (GRATE.z - HOLD.z) * e);
      c.group.rotation.z = 0.15 - 0.6 * e;
      if (c.t >= 0.16) {
        this.flash.red.intensity = 40;
        this.gratePulse = 1;
        this.emit(GRATE.clone().add(new THREE.Vector3(0, 0.25, 0)), NEON.red, 140, 2.4, 1.2);
        this.emit(GRATE.clone().add(new THREE.Vector3(0, 0.25, 0)), 0xffb36b, 40, 1.6, 1.4);
        this.finish(c);
      }
    } else if (c.phase === 'recover') {
      const e = ease(k(0.5));
      fly.setPose(lerpPose(c.pose, POSES.rest, e));
      if (c.t >= 0.5) this.current = null;
    }
  }

  finish(c) {
    this.scene.remove(c.group);
    this.on.sorted(c.item, c.res);
    const pose = c.phase === 'push' ? POSES.push : POSES.slam;
    Object.assign(c, { phase: 'recover', t: 0, pose });
  }

  tick(now) {
    const real = this.last ? clamp((now - this.last) / 1000, 0, 0.05) : 0;
    this.last = now;
    const dt = real * this.speed;
    this.time += dt;

    // Belt: cards slide toward their slot behind the front of the queue.
    this.beltTex.offset.y -= dt * BELT.speed / 0.9;
    const front = BELT.end - CARD.w / 2;
    this.belt.forEach((c, i) => {
      let target = front - i * (CARD.w + CARD.gap);
      if (i > 0) target = Math.min(target, this.belt[i - 1].group.position.z - (CARD.w + CARD.gap));
      c.group.position.z = Math.min(target, c.group.position.z + BELT.speed * dt);
    });
    if (!this.current && this.belt.length && this.belt[0].group.position.z >= front - 1e-3) {
      const c = this.belt.shift();
      const margin = Math.abs(c.res.fly.margin);
      c.decide = clamp(0.45 + 0.3 / Math.max(margin, 0.15), 0.55, 1.8);   // hesitate on hard prompts
      Object.assign(c, { phase: 'catch', t: 0, from: c.group.position.clone() });
      this.current = c;
    }
    if (this.current) this.handle(dt);
    this.fly.animate(this.time, this.current?.phase === 'sniff' ? 1 : 0.15);

    // Effects.
    const f = this.fx;
    for (let i = 0; i < MAX_PARTICLES; i++) {
      if (f.life[i] <= 0) continue;
      f.life[i] -= dt;
      f.vel[i * 3 + 1] -= 3.2 * dt;
      for (let j = 0; j < 3; j++) f.pos[i * 3 + j] += f.vel[i * 3 + j] * dt;
      if (f.life[i] <= 0) f.pos.set([0, -50, 0], i * 3);
      else {
        const fade = Math.min(1, f.life[i] * 2) / Math.max(Math.min(1, (f.life[i] + dt) * 2), 1e-3);
        for (let j = 0; j < 3; j++) f.col[i * 3 + j] *= fade;
      }
    }
    this.fxPoints.geometry.attributes.position.needsUpdate = true;
    this.fxPoints.geometry.attributes.color.needsUpdate = true;
    this.flash.red.intensity *= Math.exp(-dt * 5);
    this.flash.green.intensity *= Math.exp(-dt * 5);
    this.portalPulse = (this.portalPulse || 0) * Math.exp(-dt * 4);
    this.gratePulse = (this.gratePulse || 0) * Math.exp(-dt * 4);
    this.portal.scale.setScalar(1 + 0.25 * this.portalPulse);
    this.portalRing.rotation.x = this.time * 0.6;
    this.grate.material.color.copy(glow(0xffffff, 1.6 + 3 * this.gratePulse));
    this.dust.rotation.y = Math.sin(this.time * 0.05) * 0.05;

    // Slow cinematic drift.
    const t = this.time * 0.25;
    this.camera.position.set(-7.6 + 0.2 * Math.sin(t), 3.0 + 0.1 * Math.sin(t * 1.3), -2.4 + 0.25 * Math.cos(t * 0.8));
    this.camera.lookAt(0, 0.85, -0.5);
    this.composer.render();
    this.on.frame(dt);
  }
}
