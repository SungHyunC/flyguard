// 3-D brain panel: the whole brain as faint dust, the mushroom-body neurons on top. For each prompt,
// activity flows in two waves: glomeruli -> active Kenyon cells, then Kenyon cells -> the MBON each
// one votes for (red = quarantine, green = pass).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { clamp, css, dotTexture, textSprite } from './util.js';

const MAX_SEGMENTS = 6000;

function points(positions, size, opacity = 1) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(positions.length), 3));
  const mat = new THREE.PointsMaterial({
    size, map: dotTexture, vertexColors: true, transparent: true, opacity, depthWrite: false, alphaTest: 0.05,
  });
  return new THREE.Points(geo, mat);
}

function lines() {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_SEGMENTS * 6), 3));
  geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(MAX_SEGMENTS * 6), 3));
  geo.setDrawRange(0, 0);
  const mat = new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  return new THREE.LineSegments(geo, mat);
}

export class Brain {
  // options.labels: name the parts; options.frame: 'circuit' (zoom on the mushroom body) or 'brain'.
  constructor(canvas, layout, options = {}) {
    this.options = { labels: true, frame: 'circuit', ...options };
    this.canvas = canvas;
    this.layout = layout;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
    this.camera.position.set(2.2, 1.6, 9.5);
    this.controls = new OrbitControls(this.camera, canvas);
    Object.assign(this.controls, { enableDamping: true, autoRotate: true, autoRotateSpeed: 0.7, enablePan: false });

    // Normalise coordinates so the brain fits a radius of about 4 units.
    const ref = layout.dust.length ? layout.dust : layout.kc;
    const lo = [0, 1, 2].map((j) => Math.min(...ref.map((p) => p[j])));
    const hi = [0, 1, 2].map((j) => Math.max(...ref.map((p) => p[j])));
    const center = lo.map((l, j) => (l + hi[j]) / 2);
    const scale = 8 / Math.max(...hi.map((h, j) => h - lo[j]));
    this.v = (p) => new THREE.Vector3((p[0] - center[0]) * scale, (p[1] - center[1]) * scale, (p[2] - center[2]) * scale);
    const flat = (arr) => new Float32Array(arr.flatMap((p) => this.v(p).toArray()));

    this.dust = points(flat(layout.dust), 0.06, 0.45);
    this.kc = points(flat(layout.kc), 0.07, 0.75);
    this.glom = points(flat(layout.glom), 0.2);
    this.active = points(new Float32Array(3 * 4096), 0.16);
    this.active.geometry.setDrawRange(0, 0);
    this.wave1 = lines();
    this.wave2 = lines();
    this.scene.add(this.dust, this.kc, this.glom, this.wave1, this.wave2, this.active);

    this.mbonPos = { approach: this.v(layout.mbon.approach), avoid: this.v(layout.mbon.avoid) };
    this.mbon = {};
    for (const key of ['approach', 'avoid']) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), new THREE.MeshBasicMaterial());
      m.position.copy(this.mbonPos[key]);
      m.scale.setScalar(0.12);
      this.mbon[key] = m;
      this.scene.add(m);
    }
    this.apl = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 12), new THREE.MeshBasicMaterial());
    this.apl.position.copy(this.v(layout.apl));
    this.scene.add(this.apl);
    // Other mushroom-body neurons from the connectome (all MBONs, PAM and PPL1 dopamine neurons).
    this.extra = {};
    for (const [name, pts] of Object.entries(layout.extra || {})) {
      if (!pts.length) continue;
      this.extra[name] = points(flat(pts), 0.11, 0.8);
      this.scene.add(this.extra[name]);
    }
    if (layout.outline) {
      const seg = layout.outline.flatMap(([a, b]) => [...this.v(a).toArray(), ...this.v(b).toArray()]);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(seg), 3));
      this.outline = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ transparent: true, opacity: 0.35 }));
      this.scene.add(this.outline);
    }

    // inputs[i] = glomeruli sampled by Kenyon cell i
    this.inputs = Array.from({ length: layout.kc.length }, () => []);
    layout.edges[0].forEach((kc, e) => this.inputs[kc].push(layout.edges[1][e]));

    // Frame the circuit (Kenyon cells, glomeruli, MBONs); zooming out shows the whole brain.
    const mean = (arr) => arr.reduce((acc, p) => acc.add(this.v(p)), new THREE.Vector3()).multiplyScalar(1 / arr.length);
    this.centroids = { kc: mean(layout.kc), glom: mean(layout.glom) };
    const focus = this.centroids.kc.clone().add(this.centroids.glom).add(this.mbonPos.approach).add(this.mbonPos.avoid).multiplyScalar(0.25);
    if (this.options.frame === 'brain') {
      this.controls.target.set(0, 0, 0);
      this.camera.position.set(0, 1.2, 9.2);
    } else {
      this.controls.target.copy(focus);
      this.camera.position.copy(focus).add(new THREE.Vector3(1.4, 0.9, 4.8));
    }

    this.labels = new THREE.Group();
    this.scene.add(this.labels);
    this.trace = null;
    this.waveT = 0;
    this.applyTheme();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
    this.last = 0;
    this.setActive(true);
  }

  setActive(on) {
    this.renderer.setAnimationLoop(on ? (now) => this.tick(now) : null);
    this.last = 0;
  }

  applyTheme() {
    this.colors = {
      dust: new THREE.Color(css('--dust')),
      kc: new THREE.Color(css('--kc')),
      glom: new THREE.Color(css('--accent')),
      pass: new THREE.Color(css('--pass')),
      block: new THREE.Color(css('--block')),
      apl: new THREE.Color(css('--apl')),
    };
    const fill = (pts, color, k = 1) => {
      const c = pts.geometry.attributes.color;
      for (let i = 0; i < c.count; i++) c.setXYZ(i, color.r * k, color.g * k, color.b * k);
      c.needsUpdate = true;
    };
    fill(this.dust, this.colors.dust);
    fill(this.kc, this.colors.kc, 0.55);
    fill(this.glom, this.colors.glom, 0.35);
    const extraColor = { MBON: new THREE.Color(css('--muted')), PAM: this.colors.pass, PPL1: this.colors.block };
    for (const [name, pts] of Object.entries(this.extra)) fill(pts, extraColor[name] || this.colors.dust, 0.6);
    this.mbon.approach.material.color.copy(this.colors.pass);
    this.mbon.avoid.material.color.copy(this.colors.block);
    this.apl.material.color.copy(this.colors.apl);
    if (this.outline) this.outline.material.color.set(css('--muted'));

    this.labels.clear();
    if (!this.options.labels) {
      if (this.trace) this.show(this.trace, false);
      return;
    }
    const label = (text, pos, color, dx, dy) => {
      const s = textSprite(text, { color, height: 0.2 });
      s.position.copy(pos).add(new THREE.Vector3(dx, dy, 0));
      this.labels.add(s);
    };
    label('접근 MBON', this.mbonPos.approach, css('--pass'), -0.35, 0.3);
    label('회피 MBON', this.mbonPos.avoid, css('--block'), 0.35, -0.3);
    label('APL', this.apl.position, css('--apl'), 0.25, 0.2);
    label('케년세포', this.centroids.kc, css('--kc'), 0, 0.75);
    label('사구체', this.centroids.glom, css('--accent'), 0, -0.55);
    if (this.trace) this.show(this.trace, false);
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.canvas.parentElement;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  show(trace, animate = true) {
    this.trace = trace;
    if (animate) this.waveT = 0;
    const L = this.layout;
    const maxPn = Math.max(...trace.pn, 1e-6);
    const glomColor = this.glom.geometry.attributes.color;
    trace.pn.forEach((v, g) => {
      const k = 0.25 + 0.75 * (v / maxPn);
      glomColor.setXYZ(g, this.colors.glom.r * k, this.colors.glom.g * k, this.colors.glom.b * k);
    });
    glomColor.needsUpdate = true;

    const maxVote = Math.max(...trace.kc_vote.map(Math.abs), 1e-6);
    const pos1 = this.wave1.geometry.attributes.position, col1 = this.wave1.geometry.attributes.color;
    const pos2 = this.wave2.geometry.attributes.position, col2 = this.wave2.geometry.attributes.color;
    const apos = this.active.geometry.attributes.position, acol = this.active.geometry.attributes.color;
    let n1 = 0;
    trace.kc_active.forEach((kc, i) => {
      const vote = trace.kc_vote[i];
      const color = (vote > 0 ? this.colors.block : this.colors.pass).clone().multiplyScalar(0.45 + 0.55 * Math.abs(vote) / maxVote);
      const kp = this.v(L.kc[kc]);
      apos.setXYZ(i, kp.x, kp.y, kp.z);
      acol.setXYZ(i, color.r, color.g, color.b);
      for (const g of this.inputs[kc]) {
        if (trace.pn[g] <= 0 || n1 >= MAX_SEGMENTS) continue;
        const gp = this.v(L.glom[g]);
        const k = 0.15 + 0.6 * trace.pn[g] / maxPn;
        pos1.setXYZ(2 * n1, gp.x, gp.y, gp.z);
        pos1.setXYZ(2 * n1 + 1, kp.x, kp.y, kp.z);
        col1.setXYZ(2 * n1, this.colors.glom.r * k, this.colors.glom.g * k, this.colors.glom.b * k);
        col1.setXYZ(2 * n1 + 1, color.r * k, color.g * k, color.b * k);
        n1++;
      }
      const mp = vote > 0 ? this.mbonPos.avoid : this.mbonPos.approach;
      pos2.setXYZ(2 * i, kp.x, kp.y, kp.z);
      pos2.setXYZ(2 * i + 1, mp.x, mp.y, mp.z);
      col2.setXYZ(2 * i, color.r * 0.5, color.g * 0.5, color.b * 0.5);
      col2.setXYZ(2 * i + 1, color.r, color.g, color.b);
    });
    this.active.geometry.setDrawRange(0, trace.kc_active.length);
    this.wave1.geometry.setDrawRange(0, 2 * n1);
    this.wave2.geometry.setDrawRange(0, 2 * trace.kc_active.length);
    for (const attr of [pos1, col1, pos2, col2, apos, acol]) attr.needsUpdate = true;
    for (const g of [this.wave1, this.wave2, this.active]) g.geometry.computeBoundingSphere();
    this.mbonTarget = { approach: trace.approach, avoid: trace.avoid };
  }

  tick(now) {
    const dt = this.last ? clamp((now - this.last) / 1000, 0, 0.05) : 0;
    this.last = now;
    this.controls.update();
    if (this.trace) {
      this.waveT += dt;
      this.wave1.material.opacity = 0.75 * clamp(this.waveT / 0.6, 0, 1);
      this.wave2.material.opacity = 0.9 * clamp((this.waveT - 0.6) / 0.6, 0, 1);
      this.active.material.opacity = clamp(this.waveT / 0.6, 0, 1);
      for (const key of ['approach', 'avoid']) {
        const target = 0.1 + 0.3 * this.mbonTarget[key] * clamp((this.waveT - 0.8) / 0.5, 0, 1);
        const m = this.mbon[key];
        m.scale.setScalar(m.scale.x + (target - m.scale.x) * Math.min(1, dt * 6));
      }
    }
    this.renderer.render(this.scene, this.camera);
  }
}
