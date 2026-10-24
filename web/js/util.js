import * as THREE from 'three';

export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
export const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

// Deterministic randomness: the same prompt always produces the same walk.
export function hash(text) {
  let h = 2166136261;
  for (const ch of text) {
    h ^= ch.codePointAt(0);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function mulberry32(seed) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Soft round dot for THREE.Points.
export const dotTexture = (() => {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.5, 'rgba(255,255,255,0.85)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
})();

function wrap(ctx, text, maxWidth) {
  const lines = [];
  let line = '';
  for (const word of text.split(/\s+/)) {
    const next = line ? `${line} ${word}` : word;
    if (ctx.measureText(next).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines;
}

// Text label that always faces the camera. `height` is the world height of one text line.
export function textSprite(text, { color = '#ffffff', background = null, height = 0.35, maxWidth = 900, maxLines = 3 } = {}) {
  const size = 56;
  const font = `600 ${size}px Pretendard, "Malgun Gothic", "Apple SD Gothic Neo", system-ui, sans-serif`;
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');
  ctx.font = font;
  let lines = wrap(ctx, text, maxWidth);
  if (lines.length > maxLines) lines = [...lines.slice(0, maxLines - 1), `${lines[maxLines - 1]}…`];
  const pad = 22;
  c.width = Math.ceil(Math.max(...lines.map((l) => ctx.measureText(l).width)) + 2 * pad);
  c.height = Math.ceil(lines.length * size * 1.3 + 2 * pad - size * 0.3);
  ctx.font = font;
  if (background) {
    ctx.fillStyle = background;
    ctx.beginPath();
    ctx.roundRect(0, 0, c.width, c.height, 26);
    ctx.fill();
  }
  ctx.fillStyle = color;
  ctx.textBaseline = 'top';
  lines.forEach((l, i) => ctx.fillText(l, pad, pad + i * size * 1.3));
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  const unit = height / (size * 1.3);
  sprite.scale.set(c.width * unit, c.height * unit, 1);
  return sprite;
}
