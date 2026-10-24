// 2-D circuit panel: antennal-lobe glomeruli -> Kenyon-cell grid -> MBON gauges. The gauges show
// approach + θ/2 against avoid − θ/2, so the bar that is taller is the verdict.
import { clamp, css } from './util.js';

export class Circuit {
  constructor(canvas, layout) {
    this.canvas = canvas;
    this.nKc = layout.kc.length;
    this.nGlom = layout.glom.length;
    this.trace = null;
    this.t0 = 0;
    new ResizeObserver(() => this.draw(1)).observe(canvas.parentElement);
    this.draw(1);
  }

  show(trace) {
    this.trace = trace;
    this.t0 = performance.now();
    const frame = () => {
      const p = (performance.now() - this.t0) / 1500;
      this.draw(Math.min(p, 1));
      if (p < 1) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  draw(progress) {
    const c = this.canvas;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(window.devicePixelRatio, 2);
    c.width = w * dpr;
    c.height = h * dpr;
    const g = c.getContext('2d');
    g.scale(dpr, dpr);
    g.clearRect(0, 0, w, h);
    const col = {
      text: css('--text'), muted: css('--muted'), line: css('--line'), glom: css('--accent'),
      kc: css('--kc'), pass: css('--pass'), block: css('--block'),
    };
    const t = this.trace;
    const stage = (a, b) => clamp((progress - a) / (b - a), 0, 1);
    const top = 30, bottom = h - 12;
    g.font = '600 12px Pretendard, "Malgun Gothic", system-ui, sans-serif';
    g.textBaseline = 'alphabetic';

    // Antennal lobe: one circle per glomerulus.
    const alW = w * 0.24;
    const cols = Math.ceil(Math.sqrt(this.nGlom * 0.6));
    const rows = Math.ceil(this.nGlom / cols);
    const cell = Math.min(alW / cols, (bottom - top) / rows);
    const maxPn = t ? Math.max(...t.pn, 1e-6) : 1;
    g.fillStyle = col.muted;
    g.fillText(`사구체 ${this.nGlom}`, 0, 16);
    for (let i = 0; i < this.nGlom; i++) {
      const x = (i % cols + 0.5) * cell, y = top + (Math.floor(i / cols) + 0.5) * cell;
      const a = t ? (t.pn[i] / maxPn) * stage(0, 0.35) : 0;
      g.beginPath();
      g.arc(x, y, cell * 0.38, 0, Math.PI * 2);
      g.globalAlpha = 0.18 + 0.82 * a;
      g.fillStyle = col.glom;
      g.fill();
    }
    g.globalAlpha = 1;

    // Kenyon cells: a grid, the active 5% lit by their vote.
    const kx = w * 0.3, kw = w * 0.42;
    const kcCols = Math.ceil(Math.sqrt(this.nKc * kw / (bottom - top)));
    const kcRows = Math.ceil(this.nKc / kcCols);
    const kc = Math.min(kw / kcCols, (bottom - top) / kcRows);
    g.fillStyle = col.muted;
    g.fillText(`케년세포 ${this.nKc.toLocaleString()}${t ? ` · 발화 ${t.kc_active.length}` : ''}`, kx, 16);
    g.fillStyle = col.kc;
    g.globalAlpha = 0.22;
    for (let i = 0; i < this.nKc; i++) {
      g.fillRect(kx + (i % kcCols) * kc, top + Math.floor(i / kcCols) * kc, Math.max(1, kc - 1), Math.max(1, kc - 1));
    }
    if (t) {
      const s = stage(0.3, 0.65);
      t.kc_active.forEach((i, j) => {
        g.globalAlpha = s;
        g.fillStyle = t.kc_vote[j] > 0 ? col.block : col.pass;
        const pad = -kc * 0.6;
        g.fillRect(kx + (i % kcCols) * kc + pad / 2, top + Math.floor(i / kcCols) * kc + pad / 2, kc - pad, kc - pad);
      });
    }
    g.globalAlpha = 1;

    // MBON gauges.
    const mx = w * 0.78, mw = w * 0.22;
    g.fillStyle = col.muted;
    g.fillText('MBON', mx, 16);
    if (!t) return;
    const s = stage(0.6, 1);
    const half = t.theta / 2;
    const bars = [['접근', t.approach + half, col.pass], ['회피', t.avoid - half, col.block]];
    const maxV = Math.max(...bars.map((b) => b[1]), 0.05);
    const barW = mw * 0.3;
    bars.forEach(([name, v, color], i) => {
      const x = mx + i * (barW + mw * 0.12);
      const bh = (bottom - top - 18) * clamp(v / maxV, 0, 1) * s;
      g.fillStyle = color;
      g.globalAlpha = 0.9;
      g.fillRect(x, bottom - 18 - bh, barW, bh);
      g.globalAlpha = 1;
      g.fillStyle = col.text;
      g.fillText(name, x, bottom);
    });

    // Arrows between stages.
    g.strokeStyle = col.line;
    g.fillStyle = col.line;
    for (const [x0, x1] of [[alW + 4, kx - 6], [kx + kw + 4, mx - 6]]) {
      const y = (top + bottom) / 2;
      g.beginPath();
      g.moveTo(x0, y);
      g.lineTo(x1, y);
      g.stroke();
      g.beginPath();
      g.moveTo(x1, y);
      g.lineTo(x1 - 6, y - 4);
      g.lineTo(x1 - 6, y + 4);
      g.fill();
    }
  }
}
