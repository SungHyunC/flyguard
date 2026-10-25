// Spike raster for the HUD's "neural replay": 96 rows stand for Kenyon cells (KC index mod 96). Rows of
// the KCs that fired for the current prompt light up for a moment; the rest show sparse background.
export class Raster {
  constructor(canvas, rows = 96) {
    this.canvas = canvas;
    this.rows = rows;
    this.boost = new Float32Array(rows);
    this.acc = 0;
  }

  burst(kcActive) {
    for (const kc of kcActive) this.boost[kc % this.rows] = 1;
  }

  tick(dt) {
    const c = this.canvas;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    this.acc += dt;
    if (this.acc < 1 / 45) return;           // about 45 columns per second
    const step = 2, rowH = h / this.rows;
    const g = c.getContext('2d');
    g.drawImage(c, -step, 0);
    g.clearRect(w - step, 0, step, h);
    for (let r = 0; r < this.rows; r++) {
      const p = 0.012 + 0.7 * this.boost[r];
      if (Math.random() < p) {
        g.fillStyle = this.boost[r] > 0.15 ? '#b8fbff' : '#2f6f8f';
        g.fillRect(w - step, r * rowH, step, Math.max(1, rowH - 0.5));
      }
      this.boost[r] *= Math.exp(-this.acc / 0.5);
    }
    this.acc = 0;
  }
}
