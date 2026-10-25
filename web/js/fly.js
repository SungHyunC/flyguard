// An articulated Drosophila for the sorting stage: head toward -z, six legs with a hip (swing forward,
// lift up) and a knee (bend), a body that can rear up around its hind feet, and a head that tilts.
// Poses are plain numbers so the stage can blend between them.
import * as THREE from 'three';

export const LEGS = ['LF', 'RF', 'LM', 'RM', 'LH', 'RH'];

// [swing forward, lift up, bend] per leg, in radians; missing legs stay at rest.
export const POSES = {
  rest: { pitch: 0, head: 0, legs: {} },
  hold: { pitch: 0.42, head: -0.12, legs: { LF: [0.8, 1.0, -0.5], RF: [0.8, 1.0, -0.5], LM: [0.2, 0.05, 0.1], RM: [0.2, 0.05, 0.1] } },
  push: { pitch: 0.55, head: -0.32, legs: { LF: [0.5, 1.6, -0.9], RF: [0.5, 1.6, -0.9], LM: [0.1, 0, 0], RM: [0.1, 0, 0] } },
  raise: { pitch: 0.62, head: -0.05, legs: { LF: [0.35, 1.8, 0.75], RF: [0.35, 1.8, 0.75], LM: [0.15, 0.1, 0.1], RM: [0.15, 0.1, 0.1] } },
  slam: { pitch: 0.1, head: 0.28, legs: { LF: [1.05, 0.2, -1.1], RF: [1.05, 0.2, -1.1], LM: [-0.05, -0.12, 0.25], RM: [-0.05, -0.12, 0.25] } },
};

export function lerpPose(a, b, k) {
  const legs = {};
  for (const name of LEGS) {
    const pa = a.legs[name] || [0, 0, 0], pb = b.legs[name] || [0, 0, 0];
    legs[name] = pa.map((v, i) => v + (pb[i] - v) * k);
  }
  return { pitch: a.pitch + (b.pitch - a.pitch) * k, head: a.head + (b.head - a.head) * k, legs };
}

function stripes() {
  const c = document.createElement('canvas');
  c.width = 32;
  c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = '#d9d2c3';
  g.fillRect(0, 0, 32, 256);
  g.fillStyle = '#6b5f52';
  for (let i = 0; i < 5; i++) g.fillRect(0, 50 + i * 36, 32, 13);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function segment(a, b, radius, mat) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const m = new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.75, radius, dir.length(), 10), mat);
  m.position.copy(a).addScaledVector(dir, 0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  m.castShadow = true;
  return m;
}

export function makeFly() {
  const root = new THREE.Group();
  const pivot = new THREE.Group();          // rears the body up around the hind feet
  pivot.position.set(0, 0, 0.34);
  root.add(pivot);
  const body = new THREE.Group();
  body.position.set(0, 0, -0.34);
  pivot.add(body);

  const sphere = new THREE.SphereGeometry(1, 40, 28);
  const cuticle = new THREE.MeshPhysicalMaterial({ color: 0xdcd6c8, roughness: 0.42, clearcoat: 0.7, clearcoatRoughness: 0.3 });
  const legMat = new THREE.MeshStandardMaterial({ color: 0x8a90a2, roughness: 0.4, metalness: 0.35, emissive: 0x0d1626 });
  const eyeMat = new THREE.MeshPhysicalMaterial({
    color: 0x9a1212, roughness: 0.22, clearcoat: 1, emissive: 0x4a0606, emissiveIntensity: 0.9,
  });
  const wingMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff, transparent: true, opacity: 0.3, roughness: 0.1, side: THREE.DoubleSide, depthWrite: false,
    iridescence: 1, iridescenceIOR: 1.35, iridescenceThicknessRange: [180, 520],
  });
  const blob = (parent, mat, scale, pos) => {
    const m = new THREE.Mesh(sphere, mat);
    m.scale.set(...scale);
    m.position.set(...pos);
    m.castShadow = true;
    parent.add(m);
    return m;
  };

  blob(body, cuticle, [0.25, 0.21, 0.29], [0, 0.34, 0]);                                   // thorax
  const abdomen = blob(body, new THREE.MeshPhysicalMaterial({ map: stripes(), roughness: 0.5, clearcoat: 0.4 }),
    [0.21, 0.37, 0.18], [0, 0.31, 0.52]);
  abdomen.rotation.x = Math.PI / 2 + 0.1;

  const head = new THREE.Group();                                                          // tilts at the neck
  head.position.set(0, 0.37, -0.27);
  body.add(head);
  blob(head, cuticle, [0.16, 0.15, 0.13], [0, 0.01, -0.09]);
  for (const s of [-1, 1]) blob(head, eyeMat, [0.095, 0.125, 0.105], [s * 0.12, 0.03, -0.11]);
  head.add(segment(new THREE.Vector3(0, -0.08, -0.15), new THREE.Vector3(0, -0.2, -0.2), 0.02, legMat)); // proboscis
  const antennae = [-1, 1].map((s) => {
    const a = new THREE.Group();
    a.position.set(s * 0.04, 0.06, -0.19);
    a.add(segment(new THREE.Vector3(0, 0, 0), new THREE.Vector3(s * 0.03, 0.05, -0.1), 0.018, legMat));
    a.add(segment(new THREE.Vector3(s * 0.03, 0.05, -0.1), new THREE.Vector3(s * 0.07, 0.11, -0.16), 0.008, legMat));
    head.add(a);
    return a;
  });

  const wingShape = new THREE.Shape();
  wingShape.absellipse(0, 0.42, 0.15, 0.45, 0, Math.PI * 2);
  const wingGeo = new THREE.ShapeGeometry(wingShape, 32);
  const wings = [-1, 1].map((s) => {
    const p = new THREE.Group();
    p.position.set(s * 0.05, 0.53, -0.02);
    p.rotation.y = s * 0.36;
    const w = new THREE.Mesh(wingGeo, wingMat);
    w.rotation.x = Math.PI / 2;
    p.add(w);
    body.add(p);
    return p;
  });

  const legs = {};
  [[-0.12, -0.34, 'F'], [0.0, 0.0, 'M'], [0.12, 0.34, 'H']].forEach(([z, footZ, pair]) => {
    for (const s of [-1, 1]) {
      const coxa = new THREE.Group();
      coxa.rotation.order = 'YZX';                     // lift around the body axis, then swing
      coxa.position.set(s * 0.15, 0.27, z);
      const kneeAt = new THREE.Vector3(s * 0.25, 0.1, footZ * 0.45);
      coxa.add(segment(new THREE.Vector3(0, 0, 0), kneeAt, 0.034, legMat));
      const knee = new THREE.Group();
      knee.position.copy(kneeAt);
      const foot = new THREE.Vector3(s * 0.2, -0.36, footZ * 0.55);
      knee.add(segment(new THREE.Vector3(0, 0, 0), foot, 0.026, legMat));
      knee.add(segment(foot, foot.clone().add(new THREE.Vector3(s * 0.07, -0.02, footZ * 0.12)), 0.017, legMat));
      coxa.add(knee);
      body.add(coxa);
      legs[`${s < 0 ? 'L' : 'R'}${pair}`] = { coxa, knee, s };
    }
  });

  return {
    group: root,
    setPose(p) {
      pivot.rotation.x = p.pitch;
      head.rotation.x = p.head;
      for (const name of LEGS) {
        const [swing, lift, bend] = p.legs[name] || [0, 0, 0];
        const { coxa, knee, s } = legs[name];
        coxa.rotation.y = s * swing;
        coxa.rotation.z = s * lift;
        knee.rotation.z = s * bend;
      }
    },
    // Idle life: breathing, wing shiver, antennae; `sniff` (0-1) makes the antennae work hard.
    animate(t, sniff = 0) {
      abdomen.scale.set(0.21, 0.37 + 0.008 * Math.sin(t * 2.2), 0.18 + 0.006 * Math.sin(t * 2.2));
      wings.forEach((w, i) => { w.rotation.z = 0.03 * Math.sin(t * 7 + i) + sniff * 0.05 * Math.sin(t * 40 + i); });
      antennae.forEach((a, i) => {
        a.rotation.x = 0.15 * Math.sin(t * 3 + i) + sniff * 0.35 * Math.sin(t * 23 + i * 1.7);
        a.rotation.z = sniff * 0.25 * Math.sin(t * 17 + i);
      });
    },
  };
}
