document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("query_image");
  const preview = document.getElementById("preview");

  if (fileInput && preview) {
    fileInput.addEventListener("change", () => {
      const [file] = fileInput.files;
      if (!file) {
        preview.style.display = "none";
        preview.src = "";
        return;
      }
      const url = URL.createObjectURL(file);
      preview.src = url;
      preview.style.display = "block";
    });
  }

  const body = document.body;
  const fxFab = document.getElementById("fx-fab");
  const fxPanel = document.getElementById("fx-panel");
  const fxToggle = document.getElementById("fx-toggle");
  const fxSelect = document.getElementById("fx-select");
  const fxCanvas = document.getElementById("fx-canvas");
  const ctx = fxCanvas ? fxCanvas.getContext("2d") : null;

  if (fxFab && fxPanel) {
    fxFab.addEventListener("click", () => {
      fxPanel.classList.toggle("open");
    });
  }

  let storedEffect = "aurora";
  let storedAnim = "on";
  try {
    storedEffect = localStorage.getItem("fx-effect") || "aurora";
    storedAnim = localStorage.getItem("fx-anim") || "on";
  } catch (err) {
    storedEffect = "aurora";
    storedAnim = "on";
  }

  if (fxSelect) {
    fxSelect.value = storedEffect;
  }

  if (fxToggle) {
    fxToggle.checked = storedAnim !== "off";
  }

  let rafId = null;
  let pointerX = 0.5;
  let pointerY = 0.5;
  let mouseX = null;
  let mouseY = null;
  let currentEffect = null;
  let entities = [];
  let boidTarget = 100;
  let boidNextSpawn = 0;

  const resizeCanvas = () => {
    if (!fxCanvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    fxCanvas.width = Math.floor(window.innerWidth * dpr);
    fxCanvas.height = Math.floor(window.innerHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };

  const initParticles = () => {
    const count = Math.max(40, Math.floor(window.innerWidth / 20));
    entities = Array.from({ length: count }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.6,
      vy: (Math.random() - 0.5) * 0.6,
      r: 1.5 + Math.random() * 2.2,
      hue: 180 + Math.random() * 120,
    }));
  };

  const spawnBoid = () => ({
    id: entities.length,
    x: Math.random() * window.innerWidth,
    y: Math.random() * window.innerHeight,
    vx: (Math.random() - 0.5) * 1.4,
    vy: (Math.random() - 0.5) * 1.4,
    density: 0,
    trail: [],
  });

  const initBoids = () => {
    entities = [];
    boidNextSpawn = performance.now() + 500;
    entities.push(spawnBoid());
  };

  const initSakura = () => {
    const count = 30;
    entities = Array.from({ length: count }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: 0.4 + Math.random() * 0.6,
      vy: 0.8 + Math.random() * 1.4,
      size: 6 + Math.random() * 6,
      spin: Math.random() * Math.PI * 2,
    }));
  };

  const drawParticles = () => {
    if (!ctx) return;
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    const driftX = (pointerX - 0.5) * 0.8;
    const driftY = (pointerY - 0.5) * 0.8;
    for (const p of entities) {
      p.x += p.vx + driftX;
      p.y += p.vy + driftY;
      if (p.x < -20) p.x = window.innerWidth + 20;
      if (p.x > window.innerWidth + 20) p.x = -20;
      if (p.y < -20) p.y = window.innerHeight + 20;
      if (p.y > window.innerHeight + 20) p.y = -20;
      ctx.beginPath();
      ctx.fillStyle = `hsla(${p.hue}, 80%, 70%, 0.5)`;
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const drawBoids = () => {
    if (!ctx) return;
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    const neighborDist = 100;
    const alignDist = 60;
    const avoidDist = 48;
    const now = performance.now();
    const t = now * 0.001;
    if (entities.length < boidTarget && now >= boidNextSpawn) {
      entities.push(spawnBoid());
      boidNextSpawn = now + 500;
    }
    for (const b of entities) {
      let alignX = 0;
      let alignY = 0;
      let cohX = 0;
      let cohY = 0;
      let sepX = 0;
      let sepY = 0;
      let count = 0;

      const margin = 80;
      let boundX = 0;
      let boundY = 0;
      if (b.x < margin) boundX = (margin - b.x) / margin;
      if (b.x > window.innerWidth - margin) boundX = -((b.x - (window.innerWidth - margin)) / margin);
      if (b.y < margin) boundY = (margin - b.y) / margin;
      if (b.y > window.innerHeight - margin) boundY = -((b.y - (window.innerHeight - margin)) / margin);

      for (const other of entities) {
        if (b === other) continue;
        const dx = other.x - b.x;
        const dy = other.y - b.y;
        const dist = Math.hypot(dx, dy);
        if (dist < neighborDist) {
          const weight = 1 - dist / neighborDist;
          if (dist < alignDist) {
            const alignWeight = 1 - dist / alignDist;
            alignX += other.vx * alignWeight;
            alignY += other.vy * alignWeight;
          }
          cohX += other.x * weight;
          cohY += other.y * weight;
          count += weight;
          if (dist < avoidDist) {
            const repel = (avoidDist - dist) / avoidDist;
            sepX -= (dx / (dist + 0.01)) * repel;
            sepY -= (dy / (dist + 0.01)) * repel;
          }
        }
      }

      if (count > 0) {
        alignX /= count;
        alignY /= count;
        cohX = cohX / count - b.x;
        cohY = cohY / count - b.y;
      }

      if (mouseX !== null && mouseY !== null) {
        const mdx = b.x - mouseX;
        const mdy = b.y - mouseY;
        const mdist = Math.hypot(mdx, mdy);
        const mouseRadius = 140;
        if (mdist < mouseRadius) {
          const repel = (mouseRadius - mdist) / mouseRadius;
          sepX += (mdx / (mdist + 0.01)) * repel * 2.4;
          sepY += (mdy / (mdist + 0.01)) * repel * 2.4;
        }
      }
      b.density = count;

      const density = count / 14;
      const scatter = Math.max(0, density - 1);
      const jitter = (Math.sin(t * 1.6 + b.id * 0.9) + 1) * 0.5;
      const randAngle = t * 0.7 + b.id * 0.35;
      const randX = Math.cos(randAngle) * jitter;
      const randY = Math.sin(randAngle) * jitter;

      b.vx += (alignX - b.vx) * 0.01 + cohX * 0.0008 + sepX * (0.065 + scatter * 0.03);
      b.vy += (alignY - b.vy) * 0.01 + cohY * 0.0008 + sepY * (0.065 + scatter * 0.03);
      b.vx += randX * (0.25 + scatter * 0.2);
      b.vy += randY * (0.25 + scatter * 0.2);
      b.vx += boundX * 1.2;
      b.vy += boundY * 1.2;

      if (count > 60) {
        const centerX = cohX / count;
        const centerY = cohY / count;
        const awayX = b.x - centerX;
        const awayY = b.y - centerY;
        const awayLen = Math.hypot(awayX, awayY) || 1;
        const crowd = Math.min(1, (count - 14) / 10);
        b.vx += (awayX / awayLen) * crowd * 0.9;
        b.vy += (awayY / awayLen) * crowd * 0.9;
      }

      const speed = Math.hypot(b.vx, b.vy) || 1;
      const maxSpeed = 2.2;
      if (speed > maxSpeed) {
        b.vx = (b.vx / speed) * maxSpeed;
        b.vy = (b.vy / speed) * maxSpeed;
      }

      b.x += b.vx;
      b.y += b.vy;

      if (b.x < 4) b.x = 4;
      if (b.x > window.innerWidth - 4) b.x = window.innerWidth - 4;
      if (b.y < 4) b.y = 4;
      if (b.y > window.innerHeight - 4) b.y = window.innerHeight - 4;

      if (!b.trail) b.trail = [];
      b.trail.push({ x: b.x, y: b.y });
      const trailLen = Math.round(4 + Math.min(22, speed * 10));
      if (b.trail.length > trailLen) {
        b.trail.splice(0, b.trail.length - trailLen);
      }
    }

    const light = { r: 239, g: 246, b: 255 };
    const dark = { r: 15, g: 23, b: 120 };
    const clamp01 = (value) => Math.max(0, Math.min(1, value));
    for (const b of entities) {
      const angle = Math.atan2(b.vy, b.vx);
      const length = 8;
      const wing = 4;
      const densityNorm = clamp01((b.density - 1) / 5);
      const r = Math.round(light.r + (dark.r - light.r) * densityNorm);
      const g = Math.round(light.g + (dark.g - light.g) * densityNorm);
      const blue = Math.round(light.b + (dark.b - light.b) * densityNorm);
      const trail = b.trail || [];
      if (trail.length > 1) {
        ctx.beginPath();
        ctx.moveTo(trail[0].x, trail[0].y);
        for (let i = 1; i < trail.length; i += 1) {
          ctx.lineTo(trail[i].x, trail[i].y);
        }
        ctx.strokeStyle = `rgba(${r}, ${g}, ${blue}, ${0.18 + densityNorm * 0.25})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      ctx.save();
      ctx.translate(b.x, b.y);
      ctx.rotate(angle);
      ctx.beginPath();
      ctx.moveTo(length, 0);
      ctx.lineTo(-length * 0.6, wing);
      ctx.lineTo(-length * 0.2, 0);
      ctx.lineTo(-length * 0.6, -wing);
      ctx.closePath();
      ctx.fillStyle = `rgba(${r}, ${g}, ${blue}, 0.75)`;
      ctx.fill();
      ctx.restore();
    }
  };

  const drawSakura = () => {
    if (!ctx) return;
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    for (const s of entities) {
      s.x += s.vx + (pointerX - 0.5) * 0.5;
      s.y += s.vy;
      s.spin += 0.02;
      if (s.y > window.innerHeight + 20) {
        s.y = -20;
        s.x = Math.random() * window.innerWidth;
      }
      if (s.x > window.innerWidth + 20) s.x = -20;
      ctx.save();
      ctx.translate(s.x, s.y);
      ctx.rotate(s.spin);
      ctx.fillStyle = "rgba(251, 113, 133, 0.7)";
      ctx.beginPath();
      ctx.moveTo(0, -s.size);
      ctx.bezierCurveTo(s.size, -s.size, s.size, s.size, 0, s.size * 0.6);
      ctx.bezierCurveTo(-s.size, s.size, -s.size, -s.size, 0, -s.size);
      ctx.fill();
      ctx.restore();
    }
  };

  const stopAnimation = () => {
    if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
  };

  const startEffect = (effect) => {
    if (!fxCanvas || !ctx) return;
    stopAnimation();
    resizeCanvas();
    if (effect === "particles") initParticles();
    if (effect === "boids") initBoids();
    if (effect === "sakura") initSakura();

    const loop = () => {
      if (effect === "particles") drawParticles();
      if (effect === "boids") drawBoids();
      if (effect === "sakura") drawSakura();
      rafId = requestAnimationFrame(loop);
    };
    loop();
  };

  const applyFxState = () => {
    const effect = fxSelect ? fxSelect.value : storedEffect;
    const animOn = fxToggle ? fxToggle.checked : storedAnim !== "off";
    body.dataset.effect = effect;
    body.classList.toggle("anim-off", !animOn);

    if (fxCanvas) {
      const canvasOn = ["particles", "boids", "sakura"].includes(effect);
      fxCanvas.classList.toggle("on", canvasOn && animOn);
      if (canvasOn && animOn) {
        if (currentEffect !== effect) {
          startEffect(effect);
          currentEffect = effect;
        }
      } else {
        stopAnimation();
        if (ctx) {
          ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
        }
      }
    }

    try {
      localStorage.setItem("fx-effect", effect);
      localStorage.setItem("fx-anim", animOn ? "on" : "off");
    } catch (err) {
      return;
    }
  };

  if (fxSelect) {
    fxSelect.addEventListener("change", applyFxState);
  }

  if (fxToggle) {
    fxToggle.addEventListener("change", applyFxState);
  }

  applyFxState();

  const updatePointer = (event) => {
    pointerX = event.clientX / window.innerWidth;
    pointerY = event.clientY / window.innerHeight;
    mouseX = event.clientX;
    mouseY = event.clientY;
    body.style.setProperty("--mx", pointerX.toFixed(3));
    body.style.setProperty("--my", pointerY.toFixed(3));
  };

  window.addEventListener("pointermove", updatePointer);
  window.addEventListener("pointerleave", () => {
    pointerX = 0.5;
    pointerY = 0.5;
    mouseX = null;
    mouseY = null;
    body.style.setProperty("--mx", "0.5");
    body.style.setProperty("--my", "0.5");
  });
  window.addEventListener("resize", () => {
    resizeCanvas();
  });
});
