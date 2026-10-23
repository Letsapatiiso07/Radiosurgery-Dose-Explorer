// Steps through the exported slices and draws each plan's dose-volume histogram.

const $ = (id) => document.getElementById(id);
let patients = [];
let current = null;

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function fmt(v, digits = 1) {
  return v === null || v === undefined ? "not outlined" : Number(v).toFixed(digits);
}

function showFacts(p) {
  const facts = [
    ["Prescription", `${fmt(p.prescription_gy, 0)} Gy, one session`],
    ["Tumour volume", `${fmt(p.tumour_cc, 2)} cc`],
    ["Tumour covered by the prescription", `${fmt(100 * p.coverage)}%`],
    ["Paddick conformity index", fmt(p.paddick_ci, 2)],
    ["Gradient index", fmt(p.gradient_index, 2)],
    ["Shots / beam-on time", `${p.shots} / ${fmt(p.beam_on_min, 0)} min`],
    ["Maximum dose", `${fmt(p.max_gy)} Gy`],
    ["Mean cochlea dose", p.cochlea_mean_gy == null ? "cochlea not outlined" : `${fmt(p.cochlea_mean_gy, 2)} Gy`],
    ["Tumour to cochlea", p.gap_mm == null ? "-" : `${fmt(p.gap_mm)} mm`],
  ];
  $("facts").replaceChildren(...facts.flatMap(([k, v]) => {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    return [dt, dd];
  }));
}

function drawDvh(p) {
  const canvas = $("dvh");
  const ratio = window.devicePixelRatio || 1;
  const { width, height } = canvas.getBoundingClientRect();
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const pad = { l: 40, r: 10, t: 8, b: 26 };
  const dmax = Math.max(...p.dvh.dose);
  const x = (d) => pad.l + (d / dmax) * (width - pad.l - pad.r);
  const y = (s) => pad.t + (1 - s) * (height - pad.t - pad.b);
  ctx.fillStyle = css("--panel");
  ctx.fillRect(0, 0, width, height);
  ctx.font = "12px system-ui, sans-serif";
  ctx.strokeStyle = css("--line");
  ctx.fillStyle = css("--muted");
  for (let s = 0; s <= 1.0001; s += 0.25) {
    ctx.beginPath();
    ctx.moveTo(pad.l, y(s));
    ctx.lineTo(width - pad.r, y(s));
    ctx.stroke();
    ctx.fillText(`${Math.round(s * 100)}%`, 4, y(s) + 4);
  }
  for (let d = 0; d <= dmax; d += 5) ctx.fillText(`${d}`, x(d) - 4, height - 8);
  ctx.fillText("Gy", width - 22, height - 8);
  const line = (vals, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    vals.forEach((s, i) => (i ? ctx.lineTo(x(p.dvh.dose[i]), y(s)) : ctx.moveTo(x(p.dvh.dose[i]), y(s))));
    ctx.stroke();
  };
  line(p.dvh.target, "#ef4444");
  if (p.dvh.cochlea) line(p.dvh.cochlea, "#0891b2");
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = "#ca8a04";
  ctx.beginPath();
  ctx.moveTo(x(p.prescription_gy), pad.t);
  ctx.lineTo(x(p.prescription_gy), height - pad.b);
  ctx.stroke();
  ctx.setLineDash([]);
}

function showSlice(i) {
  const s = current.slices[i];
  $("slice").src = s.file;
  $("slice-label").textContent = `Slice ${i + 1} of ${current.slices.length}, ${s.z_mm} mm`;
  $("slider").value = i;
}

function choose(i) {
  current = patients[i];
  $("slider").max = current.slices.length - 1;
  showFacts(current);
  drawDvh(current);
  showSlice(Math.floor(current.slices.length / 2));
}

$("patient").addEventListener("change", (e) => choose(Number(e.target.value)));
$("slider").addEventListener("input", (e) => showSlice(Number(e.target.value)));
window.addEventListener("resize", () => current && drawDvh(current));

fetch("data.json")
  .then((r) => r.json())
  .then((data) => {
    patients = data;
    data.forEach((p, i) => {
      const o = document.createElement("option");
      o.value = i;
      o.textContent = `${p.patient} (${p.tumour_cc.toFixed(2)} cc)`;
      $("patient").append(o);
    });
    choose(0);
  });
