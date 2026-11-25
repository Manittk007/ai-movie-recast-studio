"use strict";
const $ = (id) => document.getElementById(id);
let current = null; // the open project

function show(screen) {
  $("home").hidden = screen !== "home";
  $("video").hidden = screen !== "video";
  window.scrollTo(0, 0);
}

function fmtTime(s) {
  const m = Math.floor(s / 60);
  return m + ":" + String(Math.floor(s % 60)).padStart(2, "0");
}

async function api(url, opts) {
  const r = await fetch(url, opts);
  let data = null;
  try { data = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) throw new Error((data && data.error) || "Request failed (" + r.status + ")");
  return data;
}

// ---------- home ----------
async function loadProjects() {
  const ul = $("projects");
  ul.textContent = "";
  let items = [];
  try { items = await api("/api/projects"); }
  catch (e) { $("homeMsg").textContent = e.message; return; }
  if (!items.length) {
    const li = document.createElement("li");
    li.textContent = "No projects yet.";
    ul.appendChild(li);
    return;
  }
  items.forEach((p) => {
    const li = document.createElement("li");
    const t = document.createElement("div");
    t.className = "t";
    const b = document.createElement("b");
    b.textContent = p.name;
    const sub = document.createElement("span");
    sub.className = "info";
    sub.textContent = fmtTime(p.info.duration) + " · " + p.info.width + "x" + p.info.height;
    t.append(b, sub);
    const open = document.createElement("button");
    open.className = "small";
    open.textContent = "Open";
    open.onclick = () => openProject(p.id);
    const del = document.createElement("button");
    del.className = "small danger";
    del.textContent = "Delete";
    del.onclick = async () => {
      if (!confirm("Delete this project and its files from the phone?")) return;
      try { await api("/api/projects/" + p.id, { method: "DELETE" }); }
      catch (e) { $("homeMsg").textContent = e.message; }
      loadProjects();
    };
    li.append(t, open, del);
    ul.appendChild(li);
  });
}

$("startBtn").onclick = () => {
  const file = $("videoFile").files[0];
  const msg = $("homeMsg");
  msg.textContent = "";
  if (!file) { msg.textContent = "Choose a video first."; return; }
  if (!$("consentVideo").checked) { msg.textContent = "Please confirm you have the right to edit this footage."; return; }
  const fd = new FormData();
  fd.append("video", file);
  fd.append("name", $("name").value.trim());
  const xhr = new XMLHttpRequest();
  const prog = $("upProg");
  prog.hidden = false;
  prog.value = 0;
  $("startBtn").disabled = true;
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) prog.value = (e.loaded / e.total) * 100; };
  xhr.onload = () => {
    $("startBtn").disabled = false;
    prog.hidden = true;
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch (e) { /* ignore */ }
    if (xhr.status !== 200) { msg.textContent = data.error || "Upload failed."; return; }
    $("videoFile").value = "";
    $("name").value = "";
    $("consentVideo").checked = false;
    openProject(data.id);
  };
  xhr.onerror = () => {
    $("startBtn").disabled = false;
    prog.hidden = true;
    msg.textContent = "Upload failed. Is the server still running in Termux?";
  };
  xhr.open("POST", "/api/projects");
  xhr.send(fd);
};

// ---------- video screen ----------
async function openProject(id) {
  try { current = await api("/api/projects/" + id); }
  catch (e) { $("homeMsg").textContent = e.message; return; }
  const i = current.info;
  $("vTitle").textContent = current.name;
  $("player").src = "/api/projects/" + id + "/video";
  $("vInfo").textContent = "Duration " + fmtTime(i.duration) + " · " + i.width + "x" + i.height + " · " + i.fps + " fps";
  const slider = $("frameSlider");
  slider.max = Math.max(0, i.duration - 0.1);
  slider.value = 0;
  loadFrame(0);
  const saved = JSON.parse(localStorage.getItem("settings:" + id) || "{}");
  $("resSel").value = saved.res || "480";
  $("fpsSel").value = saved.fps || "0";
  updateEstimate();
  show("video");
}

let frameTimer = null;
function loadFrame(t) {
  $("frameTime").textContent = "Time " + t.toFixed(1) + " s";
  clearTimeout(frameTimer);
  frameTimer = setTimeout(() => {
    $("frameImg").src = "/api/projects/" + current.id + "/frame?w=480&t=" + t + "&_=" + Date.now();
  }, 250);
}
$("frameSlider").oninput = (e) => loadFrame(parseFloat(e.target.value));

function updateEstimate() {
  if (!current) return;
  const fps = parseInt($("fpsSel").value, 10) || current.info.fps;
  const frames = Math.round(current.info.duration * fps);
  $("estimate").textContent = "About " + frames + " frames would be processed at this setting.";
  localStorage.setItem("settings:" + current.id, JSON.stringify({ res: $("resSel").value, fps: $("fpsSel").value }));
}
$("resSel").onchange = updateEstimate;
$("fpsSel").onchange = updateEstimate;

$("backBtn").onclick = () => {
  $("player").pause();
  $("player").removeAttribute("src");
  $("player").load();
  current = null;
  show("home");
  loadProjects();
};

loadProjects();
