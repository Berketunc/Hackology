'use strict';

const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = number => Number(number).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2});
const tierLabel = tier => ({high:'≥6 h',intermediate:'2–6 h',low:'<2 h'}[tier]);
const badge = tier => `<span class="tag tag-${tier}">${escapeHTML(tierLabel(tier))}</span>`;
const pairKey = row => `${row.peptide}|${row.allele}`;
const range = model => `${fmt(model.low)}–${fmt(model.high)}`;
const STORAGE_KEY = 'peptide-hla-workspace-runs-v1';
const state = {bootstrap:null, scan:null, draftAlleles:[], rankingAllele:'', tier:'all', shown:12,
  selected:null, selectedResult:null, shortlist:[], batch:null, comparison:null, single:null,
  runs:[], saving:'scan', scanRequest:0, selectedRequest:0};

function notice(message, error=false) {
  $('notice').hidden = !message;
  $('notice').classList.toggle('error', error);
  $('notice').replaceChildren();
  if (!message) return;
  const text = document.createElement('span'); text.textContent = message;
  const dismiss = document.createElement('button'); dismiss.textContent = '×'; dismiss.setAttribute('aria-label','Dismiss message');
  dismiss.onclick = () => notice(''); $('notice').append(text, dismiss);
}

async function api(path, data) {
  const response = await fetch(`/api/${path}`, data === undefined ? {} : {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  let result;
  try { result = await response.json(); } catch { throw new Error('The server returned an unreadable response. Please retry.'); }
  if (!response.ok) {
    const detail = Array.isArray(result.detail) ? result.detail.map(x=>x.msg).join('; ') : result.detail;
    throw new Error(detail || 'This request could not be completed.');
  }
  return result;
}

async function withBusy(buttonId, label, action) {
  const button = $(buttonId), original = button.innerHTML;
  button.disabled = true; button.classList.add('busy'); button.textContent = label;
  button.setAttribute('aria-busy','true');
  try { await action(); } catch (error) { notice(error.message || 'Could not reach the model server.', true); }
  finally { button.disabled = false; button.classList.remove('busy'); button.innerHTML = original; button.removeAttribute('aria-busy'); }
}

function marks(root=document) {
  root.querySelectorAll('.blueprint, .btn-primary').forEach(element => {
    if (element.querySelector(':scope > .corner')) return;
    ['tl','tr','bl','br'].forEach(corner => {
      const mark=document.createElement('i'); mark.className=`corner ${corner}`; mark.setAttribute('aria-hidden','true'); element.append(mark);
    });
  });
}

function showPage() {
  const valid=['scan','single','batch','compare','runs','methods'];
  const view=valid.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'scan';
  document.querySelectorAll('.page').forEach(page=>page.hidden=page.id!==`page-${view}`);
  document.querySelectorAll('#navigation a').forEach(link=>{
    if(link.hash===`#${view}`) link.setAttribute('aria-current','page'); else link.removeAttribute('aria-current');
  });
  if(view==='runs') renderRuns();
  if(view==='compare') renderComparison();
}
window.addEventListener('hashchange',showPage);

function options(select, items, selected) {
  select.innerHTML=items.map(item=>`<option value="${escapeHTML(item.value)}">${escapeHTML(item.label)}</option>`).join('');
  if(items.some(item=>item.value===selected)) select.value=selected;
}
function updateSequenceCount() {
  const lines=$('protein').value.trim().split(/\r?\n/).filter(line=>!line.startsWith('>'));
  const n=lines.join('').replace(/\s/g,'').length;
  $('sequence-count').textContent=`${n.toLocaleString()} residues · ${Math.max(0,n-8).toLocaleString()} windows`;
}
function renderAlleles() {
  $('allele-choices').innerHTML=state.draftAlleles.map(a=>`<button class="allele-chip" type="button" data-remove-allele="${escapeHTML(a)}" aria-label="Remove ${escapeHTML(a)}">${escapeHTML(a)} <span aria-hidden="true">×</span></button>`).join('');
  options($('add-allele'),state.bootstrap.alleles.filter(a=>!state.draftAlleles.includes(a.name)).map(a=>({value:a.name,label:a.name})), $('add-allele').value);
  $('add-allele-button').disabled=state.draftAlleles.length>=6;
}

function rankedRows() {
  return (state.scan?.rows || []).filter(r=>r.allele===state.rankingAllele && (state.tier==='all'||r.models[0].tier===state.tier))
    .sort((a,b)=>b.models[0].hours-a.models[0].hours || a.position-b.position);
}
function isShortlisted(row) { return state.shortlist.some(s=>pairKey(s)===pairKey(row)); }
function shortlistButton(row) {
  const selected=isShortlisted(row);
  return `<button type="button" class="btn btn-ghost" data-shortlist="${escapeHTML(pairKey(row))}" aria-pressed="${selected}" aria-label="${selected?'Remove':'Add'} ${escapeHTML(row.peptide)} ${escapeHTML(row.allele)} ${selected?'from':'to'} shortlist">${selected?'Remove':'Add'}</button>`;
}
function renderRanked() {
  const rows=rankedRows();
  $('shown-count').textContent=state.scan ? `${Math.min(state.shown,rows.length)} of ${rows.length} shown` : '';
  $('ranked-rows').innerHTML=rows.slice(0,state.shown).map((r,i)=>{
    const m=r.models[0], selected=state.selected && pairKey(r)===pairKey(state.selected) && r.position===state.selected.position;
    return `<tr class="${selected?'selected':''}"><td>${i+1}</td><td class="number">${r.position}</td><td><button type="button" class="peptide-button" data-select="${escapeHTML(pairKey(r))}" data-position="${r.position}" aria-label="Inspect ${r.peptide} at position ${r.position}">${r.peptide}</button></td><td class="number">${fmt(m.hours)}</td><td class="range">${range(m)}</td><td>${badge(m.tier)}</td><td>${shortlistButton(r)}</td></tr>`;
  }).join('') || `<tr><td colspan="7" class="empty">${state.scan?'No windows match this tier. Try another filter.':'Run a scan to explore candidate windows.'}</td></tr>`;
  $('show-more').hidden=rows.length<=state.shown;
  $('export-scan').disabled=!rows.length;
  document.querySelectorAll('[data-tier]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.tier===state.tier)));
}

function renderHeatmap() {
  if(!state.scan) return;
  const cells=state.scan.alleles.map(a=>{
    const rows=state.scan.rows.filter(r=>r.allele===a);
    return `<div class="heat-row"><span class="heat-label">${escapeHTML(a)}</span>${rows.map(r=>{
      const opacity=Math.max(.06,Math.min(1,Math.log1p(r.models[0].hours)/Math.log(13)));
      const selected=state.selected && pairKey(r)===pairKey(state.selected) && r.position===state.selected.position;
      const label=`Position ${r.position}: ${r.peptide}, ${r.allele}, ${fmt(r.models[0].hours)} hours`;
      return `<button type="button" class="heat-cell ${selected?'selected':''}" style="background:color-mix(in srgb,var(--color-accent) ${(opacity*100).toFixed(2)}%,var(--color-bg))" data-select="${escapeHTML(pairKey(r))}" data-position="${r.position}" title="${escapeHTML(label)}" aria-label="${escapeHTML(label)}" aria-pressed="${!!selected}"></button>`;
    }).join('')}</div>`;
  }).join('');
  $('heatmap').innerHTML=cells+`<div class="heat-row"><span class="heat-label">Position</span>${Array.from({length:state.scan.window_count},(_,i)=>`<span class="heat-position">${i+1}</span>`).join('')}</div>`;
}

function renderScan() {
  if(!state.scan) return;
  options($('ranking-allele'),state.scan.alleles.map(a=>({value:a,label:a})),state.rankingAllele);
  state.rankingAllele=$('ranking-allele').value;
  $('scan-summary').hidden=false;
  const name=state.bootstrap.models.find(m=>m.id===state.scan.arm)?.name || state.scan.arm;
  $('scan-summary').innerHTML=`<span><strong>${state.scan.window_count}</strong> windows</span><span><strong>${state.scan.alleles.length}</strong> alleles</span><span><strong>${state.scan.rows.length.toLocaleString()}</strong> scored pairs</span><span>Ranked by <strong>${escapeHTML(name)}</strong></span>`;
  $('save-scan').disabled=false; renderRanked(); renderHeatmap();
}

async function runScan() {
  const request=++state.scanRequest;
  await withBusy('scan-submit','Scoring windows…',async()=>{
    notice('Scoring every window with the selected sequence model…');
    const result=await api('scan',{sequence:$('protein').value,alleles:state.draftAlleles,arm:$('scan-model').value});
    if(request!==state.scanRequest) return;
    state.scan=result; state.shown=12;
    if(!result.alleles.includes(state.rankingAllele)) state.rankingAllele=result.alleles[0];
    state.selected=null; state.selectedResult=null; renderScan(); notice('');
    const first=rankedRows()[0] || result.rows[0];
    if(first) await selectPair(first,false);
  });
  marks();
}

function predictionExplanation(result) {
  return result.prediction_source==='held_out_peptide_fold'
    ? 'Central predictions use the fit that held this peptide out. Other fits in the range can include this peptide.'
    : 'This peptide is new to the dataset. Central predictions average five fitted models on the log scale.';
}

function renderSelected() {
  const row=state.selected; if(!row) return;
  $('selected-title').innerHTML=`${escapeHTML(row.allele)}<span class="mono">${row.peptide}</span>`;
  const result=state.selectedResult || row;
  const available=new Map(result.models.map(m=>[m.id,m]));
  $('selected-content').innerHTML=`<div class="table-scroll"><table class="table selected-table"><thead><tr><th>Model</th><th>Half-life (h)</th><th>Range</th></tr></thead><tbody>${state.bootstrap.models.map(arm=>{
    const m=available.get(arm.id);return `<tr><td>${escapeHTML(arm.name)}</td><td class="number">${m?fmt(m.hours):'—'}</td><td class="range">${m?range(m):'Not run'}</td></tr>`;
  }).join('')}</tbody></table></div><p class="selected-note">${escapeHTML(predictionExplanation(result))}</p>`;
  if(result.context) {
    const c=result.context;
    $('selected-content').insertAdjacentHTML('beforeend',`<p class="selected-note"><strong>${c.measurements.toLocaleString()}</strong> dataset measurements · ${(c.zero_fraction*100).toFixed(1)}% reported zeros.</p>`);
  }
  $('compare-selected').disabled=false;
}

async function selectPair(row, allModels=false) {
  state.selected=row; state.selectedResult=null;
  const request=++state.selectedRequest;
  renderSelected(); renderRanked(); renderHeatmap();
  try {
    const result=await api('predict',{peptide:row.peptide,allele:row.allele,include_plm:allModels});
    if(request!==state.selectedRequest) return;
    state.selectedResult=result; renderSelected();
  } catch(error) { if(request===state.selectedRequest) notice(error.message,true); }
}

function allAvailableRows() {
  return [...(state.scan?.rows||[]),...(state.batch?.rows||[]),...(state.single?[state.single]:[]),...state.shortlist];
}
function toggleShortlist(row) {
  const index=state.shortlist.findIndex(r=>pairKey(r)===pairKey(row));
  if(index>=0) state.shortlist.splice(index,1);
  else {
    if(state.shortlist.length>=30) { notice('The shortlist holds up to 30 pairs. Export it or remove a candidate before adding more.',true); return; }
    state.shortlist.push(structuredClone(row));
  }
  state.comparison=null; renderShortlist(); renderRanked(); renderBatch(); renderComparison();
  if(state.single) renderSingle();
}
function renderShortlist() {
  $('short-count').textContent=`· ${state.shortlist.length}`;
  $('shortlist').innerHTML=state.shortlist.map(r=>`<div class="short-item"><div><span class="mono">${r.peptide}</span><small>${escapeHTML(r.allele)}</small></div><span class="number">${fmt(r.models[0].hours)} h</span><button type="button" class="btn btn-ghost" data-shortlist="${escapeHTML(pairKey(r))}" aria-label="Remove ${r.peptide} ${escapeHTML(r.allele)} from shortlist">×</button></div>`).join('') || '<p class="muted small">Use “Add” on a row to keep candidates here.</p>';
  ['view-comparison','export-shortlist','run-comparison'].forEach(id=>$(id).disabled=!state.shortlist.length);
}

function renderSingle() {
  const r=state.single; if(!r) return;
  const c=r.context,max=Math.max(12,...r.models.map(m=>m.hours));
  $('single-result').innerHTML=`<div class="single-output"><div class="section-heading"><h2>${escapeHTML(r.allele)} · <span class="mono">${r.peptide}</span></h2>${shortlistButton(r)}</div>
    <div class="single-context"><div class="stat"><strong>${c.measurements.toLocaleString()}</strong><span>recorded measurements in this dataset</span></div><div class="stat"><strong>${(c.zero_fraction*100).toFixed(1)}%</strong><span>reported zeros for this allele</span></div><div class="stat"><strong>${c.recorded_hours===null?'Not measured':`${fmt(c.recorded_hours)} h`}</strong><span>${c.recorded_hours===null?'pair absent from organiser dataset':'recorded half-life · retrospective comparison'}</span></div></div>
    <div class="blueprint table-frame"><div class="table-scroll"><table class="table"><thead><tr><th>Model</th><th>Half-life (h)</th><th>Scale</th><th>Five-fit range (h)</th><th>Tier</th><th>Allele training rows</th></tr></thead><tbody>${r.models.map(m=>`<tr><td>${escapeHTML(m.name)}</td><td class="number">${fmt(m.hours)}</td><td><div class="bar-track"><div class="bar-fill" style="width:${(m.hours/max*100).toFixed(2)}%"></div></div></td><td class="range">${range(m)}</td><td>${badge(m.tier)}</td><td class="number">${m.training_rows}</td></tr>`).join('')}</tbody></table></div></div>
    <p class="small muted">${escapeHTML(predictionExplanation(r))} Five-fit ranges are not calibrated confidence intervals.</p>
    <p class="small muted">Nearest better-measured neighbour: ${c.nearest?`${escapeHTML(c.nearest.allele)} · ${c.nearest.measurements} measurements · ${(c.nearest.identity*100).toFixed(1)}% contact-residue identity`:'none; this allele has the largest measurement count.'}</p>
    <p class="exposure"><strong>Training exposure:</strong> ${escapeHTML(c.exposure)}</p>
    ${c.zero_fraction>=.5?'<p class="small muted">At least half of this allele’s measurements are zero; interpret ranking scores alongside tier AUC.</p>':''}</div>`;
  marks($('single-result'));
}
async function runSingle() {
  await withBusy('single-submit','Comparing predictions…',async()=>{
    notice($('single-plm').checked?'Comparing models. ESM-2 on a new pair may take several minutes on first use.':'Scoring the three sequence models…');
    state.single=await api('predict',{peptide:$('single-peptide').value,allele:$('single-allele').value,include_plm:$('single-plm').checked});
    renderSingle(); notice('');
  }); marks();
}

function renderBatch() {
  if(!state.batch) return;
  $('batch-count').textContent=`${state.batch.rows.length} scored pairs`;
  $('batch-rows').innerHTML=state.batch.rows.map(r=>{
    const m=r.models[0];return `<tr><td><button type="button" class="peptide-button" data-single="${escapeHTML(pairKey(r))}">${r.peptide}</button></td><td>${escapeHTML(r.allele)}</td><td class="number">${fmt(m.hours)}</td><td class="range">${range(m)}</td><td>${badge(m.tier)}</td><td>${shortlistButton(r)}</td></tr>`;
  }).join(''); $('export-batch').disabled=false; $('save-batch').disabled=false;
}
async function runBatch() {
  await withBusy('batch-submit','Scoring pairs…',async()=>{
    notice('Validating and scoring your batch…');
    const input=$('batch-text').value;
    state.batch=await api('batch',{text:input,arm:$('batch-model').value});
    state.batch.input=input; renderBatch(); notice('');
  }); marks();
}
async function readFile(file) {
  if(!file) return;
  if(!file.name.toLowerCase().endsWith('.csv')) {notice('Choose a CSV file with peptide,allele columns.',true);return;}
  if(file.size>30000) {notice('This batch file is too large. Use at most 200 peptide,allele pairs.',true);return;}
  $('batch-text').value=await file.text(); notice(`Loaded ${file.name}. Select “Score pairs” to run the model.`);
}

function renderComparison() {
  const rows=state.comparison?.rows;
  $('export-comparison').disabled=!rows?.length;
  $('run-comparison').disabled=!state.shortlist.length;
  if(!rows?.length) {
    $('comparison-results').innerHTML=`<p class="empty">${state.shortlist.length?`${state.shortlist.length} shortlisted pairs ready. Select “Compare shortlist” to run the models.`:'Your shortlist is empty. Add candidates from a scan, single pair, or batch.'}</p>`;
  } else {
    const arms=rows[0].models;
    $('comparison-results').innerHTML=`<div class="table-scroll"><table class="table"><thead><tr><th>Peptide · allele</th>${arms.map(m=>`<th>${escapeHTML(m.name)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr><td><span class="mono">${r.peptide}</span><br><span class="small muted">${escapeHTML(r.allele)}</span></td>${r.models.map(m=>`<td class="number">${fmt(m.hours)} h<br><span class="range">${range(m)}</span></td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  marks();
}
async function runComparison() {
  if(!state.shortlist.length) return;
  const keys=state.shortlist.map(pairKey).join(',');
  await withBusy('run-comparison','Comparing shortlist…',async()=>{
    notice('Comparing your shortlisted pairs. ESM-2 on new sequences can take several minutes.');
    const result=await api('compare',{pairs:state.shortlist.map(({peptide,allele})=>({peptide,allele})),include_plm:$('compare-plm').checked});
    if(keys!==state.shortlist.map(pairKey).join(',')) {notice('Your shortlist changed during comparison. Run it again for the current selection.');return;}
    state.comparison=result; renderComparison(); notice('');
  }); marks();
}

function exportRows(rows, filename) {
  const columns=['peptide','allele','position','model','predicted_hours','five_fit_min_hours','five_fit_max_hours','tier','prediction_source'];
  const values=rows.flatMap(r=>r.models.map(m=>[r.peptide,r.allele,r.position??'',m.name,m.hours,m.low,m.high,tierLabel(m.tier),r.prediction_source]));
  const encode=cell=>`"${String(cell).replace(/"/g,'""')}"`;
  const csv=[columns,...values].map(row=>row.map(encode).join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

function readRuns() {
  try {
    const data=JSON.parse(localStorage.getItem(STORAGE_KEY)||'[]');
    state.runs=Array.isArray(data)?data.filter(r=>r.version===1 && r.id && ['scan','batch'].includes(r.type)):[];
  } catch {state.runs=[];notice('Saved runs could not be read from this browser. Existing storage has not been changed.',true);}
}
function beginSave(type) {
  state.saving=type;
  $('run-name').value=type==='scan'?`Protein scan · ${state.scan.window_count} windows`:`Batch · ${state.batch.rows.length} pairs`;
  $('save-dialog').showModal(); $('run-name').focus(); $('run-name').select();
}
function saveRun() {
  const name=$('run-name').value.trim();if(!name) return;
  const run={id:crypto.randomUUID(),version:1,name,date:new Date().toISOString(),type:state.saving,
    shortlist:structuredClone(state.shortlist),scan:state.saving==='scan'?structuredClone(state.scan):null,
    batch:state.saving==='batch'?structuredClone(state.batch):null,rankingAllele:state.rankingAllele,tier:state.tier};
  const runs=[run,...state.runs];
  try {localStorage.setItem(STORAGE_KEY,JSON.stringify(runs));} catch {notice('This browser cannot save more data. Export your CSV or remove an older saved run, then retry.',true);return;}
  state.runs=runs;$('save-dialog').close();renderRuns();notice(`Saved “${name}” in this browser.`);
}
function renderRuns() {
  $('saved-runs').innerHTML=state.runs.map(r=>{
    const alleles=r.type==='scan'?r.scan.alleles:[...new Set(r.batch.rows.map(x=>x.allele))];
    const count=r.type==='scan'?r.scan.window_count:r.batch.rows.length;
    return `<tr><td>${escapeHTML(r.name)}<br><span class="small muted">${r.type==='scan'?'Protein scan':'Batch'}</span></td><td class="small">${escapeHTML(new Date(r.date).toLocaleString())}</td><td class="small">${alleles.map(escapeHTML).join(', ')}</td><td class="number">${count}</td><td class="number">${r.shortlist.length}</td><td><div class="actions"><button class="btn btn-ghost" type="button" data-open-run="${escapeHTML(r.id)}">Open</button><button class="btn btn-ghost" type="button" data-delete-run="${escapeHTML(r.id)}">Delete</button></div></td></tr>`;
  }).join('') || '<tr><td colspan="6" class="empty">No saved runs yet. Save a scan or batch to return to it later.</td></tr>';
}
function openRun(id) {
  const run=state.runs.find(r=>r.id===id);if(!run) return;
  ++state.scanRequest;++state.selectedRequest;state.shortlist=structuredClone(run.shortlist);state.comparison=null;
  if(run.type==='scan') {
    state.scan=structuredClone(run.scan);state.draftAlleles=[...run.scan.alleles];state.rankingAllele=run.rankingAllele;
    state.tier=run.tier;state.shown=12;$('protein').value=run.scan.sequence;$('scan-model').value=run.scan.arm;
    updateSequenceCount();renderAlleles();renderScan();location.hash='scan';
    const row=rankedRows()[0]||state.scan.rows[0];if(row) selectPair(row,false);
  } else {state.batch=structuredClone(run.batch);$('batch-text').value=run.batch.input;$('batch-model').value=run.batch.arm;renderBatch();location.hash='batch';}
  renderShortlist();renderComparison();notice(`Opened “${run.name}”.`);
}

document.addEventListener('click',event=>{
  const remove=event.target.closest('[data-remove-allele]');
  if(remove) {state.draftAlleles=state.draftAlleles.filter(a=>a!==remove.dataset.removeAllele);renderAlleles();return;}
  const filter=event.target.closest('[data-tier]');
  if(filter) {state.tier=filter.dataset.tier;state.shown=12;renderRanked();return;}
  const selected=event.target.closest('[data-select]');
  if(selected) {
    const row=state.scan?.rows.find(r=>pairKey(r)===selected.dataset.select && r.position===Number(selected.dataset.position));
    if(row) selectPair(row,false);return;
  }
  const short=event.target.closest('[data-shortlist]');
  if(short) {const row=allAvailableRows().find(r=>pairKey(r)===short.dataset.shortlist);if(row)toggleShortlist(row);return;}
  const single=event.target.closest('[data-single]');
  if(single) {const [p,a]=single.dataset.single.split('|');$('single-peptide').value=p;$('single-allele').value=a;location.hash='single';runSingle();return;}
  const example=event.target.closest('[data-example]');
  if(example) {const e=state.bootstrap.examples[Number(example.dataset.example)];$('single-peptide').value=e.peptide;$('single-allele').value=e.allele;runSingle();return;}
  const open=event.target.closest('[data-open-run]');if(open){openRun(open.dataset.openRun);return;}
  const del=event.target.closest('[data-delete-run]');
  if(del) {
    const run=state.runs.find(r=>r.id===del.dataset.deleteRun);
    if(!confirm(`Delete saved run “${run.name}” from this browser?`))return;
    const updated=state.runs.filter(r=>r.id!==run.id);
    try{localStorage.setItem(STORAGE_KEY,JSON.stringify(updated));state.runs=updated;renderRuns();}catch{notice('Could not update browser storage.',true);}
  }
});

$('scan-form').onsubmit=event=>{event.preventDefault();runScan();};
$('protein').oninput=updateSequenceCount;
$('load-example').onclick=()=>{$('protein').value=state.bootstrap.example_sequence;updateSequenceCount();runScan();};
$('add-allele-button').onclick=()=>{const value=$('add-allele').value;if(value && state.draftAlleles.length<6 && !state.draftAlleles.includes(value)){state.draftAlleles.push(value);renderAlleles();}};
$('ranking-allele').onchange=()=>{state.rankingAllele=$('ranking-allele').value;state.shown=12;renderRanked();};
$('show-more').onclick=()=>{state.shown+=25;renderRanked();};
$('compare-selected').onclick=()=>withBusy('compare-selected','Comparing all models…',async()=>{
  notice('Comparing all five models for the selected pair…');await selectPair(state.selected,true);
  if(state.selectedResult?.models.length===5)notice('');
});
$('single-form').onsubmit=event=>{event.preventDefault();runSingle();};
$('batch-form').onsubmit=event=>{event.preventDefault();runBatch();};
$('batch-file').onchange=event=>readFile(event.target.files[0]);
$('dropzone').ondragover=event=>{event.preventDefault();$('dropzone').classList.add('dragging');};
$('dropzone').ondragleave=()=>$('dropzone').classList.remove('dragging');
$('dropzone').ondrop=event=>{event.preventDefault();$('dropzone').classList.remove('dragging');readFile(event.dataTransfer.files[0]);};
$('run-comparison').onclick=runComparison;
$('view-comparison').onclick=()=>{location.hash='compare';};
$('export-scan').onclick=()=>exportRows(rankedRows(),'ranked-windows.csv');
$('export-shortlist').onclick=()=>exportRows(state.shortlist,'shortlisted-pairs.csv');
$('export-batch').onclick=()=>exportRows(state.batch.rows,'batch-predictions.csv');
$('export-comparison').onclick=()=>exportRows(state.comparison.rows,'model-comparison.csv');
$('save-scan').onclick=()=>beginSave('scan');$('save-batch').onclick=()=>beginSave('batch');
$('save-form').onsubmit=event=>{event.preventDefault();saveRun();};$('cancel-save').onclick=()=>$('save-dialog').close();

async function initialize() {
  showPage();marks();notice('Loading the research workspace…');
  try {
    state.bootstrap=await api('bootstrap');
    const alleleOptions=state.bootstrap.alleles.map(a=>({value:a.name,label:a.name}));
    const baselineOptions=state.bootstrap.models.filter(m=>m.scan).map(m=>({value:m.id,label:m.name}));
    options($('single-allele'),alleleOptions,state.bootstrap.examples[0].allele);
    options($('scan-model'),baselineOptions,'blosum_nn');options($('batch-model'),baselineOptions,'blosum_nn');
    const defaults=['HLA-B*54:01','HLA-A*02:01','HLA-A*24:02','HLA-B*07:02','HLA-B*13:02','HLA-B*15:01'];
    state.draftAlleles=defaults.filter(a=>alleleOptions.some(o=>o.value===a));state.rankingAllele=state.draftAlleles[0];
    renderAlleles();$('protein').value=state.bootstrap.example_sequence;updateSequenceCount();
    $('single-peptide').value=state.bootstrap.examples[0].peptide;
    $('single-examples').innerHTML=state.bootstrap.examples.map((e,i)=>`<button class="btn btn-ghost" type="button" data-example="${i}">${i===0?'Well-measured':'Sparse'} example · ${escapeHTML(e.allele)} →</button>`).join('');
    $('batch-text').value='peptide,allele\n'+state.bootstrap.examples.map(e=>`${e.peptide},${e.allele}`).join('\n');
    const d=state.bootstrap.dataset;$('dataset-facts').textContent=`${d.rows.toLocaleString()} measured pairs · ${d.alleles} alleles · ${d.zeros.toLocaleString()} reported zeros in the main dataset.`;
    notice('');readRuns();renderRuns();await runScan();
  } catch(error){notice(`Could not initialize the workspace: ${error.message}`,true);}
}
initialize();
