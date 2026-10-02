const state={summary:null,task:null,runs:{sft:null,rl:null},tab:'trajectory'};
const labels={both_success:'都成功',rl_gained:'仅 RL 成功',sft_only:'仅 SFT 成功',both_failed:'都失败'};
const $=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const yes=r=>r?.reward===1;
const fmt=n=>n==null?'—':Number(n).toLocaleString();

async function init(){
  state.summary=await fetch('data/summary.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()});
  renderCards(); renderRows(); $('status').textContent=`${state.summary.total} tasks · SFT / RL`;
  const task=new URLSearchParams(location.search).get('task'); if(task) await openTask(task);
}
function renderCards(){const c=state.summary.counts; $('cards').innerHTML=[['总任务',state.summary.total],['都成功',c.both_success],['RL 新增成功',c.rl_gained],['SFT 独有成功',c.sft_only],['都失败',c.both_failed]].map(([a,b])=>`<div class="card"><strong>${b}</strong><span>${a}</span></div>`).join('')}
function renderRows(){
  const q=$('search').value.toLowerCase(),cat=$('category').value,sort=$('sort').value;
  let rows=state.summary.tasks.filter(t=>(cat==='all'||t.category===cat)&&`${t.instance_id} ${t.repo} ${t.problem_statement}`.toLowerCase().includes(q));
  rows.sort((a,b)=>sort==='category'?a.category.localeCompare(b.category):sort==='steps'?Math.max(b.sft?.steps||0,b.rl?.steps||0)-Math.max(a.sft?.steps||0,a.rl?.steps||0):sort==='tokens'?Math.max(b.sft?.completion_tokens||0,b.rl?.completion_tokens||0)-Math.max(a.sft?.completion_tokens||0,a.rl?.completion_tokens||0):a.instance_id.localeCompare(b.instance_id));
  $('taskRows').innerHTML=rows.map(t=>`<tr data-id="${esc(t.instance_id)}"><td><div class="task-id">${esc(t.instance_id)}</div><small>${esc(t.repo||'')}</small></td><td><span class="badge">${labels[t.category]}</span></td><td>${runCell(t.sft)}</td><td>${runCell(t.rl)}</td><td class="preview">${esc(t.problem_preview)}</td></tr>`).join('');
  document.querySelectorAll('#taskRows tr').forEach(tr=>tr.onclick=()=>openTask(tr.dataset.id));
}
function runCell(r){if(!r)return '—';return `<div class="metric ${yes(r)?'ok':'fail'}">reward ${esc(r.reward)}</div><small>${fmt(r.steps)} steps · ${fmt(r.completion_tokens)} tok<br>${esc(r.exit_reason||'—')}</small>`}
async function openTask(id){
  const t=state.summary.tasks.find(x=>x.instance_id===id); if(!t)return;
  state.task=t; const load=async label=>fetch(`data/tasks/${t.safe_id}/${label}.json`).then(r=>r.ok?r.json():null);
  [state.runs.sft,state.runs.rl]=await Promise.all([load('sft'),load('rl')]);
  $('dashboard').hidden=true;$('detail').hidden=false;$('taskTitle').textContent=t.instance_id;$('taskRepo').textContent=t.repo||'';$('taskCategory').textContent=labels[t.category];$('problem').textContent=t.problem_statement;$('runMetrics').innerHTML=['sft','rl'].map(runMetric).join('');
  history.replaceState(null,'',`?task=${encodeURIComponent(id)}`);renderTab();scrollTo(0,0);
}
function runMetric(label){const r=state.runs[label];if(!r)return `<div class="run-card"><h3>${label.toUpperCase()}</h3>无数据</div>`;return `<div class="run-card"><h3>${label.toUpperCase()}</h3><b class="${yes(r)?'ok':'fail'}">reward ${esc(r.reward)}</b> · ${r.trajectory_steps?.length||0} steps · ${fmt(r.trajectory_completion_tokens)} completion tokens<br><span>exit: ${esc(r.exit_reason||'—')} · patch: ${fmt((r.output_patch||'').length)} chars · tests: ${fmt((r.test_output||'').length)} chars</span></div>`}
function renderTab(){document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('active',b.dataset.tab===state.tab));$('tabContent').innerHTML=`<div class="compare">${['sft','rl'].map(label=>column(label,state.runs[label])).join('')}</div>`}
function column(label,r){return `<section class="column"><h3>${label.toUpperCase()}</h3>${r?content(r):'<div class="empty">无此运行</div>'}</section>`}
function content(r){if(state.tab==='trajectory')return `<div class="steps">${(r.trajectory_steps||[]).map(step).join('')||'<div class="empty">无轨迹步骤</div>'}</div>`;if(state.tab==='patch')return `<pre>${esc(r.output_patch||'无 patch')}</pre>`;if(state.tab==='tests')return `<details open><summary>test_output</summary><pre>${esc(r.test_output||'')}</pre></details><details><summary>regression_test_output</summary><pre>${esc(r.regression_test_output||'')}</pre></details>`;return `<pre>${esc(JSON.stringify(r,null,2))}</pre>`}
function step(s,i){return `<details class="step" ${i===0?'open':''}><summary><b>Step ${esc(s.step_idx??i)}</b><span>${fmt(s.token_usage_prompt)} in / ${fmt(s.token_usage_completion)} out</span></summary><div class="step-body"><h4>Thought</h4><pre>${esc(s.thought||'')}</pre><h4>Action</h4><pre>${esc(typeof s.action==='string'?s.action:JSON.stringify(s.action,null,2))}</pre><h4>Observation</h4><pre>${esc(typeof s.observation==='string'?s.observation:JSON.stringify(s.observation,null,2))}</pre></div></details>`}
$('search').oninput=renderRows;$('category').onchange=renderRows;$('sort').onchange=renderRows;
$('back').onclick=()=>{$('detail').hidden=true;$('dashboard').hidden=false;history.replaceState(null,'',location.pathname);scrollTo(0,0)};
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{state.tab=b.dataset.tab;renderTab()});
init().catch(e=>{$('status').textContent='加载失败：'+e.message});
