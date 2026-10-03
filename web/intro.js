/* Decorative introduction only. Real scoring starts independently in workspace.js. */
(() => {
  'use strict';
  const dialog = document.getElementById('intro');
  const svg = document.getElementById('intro-sequence');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const NS = 'http://www.w3.org/2000/svg';
  let finishTimer, closeTimer;

  function element(name, attributes={}, text) {
    const node = document.createElementNS(NS, name);
    Object.entries(attributes).forEach(([key,value]) => node.setAttribute(key,value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function wave(y, amplitude) {
    let path = `M60 ${y}`;
    for (let x=60;x<=660;x+=10) path += ` L${x} ${(y+amplitude*Math.sin((x-60)/38)).toFixed(1)}`;
    return path;
  }
  [[88,10,150,false],[100,10,300,true],[232,-10,400,false],[220,-10,550,true]].forEach(([y,amplitude,delay,secondary]) => {
    svg.append(element('path',{d:wave(y,amplitude),pathLength:1,
      class:`intro-helix${secondary?' secondary':''}`,style:`--delay:${delay}ms`}));
  });
  svg.append(element('text',{x:60,y:62,class:'intro-label'},'α1 HELIX'));
  svg.append(element('text',{x:60,y:276,class:'intro-label'},'α2 HELIX'));
  const windowGroup = element('g',{class:'intro-window'});
  const windowContent = element('g',{class:'intro-window-content'});
  windowContent.append(element('rect',{x:60,y:124,width:360,height:72}));
  [[60,124],[420,124],[60,196],[420,196]].forEach(([x,y]) => {
    windowContent.append(element('path',{d:`M${x-5} ${y} H${x+5} M${x} ${y-5} V${y+5}`}));
  });
  windowContent.append(element('text',{x:240,y:214,'text-anchor':'middle'},'9-MER WINDOW'));
  windowGroup.append(windowContent); svg.append(windowGroup);
  'DGEKTYVPHLASPRWN'.split('').forEach((letter,i) => {
    // The final 360-unit window starts 200 units (five residues) along the rail.
    const residue = element('g',{class:`intro-residue${i>=5&&i<=13?' is-inside':''}`,
      style:`--dx:${(i-8)*30}px;--delay:${550+i*50}ms`});
    residue.append(element('text',{x:80+i*40,y:168,'text-anchor':'middle'},letter));
    svg.append(residue);
  });

  function cleanup() {
    clearTimeout(finishTimer); clearTimeout(closeTimer);
    dialog.classList.remove('intro-playing','is-leaving');
  }
  function finish(immediate=false) {
    clearTimeout(finishTimer);
    if (!dialog.open) return;
    if (immediate || reducedMotion.matches) { dialog.close(); cleanup(); return; }
    if (dialog.classList.contains('is-leaving')) return;
    dialog.classList.add('is-leaving');
    closeTimer = setTimeout(() => { dialog.close(); cleanup(); },550);
  }
  function play(automatic=false) {
    if (dialog.open || (automatic && reducedMotion.matches)) return;
    cleanup(); dialog.showModal();
    dialog.classList.add('intro-playing');
    // Dock/window finish at 2.8s, tagline at 3.4s, then hold before fading.
    finishTimer = setTimeout(() => finish(),4200);
  }
  dialog.addEventListener('close',cleanup);
  dialog.addEventListener('cancel',event => {event.preventDefault();finish(true);});
  document.getElementById('skip-intro').addEventListener('click',()=>finish(true));
  document.getElementById('replay-intro').addEventListener('click',()=>play());
  reducedMotion.addEventListener('change',event => { if(event.matches)finish(true); });
  play(true);
})();
