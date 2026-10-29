const methodData = {
  baseline: { label: 'REFERENCE MODEL', title: 'Baseline logistic regression', description: 'The model does not receive gender as an input. Even so, correlated features lead to a substantial approval-rate gap.', note: 'Use as the reference point — not as a fair outcome.', dir: '0.718', eod: '0.164', auc: '0.745', profit: '₹246.3' },
  reweigh: { label: 'PRE-PROCESSING', title: 'Reweighing', description: 'Training examples are weighted to reduce dependence between group membership and the repayment label. In this experiment, the approval disparity remained.', note: 'Pre-processing did not address the dominant proxy effects here.', dir: '0.665', eod: '0.181', auc: '0.745', profit: '₹253.3' },
  eo: { label: 'POST-PROCESSING', title: 'Equal-opportunity thresholds', description: 'Group-specific thresholds are selected on validation data to bring the true-positive-rate gap within a specified tolerance.', note: 'Uses the protected attribute at decision time; this raises governance questions.', dir: '0.909', eod: '0.051', auc: '0.745', profit: '₹267.6' },
  dir: { label: 'CONSTRAINED THRESHOLDS', title: 'Disparate impact ≥ 0.80', description: 'The threshold pair is chosen to maximise validation profit while meeting the DIR constraint, then evaluated on a held-out test set.', note: 'The 0.80 ratio is a screening benchmark, not a legal determination.', dir: '0.980', eod: '0.011', auc: '0.745', profit: '₹267.1' },
  combined: { label: 'COMBINED APPROACH', title: 'Reweighing + equal opportunity', description: 'Combines weighted training with post-processing thresholds. It achieved a high DIR in the seed-42 test split.', note: 'Results are synthetic and should not be interpreted as real-world performance.', dir: '0.967', eod: '0.020', auc: '0.745', profit: '₹269.1' }
};
document.querySelectorAll('.method-tab').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.method-tab').forEach(tab => { tab.classList.remove('active'); tab.setAttribute('aria-selected', 'false'); });
    button.classList.add('active'); button.setAttribute('aria-selected', 'true');
    const d = methodData[button.dataset.method];
    document.getElementById('method-label').textContent = d.label;
    document.getElementById('method-title').textContent = d.title;
    document.getElementById('method-description').textContent = d.description;
    document.getElementById('method-note').textContent = d.note;
    document.getElementById('method-dir').textContent = d.dir;
    document.getElementById('method-eod').textContent = d.eod;
    document.getElementById('method-auc').textContent = d.auc;
    document.getElementById('method-profit').textContent = d.profit;
  });
});
const menuToggle = document.querySelector('.menu-toggle');
const nav = document.querySelector('.main-nav');
menuToggle.addEventListener('click', () => {
  const open = nav.classList.toggle('open');
  menuToggle.setAttribute('aria-expanded', String(open));
  menuToggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
});
nav.querySelectorAll('a').forEach(link => link.addEventListener('click', () => { nav.classList.remove('open'); menuToggle.setAttribute('aria-expanded', 'false'); }));

// Responsive interaction layer and local-only educational tools.
(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const toast = $('#toast');
  let toastTimer;
  function notify(message) {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 2600);
  }

  // Appearance toggle; preference is kept on this device only.
  const themeToggle = $('#themeToggle');
  const savedTheme = (() => { try { return localStorage.getItem('faircredit-theme'); } catch { return null; } })();
  if (savedTheme === 'dark') document.body.classList.add('dark-mode');
  if (themeToggle) {
    themeToggle.textContent = document.body.classList.contains('dark-mode') ? '☀' : '◐';
    themeToggle.addEventListener('click', () => {
      document.body.classList.toggle('dark-mode');
      const dark = document.body.classList.contains('dark-mode');
      themeToggle.textContent = dark ? '☀' : '◐';
      try { localStorage.setItem('faircredit-theme', dark ? 'dark' : 'light'); } catch {}
    });
  }

  // Accessible mobile navigation closes after choosing a section or pressing Escape.
  const nav = $('.main-nav');
  const menuToggle = $('.menu-toggle');
  const closeNav = () => {
    if (!nav || !menuToggle) return;
    nav.classList.remove('open');
    menuToggle.setAttribute('aria-expanded', 'false');
    menuToggle.setAttribute('aria-label', 'Open navigation');
  };
  if (nav && menuToggle) {
    nav.querySelectorAll('a').forEach(link => link.addEventListener('click', closeNav));
    document.addEventListener('keydown', event => { if (event.key === 'Escape') closeNav(); });
  }

  // Applicant playground: transparent illustrative formula, not the trained model.
  const controls = {
    income: $('#incomeRange'), bill: $('#billRange'), upi: $('#upiRange'), stability: $('#stabilityRange'), method: $('#demoMethod')
  };
  const money = n => '₹' + Number(n).toLocaleString('en-IN');
  function updateDemo() {
    if (!controls.income) return;
    const income = Number(controls.income.value);
    const bills = Number(controls.bill.value);
    const upi = Number(controls.upi.value);
    const stability = Number(controls.stability.value);
    const method = controls.method.value;
    $('#incomeOutput').textContent = money(income);
    $('#billOutput').textContent = `${bills} / 10`;
    $('#upiOutput').textContent = `${upi} transactions`;
    $('#stabilityOutput').textContent = `${stability} / 10`;
    let score = 18 + (Math.log(income / 8000) / Math.log(10)) * 22 + bills * 3.2 + upi * 0.13 + stability * 1.35;
    if (method === 'eo') score = score * 0.92 + 5;
    if (method === 'dir') score = score * 0.88 + 8;
    score = Math.round(Math.max(0, Math.min(100, score)));
    $('#demoScore').innerHTML = `${score}<span>/100</span>`;
    $('#ringValue').textContent = `${score}%`;
    $('#scoreRing').style.setProperty('--score', score);
    $('#scoreMeter').style.width = `${score}%`;
    const outcome = score >= 72 ? ['Review recommended', 'MANUAL REVIEW'] : score >= 54 ? ['Additional information', 'MORE INFO'] : ['Insufficient demo signals', 'REVIEW NEEDED'];
    $('#demoOutcome').textContent = outcome[0];
    $('#outcomePill').textContent = outcome[1];
    const reasons = [];
    reasons.push([bills >= 7, bills >= 7 ? 'Consistent bill payment signal' : 'Limited on-time bill payment signal']);
    reasons.push([income >= 25000, income >= 25000 ? 'Income signal contributes positively' : 'Lower income in this illustrative formula']);
    reasons.push([upi >= 30, upi >= 30 ? 'Regular transaction activity signal' : 'Lower transaction activity signal']);
    reasons.push([stability >= 6, stability >= 6 ? 'Higher address stability signal' : 'Lower address stability signal']);
    $('#reasonList').innerHTML = reasons.map(([positive, text]) => `<div class="reason-item ${positive ? '' : 'negative'}"><span>${positive ? '+' : '−'}</span>${text}</div>`).join('') + '<div class="reason-item negative"><span>!</span>This toy score ignores many real-world factors and can be misleading.</div>';
  }
  Object.values(controls).forEach(control => control && control.addEventListener('input', updateDemo));
  $('#resetDemo')?.addEventListener('click', () => {
    controls.income.value = 25000; controls.bill.value = 8; controls.upi.value = 35; controls.stability.value = 6; controls.method.value = 'baseline'; updateDemo(); notify('Applicant example reset.');
  });
  updateDemo();

  // Metric explorer.
  const metricData = {
    dir: { eyebrow: 'APPROVAL RATE RATIO', title: 'Disparate impact ratio (DIR)', description: 'Female approval rate divided by male approval rate. A value closer to 1 indicates closer approval rates; 0.80 is a common screening benchmark, not a universal legal test.', formula: 'DIR = female approval rate ÷ male approval rate', baseline: 0.718, combined: 0.967, direction: 'Higher is closer to approval-rate parity.' },
    spd: { eyebrow: 'ABSOLUTE APPROVAL GAP', title: 'Statistical parity difference (SPD)', description: 'Female approval rate minus male approval rate. Negative values here mean the female approval rate is lower. Values nearer zero indicate a smaller gap.', formula: 'SPD = female approval rate − male approval rate', baseline: -0.157, combined: -0.016, direction: 'Closer to zero indicates a smaller approval-rate gap.' },
    eod: { eyebrow: 'TRUE-POSITIVE-RATE GAP', title: 'Equal opportunity difference (EOD)', description: 'Difference in true-positive rates between groups among applicants who would repay. A smaller absolute gap means the groups have more similar true-positive rates.', formula: 'EOD = female TPR − male TPR', baseline: -0.164, combined: -0.020, direction: 'Closer to zero indicates a smaller true-positive-rate gap.' },
    auc: { eyebrow: 'RANKING DISCRIMINATION', title: 'Area under ROC curve (AUC)', description: 'Measures how well a model ranks repayers above non-repayers. AUC does not by itself measure fairness or whether predicted probabilities are calibrated.', formula: 'AUC = area under the ROC curve', baseline: 0.745, combined: 0.745, direction: 'Higher indicates stronger ranking performance, not fairness.' }
  };
  $$('.metric-pick').forEach(button => button.addEventListener('click', () => {
    $$('.metric-pick').forEach(item => { item.classList.remove('active'); item.setAttribute('aria-selected', 'false'); });
    button.classList.add('active'); button.setAttribute('aria-selected', 'true');
    const data = metricData[button.dataset.metric];
    $('#metricEyebrow').textContent = data.eyebrow; $('#metricTitle').textContent = data.title;
    $('#metricDescription').textContent = data.description; $('#metricFormula').textContent = data.formula;
    const format = n => (n < 0 ? '−' + Math.abs(n).toFixed(3) : n.toFixed(3));
    $('#metricBaseline').textContent = format(data.baseline); $('#metricCombined').textContent = format(data.combined);
    const max = Math.max(Math.abs(data.baseline), Math.abs(data.combined), 0.001);
    $('#baselineBar').style.width = `${Math.max(4, Math.abs(data.baseline) / max * 100)}%`;
    $('#combinedBar').style.width = `${Math.max(4, Math.abs(data.combined) / max * 100)}%`;
    $('#metricDirection').textContent = data.direction;
  }));

  // Filter model table and export the visible experiment data to CSV.
  const search = $('#tableSearch');
  search?.addEventListener('input', () => {
    const query = search.value.trim().toLowerCase();
    $$('.table-scroll tbody tr').forEach(row => { row.hidden = !row.textContent.toLowerCase().includes(query); });
  });
  $('#exportMetrics')?.addEventListener('click', () => {
    const table = $('.table-scroll table');
    if (!table) return;
    const csv = [...table.rows].map(row => [...row.cells].map(cell => '"' + cell.innerText.replace(/"/g, '""').replace(/₹/g, 'INR ') + '"').join(',')).join('\r\n');
    const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'fair-credit-model-comparison.csv'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    notify('Model comparison CSV downloaded.');
  });

  // Click any chart to expand it; supports Escape and backdrop close.
  const modal = $('#chartModal'); const modalImage = $('#modalImage'); const modalCaption = $('#modalCaption');
  $$('.chart-img, .method-bottom img').forEach(img => {
    img.setAttribute('tabindex', '0'); img.setAttribute('role', 'button'); img.setAttribute('aria-label', 'Expand chart: ' + (img.alt || 'project chart'));
    const open = () => { modalImage.src = img.src; modalImage.alt = img.alt; modalCaption.textContent = img.alt || 'Project chart'; modal.classList.add('open'); modal.setAttribute('aria-hidden', 'false'); $('#modalClose').focus(); };
    img.addEventListener('click', open); img.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); } });
  });
  const closeModal = () => { modal.classList.remove('open'); modal.setAttribute('aria-hidden', 'true'); modalImage.src = ''; };
  $('#modalClose')?.addEventListener('click', closeModal);
  modal?.addEventListener('click', event => { if (event.target === modal) closeModal(); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && modal?.classList.contains('open')) closeModal(); });

  // Small scroll reveals; content remains visible if observer is unsupported.
  if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    document.body.classList.add('reveal-ready');
    const targets = $$('.panel, .governance-card, .pipeline-step, .proxy-point');
    targets.forEach(el => el.classList.add('reveal'));
    const observer = new IntersectionObserver(entries => entries.forEach(entry => { if (entry.isIntersecting) { entry.target.classList.add('in-view'); observer.unobserve(entry.target); } }), { threshold: 0.08 });
    targets.forEach(el => observer.observe(el));
  }
})();
