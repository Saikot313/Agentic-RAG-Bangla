const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const fileChipHolder = document.getElementById('fileChipHolder');
const uploadBtn = document.getElementById('uploadBtn');
const uploadStatus = document.getElementById('uploadStatus');
const askBtn = document.getElementById('askBtn');
const queryStatus = document.getElementById('queryStatus');
const results = document.getElementById('results');

let selectedFile = null;

function setFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    uploadStatus.innerHTML = '<span class="err">Only PDF files are supported</span>';
    return;
  }
  selectedFile = file;
  uploadBtn.disabled = false;
  fileChipHolder.innerHTML = `<div class="file-chip"><span class="dot"></span>${file.name}</div>`;
  uploadStatus.textContent = '';
}

dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => setFile(e.target.files[0]));
['dragover', 'dragenter'].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('drag'); }));
['dragleave', 'drop'].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove('drag'); }));
dropzone.addEventListener('drop', e => { e.preventDefault(); setFile(e.dataTransfer.files[0]); });

uploadBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  uploadBtn.disabled = true;
  uploadStatus.innerHTML = '<span class="spinner"></span> indexing…';
  const form = new FormData();
  form.append('file', selectedFile);
  try {
    const res = await fetch('/upload', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    uploadStatus.textContent = `✓ ${data.chunks_indexed} chunks indexed — languages: ${data.detected_languages.join(', ')}`;
  } catch (err) {
    uploadStatus.innerHTML = `<span class="err">✕ ${err.message}</span>`;
  } finally {
    uploadBtn.disabled = false;
  }
});

askBtn.addEventListener('click', async () => {
  const question = document.getElementById('question').value.trim();
  if (!question) {
    queryStatus.innerHTML = '<span class="err">Please enter a question</span>';
    return;
  }
  askBtn.disabled = true;
  queryStatus.innerHTML = '<span class="spinner"></span> agents working…';
  results.classList.remove('show');
  document.getElementById('seal').classList.remove('stamped');

  const payload = {
    question,
    bilingual_answer: document.getElementById('bilingual').checked,
    top_k: parseInt(document.getElementById('topK').value, 10) || null,
  };

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Query failed');
    renderResult(data);
    queryStatus.textContent = '';
  } catch (err) {
    queryStatus.innerHTML = `<span class="err">✕ ${err.message}</span>`;
  } finally {
    askBtn.disabled = false;
  }
});

function renderResult(data) {
  document.getElementById('answerText').textContent = data.answer;
  document.getElementById('detLang').textContent = data.detected_language;
  document.getElementById('rewritten').textContent = data.rewritten_query;

  const retryHolder = document.getElementById('retryFlagHolder');
  retryHolder.innerHTML = data.retried
    ? '<span class="retry-flag">↻ retried after low-confidence verification</span>' : '';

  const claimsHolder = document.getElementById('claimsHolder');
  if (data.verification.unsupported_claims && data.verification.unsupported_claims.length) {
    claimsHolder.innerHTML = `<div class="claims"><b>Unsupported claims flagged</b>${data.verification.unsupported_claims.map(c => `<div>· ${escapeHtml(c)}</div>`).join('')
      }</div>`;
  } else {
    claimsHolder.innerHTML = '';
  }

  const seal = document.getElementById('seal');
  const sealLabel = document.getElementById('sealLabel');
  const sealPct = document.getElementById('sealPct');
  const pct = Math.round(data.verification.confidence * 100);
  sealPct.textContent = pct + '%';
  if (data.verification.is_grounded) {
    seal.classList.remove('warn');
    sealLabel.textContent = 'Grounded';
  } else {
    seal.classList.add('warn');
    sealLabel.textContent = 'Ungrounded';
  }

  const chunkList = document.getElementById('chunkList');
  chunkList.innerHTML = data.retrieved_chunks.map(c => `
    <div class="chunk">
      <div class="chunk-head"><span>${escapeHtml(c.source)}</span><span class="score">score ${c.score.toFixed(3)}</span></div>
      <div>${escapeHtml(c.text)}</div>
    </div>
  `).join('');

  results.classList.add('show');
  requestAnimationFrame(() => requestAnimationFrame(() => seal.classList.add('stamped')));
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
