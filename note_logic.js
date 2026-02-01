export function detectDomainFromText(text, stopwordConfig) {
  const s = String(text || "");
  const sig = stopwordConfig?.domainSignals || {};
  let best = "";
  let bestScore = 0;
  for (const [domain, signals] of Object.entries(sig)) {
    const arr = Array.isArray(signals) ? signals : [];
    let score = 0;
    for (const w of arr) {
      const ww = String(w || "").trim();
      if (!ww) continue;
      if (s.includes(ww)) score += 1;
    }
    if (score > bestScore) {
      bestScore = score;
      best = domain;
    }
  }
  return bestScore >= 2 ? best : "";
}

export function buildStopwordSet(domain, stopwordConfig) {
  const base = new Set((stopwordConfig?.base || []).map(String));
  const dom = String(domain || "");
  const ds = stopwordConfig?.domains || {};
  const list = Array.isArray(ds?.[dom]) ? ds[dom] : [];
  for (const w of list) base.add(String(w || ""));
  return base;
}

export function buildAdaptiveStopwordsFromNotes(notes, limit = 10) {
  const s = String(notes || "");
  const tokens = s.match(/[\u4e00-\u9fff]{2,4}/g) || [];
  const freq = new Map();
  for (const t of tokens) {
    const tok = String(t || "").trim();
    if (!tok) continue;
    if (/^(.)\1+$/.test(tok)) continue;
    freq.set(tok, (freq.get(tok) || 0) + 1);
  }
  const top = Array.from(freq.entries()).sort((a, b) => b[1] - a[1]).slice(0, limit).map(([k]) => k);
  return new Set(top);
}

export function extractCnKeywords2to4(text, stopSet) {
  const s = String(text || "");
  const matches = s.match(/[\u4e00-\u9fff]{2,4}/g) || [];
  const out = [];
  for (const t of matches) {
    const token = String(t || "").trim();
    if (!token) continue;
    if (stopSet && stopSet.has(token)) continue;
    if (/^(.)\1+$/.test(token)) continue;
    out.push(token);
  }
  return out;
}

export function computeTfIdfTop(tokens, perDocTokenSets, limit = 14) {
  const tf = new Map();
  for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
  const df = new Map();
  for (const set of perDocTokenSets) {
    for (const t of set) df.set(t, (df.get(t) || 0) + 1);
  }
  const N = perDocTokenSets.length || 1;
  const scored = [];
  for (const [t, c] of tf.entries()) {
    const d = df.get(t) || 1;
    const idf = Math.log((N + 1) / (d + 1)) + 1;
    scored.push([t, c * idf]);
  }
  scored.sort((a, b) => b[1] - a[1] || b[0].length - a[0].length);
  return scored.slice(0, limit).map(([t]) => t);
}

export function collectWrongHighlights(intent, sessionIds, sessionStore, stopwordConfig, limit = 12) {
  const freq = new Map();
  const ids = Array.isArray(sessionIds) ? sessionIds.map(String).filter(Boolean) : [];
  const notesParts = [];
  const perDocTokenSets = [];
  const allTokens = [];

  for (const sid of ids) {
    const session = sessionStore.get(sid);
    if (!session?.history) continue;
    const notes = String(session?.notesRaw || session?.notes || "").trim();
    if (notes) notesParts.push(notes);
  }

  const combinedNotes = notesParts.join("\n");
  const domain = detectDomainFromText(combinedNotes, stopwordConfig);
  const stopSet = buildStopwordSet(domain, stopwordConfig);
  const adaptive = buildAdaptiveStopwordsFromNotes(combinedNotes, 10);
  for (const w of adaptive) stopSet.add(w);

  for (const sid of ids) {
    const session = sessionStore.get(sid);
    if (!session?.history) continue;
    for (const entry of session.history) {
      if (!entry?.results || !entry?.quiz?.questions) continue;
      for (const resItem of entry.results) {
        if (resItem?.isCorrect !== false) continue;
        const qDetail = entry.quiz.questions.find(q => q.id === resItem.id);
        if (!qDetail) continue;
        if (String(qDetail.intent || "").toUpperCase() !== String(intent).toUpperCase()) continue;
        const c = String(qDetail.concept || "").trim();
        if (c && c.length <= 20) freq.set(c, (freq.get(c) || 0) + 3);
        const stem = String(qDetail.stem || "");
        const kws = extractCnKeywords2to4(stem, stopSet);
        const docSet = new Set(kws);
        perDocTokenSets.push(docSet);
        for (const kw of kws) allTokens.push(kw);
      }
    }
  }
  const tfidfTop = computeTfIdfTop(allTokens, perDocTokenSets, 14);
  for (const t of tfidfTop) freq.set(t, (freq.get(t) || 0) + 2);
  return Array.from(freq.entries())
    .sort((a, b) => (b[1] - a[1]) || (String(b[0]).length - String(a[0]).length))
    .slice(0, limit)
    .map(([k]) => k);
}

export function estimateNoteIntentEtaMs(intent, notesLen) {
  const base = intent === "A" ? 9500 : (intent === "D" ? 9000 : (intent === "B" ? 8000 : 7000));
  const extra = Math.max(0, Number(notesLen) || 0) / 1200 * 1200;
  const ms = base + extra;
  return Math.max(5000, Math.min(25000, Math.round(ms)));
}

