const oldSelect = document.querySelector('#old-doc');
const newSelect = document.querySelector('#new-doc');
const statusEl = document.querySelector('#status');
const changeRoot = document.querySelector('#changes');
const metaRoot = document.querySelector('#doc-meta');
const compareButton = document.querySelector('#compare');
const uploadForm = document.querySelector('#upload-form');
const uploadButton = document.querySelector('#upload-compare');
const aiHealthButton = document.querySelector('#ai-health-check') || document.querySelector('#gemini-health-check');
const aiHealthResult = document.querySelector('#ai-health-result') || document.querySelector('#gemini-health-result');
const aiOfflineBanner = document.querySelector('#ai-offline-banner');
const aiOfflineDetail = document.querySelector('#ai-offline-detail');
let samples = [];
let visibleComparison = null;
let analysisPollSequence = 0;
const MAX_ANALYSIS_POLL_MS = 60000;

/** Show or hide the AI-offline banner. When visible, it displays the actual
 *  status message from the backend without making extra API calls. */
function setAiOfflineBanner(visible, detail) {
  if (!aiOfflineBanner) return;
  aiOfflineBanner.hidden = !visible;
  const defDetail = typeof t === 'function' ? t('banner.ai_offline_detail') : ' Document comparison, source verification, evidence highlights and all filters remain fully available.';
  if (visible && detail && aiOfflineDetail) {
    aiOfflineDetail.textContent = ' ' + detail + ' ' + defDetail;
  } else if (!visible && aiOfflineDetail) {
    aiOfflineDetail.textContent = ' ' + defDetail;
  }
}

function option(doc) { return `<option value="${doc.id}">${doc.date} · ${doc.version}</option>`; }

function sourceLink(ref) {
  const quote = encodeURIComponent(ref.text.split(/\s+/).slice(0, 9).join(' '));
  if (ref.source_path) return `${ref.source_path}#page=${ref.page}&search=${quote}`;
  const pdf = `/documents/${encodeURIComponent(ref.version === samples[0].version ? 'cgfmu_2023.pdf' : 'cgfmu_2024.pdf')}#page=${ref.page}&search=${quote}`;
  return pdf;
}

function quoteCard(ref, label) {
  const notFoundText = typeof t === 'function' ? t('card.no_earlier_passage') : 'No corresponding passage was found in this version.';
  if (!ref) return `<article class="quote"><div class="quote-label">${escapeHtml(label)}</div><p class="source-note">${escapeHtml(notFoundText)}</p></article>`;
  const pageRefs = ref.page_refs || [{page: ref.page, text: ref.text, evidence_segments: ref.evidence_segments}];
  const pageText = typeof t === 'function' ? t('card.page') : 'Page';
  const viewSourceText = typeof t === 'function' ? t('card.view_source') : 'View source ↗';
  const citations = pageRefs.map(pageRef => {
    const link = sourceLink({...ref, page: pageRef.page, text: pageRef.text});
    const segments = pageRef.evidence_segments || [pageRef.text];
    return `<div class="evidence-page"><div class="quote-label"><span>${pageText} ${pageRef.page}</span><a href="${link}" target="_blank" rel="noopener">${viewSourceText}</a></div><blockquote>${segments.map((segment, index) => `${index ? '<br>' : ''}<mark>${escapeHtml(segment)}</mark>`).join('')}</blockquote></div>`;
  }).join('');
  return `<article class="quote"><div class="quote-label"><span>${escapeHtml(label)}</span></div>${citations}<small>${escapeHtml(ref.document)} · ${escapeHtml(ref.version)}</small><div class="source-note">${escapeHtml(ref.host_note)}</div></article>`;
}

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function claimBadge(status) {
  if (status === 'interpretation') return `<small class="claim-status interpretation">${escapeHtml(typeof t === 'function' ? t('badge.interpretation') : 'Interpretation · not a verified policy fact')}</small>`;
  if (status === 'insufficient_evidence') return `<small class="claim-status insufficient">${escapeHtml(typeof t === 'function' ? t('badge.insufficient') : 'Not established by the cited passages')}</small>`;
  return `<small class="claim-status">${escapeHtml(typeof t === 'function' ? t('badge.aligned') : 'Evidence-aligned · automated text check')}</small>`;
}

function claimBadgeNode(status) {
  const badge = document.createElement('small');
  badge.className = `claim-status${status === 'interpretation' ? ' interpretation' : status === 'insufficient_evidence' ? ' insufficient' : ''}`;
  badge.textContent = status === 'interpretation'
    ? (typeof t === 'function' ? t('badge.interpretation') : 'Interpretation · not a verified policy fact')
    : status === 'insufficient_evidence'
      ? (typeof t === 'function' ? t('badge.insufficient') : 'Not established by the cited passages')
      : (typeof t === 'function' ? t('badge.aligned') : 'Evidence-aligned · automated text check');
  return badge;
}

function resolvePolicyCategory(change) {
  const rawCat = change.change_category || '';
  const text = [
    change.type || '',
    change.summary || '',
    change.updated?.text || '',
    change.previous?.text || ''
  ].join(' ').toLowerCase();

  if (/\b(?:repay\w*|repaid|recovery|emi\b|amortiz\w*|installment\w*)\b/i.test(text)) {
    return 'Repayment';
  }
  if (rawCat === 'Financial assistance' || /\b(?:loan\b|subsidy|grant\b|guarantee|financial assistance|lakh|crore|fee\b|interest|borrower limit|loan limit|credit limit)\b/i.test(text)) {
    return 'Finance / Financial Assistance';
  }
  if (rawCat === 'Eligibility' || /\b(?:eligib\w*|qualif\w*|entitl\w*|criterion|criteria|age limit|income ceiling)\b/i.test(text)) {
    return 'Eligibility';
  }
  if (rawCat === 'Application process' || /\b(?:apply|application|portal|submission|submit|procedure|form)\b/i.test(text)) {
    return 'Application Process';
  }
  if (rawCat === 'Required documents' || /\b(?:document\w*|certificate|proof|records|attachment)\b/i.test(text)) {
    return 'Documentation';
  }
  if (rawCat === 'Implementation' || /\b(?:implement\w*|commence|operational|roll.?out|transition)\b/i.test(text)) {
    return 'Implementation';
  }
  if (rawCat === 'Deadlines' || /\b(?:deadline\b|due date|valid until|expiry|expires?)\b/i.test(text)) {
    return 'Deadlines';
  }
  if (rawCat === 'Responsibilities' || /\b(?:responsib\w*|authority|ministry|agency|nodal)\b/i.test(text)) {
    return 'Responsibilities';
  }
  if (rawCat === 'Monitoring' || /\b(?:monitor\w*|audit\b|inspection|reporting|oversight)\b/i.test(text)) {
    return 'Monitoring';
  }
  if (rawCat && rawCat !== 'Other') {
    return rawCat;
  }
  return 'Other';
}

function resolveChangeTitle(c, category) {
  if (c.title) return c.title;
  const section = c.updated_section || c.previous_section || c.policy_group;
  if (section && section !== 'Other' && !section.startsWith('Other')) {
    return `${section} · ${c.type}`;
  }
  return `${category} · ${c.type}`;
}

function resolveChangeAudiences(change) {
  const text = [
    change.type || '',
    change.summary || '',
    change.policy_group || '',
    Array.isArray(change.who_may_be_affected) ? change.who_may_be_affected.join(' ') : (change.who_may_be_affected || ''),
    Array.isArray(change.potentially_affected_groups) ? change.potentially_affected_groups.join(' ') : '',
    change.previous?.text || '',
    change.updated?.text || ''
  ].join(' ').toLowerCase();

  const audiences = [];
  if (/\b(?:business(?:es)?|enterprise(?:s)?|entrepreneur(?:s)?|msme(?:s)?|firm(?:s)?|compan(?:y|ies)|startup(?:s)?|industry|commercial|borrower(?:s)?|trade|lender(?:s)?|bank(?:s)?|micro unit(?:s)?)\b/i.test(text)) {
    audiences.push('Businesses');
  }
  if (/\b(?:student(?:s)?|education(?:al)?|scholar(?:ship|s)?|academic|university|college|school|youth|training|apprentice(?:s)?)\b/i.test(text)) {
    audiences.push('Students');
  }
  if (/\b(?:citizen(?:s)?|individual(?:s)?|person(?:s)?|public|consumer(?:s)?|beneficiar(?:y|ies)|applicant(?:s)?|household(?:s)?|famil(?:y|ies)|women|farmer(?:s)?|borrower(?:s)?)\b/i.test(text)) {
    audiences.push('General Citizens');
  }
  return audiences;
}

async function loadComparison() {
  compareButton.disabled = true;
  statusEl.textContent = typeof t === 'function' ? t('status.extracting') : 'Extracting and checking page-level evidence…';
  changeRoot.setAttribute('aria-busy', 'true');
  try {
    const res = await fetch(`/api/compare?old=${encodeURIComponent(oldSelect.value)}&new=${encodeURIComponent(newSelect.value)}`);
    const data = await res.json(); if (!res.ok) throw new Error(data.error || 'Comparison failed.');
    renderComparison(data);
  } catch (err) { statusEl.textContent = err.message; }
  finally { compareButton.disabled = false; changeRoot.setAttribute('aria-busy', 'false'); }
}

function renderComparison(data) {
  const renderStarted = performance.now();
  visibleComparison = data;
  changeRoot.setAttribute('aria-busy', 'false');
  const hasChanges = Boolean(data.comparison_id && data.changes?.length);

  setAiOfflineBanner(data.ai_status === 'unavailable', data.ai_error || null);
  const pollSequence = ++analysisPollSequence;
  metaRoot.hidden = false;
  metaRoot.innerHTML = [data.old, data.new].map((doc, i) => {
    const href = doc.view_url || doc.url;
    const linkText = doc.view_url
      ? (typeof t === 'function' ? t('card.open_uploaded_pdf') : 'Open uploaded PDF ↗')
      : (typeof t === 'function' ? t('card.open_source_doc') : 'Open source document ↗');
    const link = href ? `<a href="${escapeHtml(href)}" target="_blank" rel="noopener">${linkText}</a>` : '<small>No source URL supplied</small>';
    const documentName = doc.title ? `${doc.title} · ${doc.department}` : doc.department;
    const sideLabel = i ? (typeof t === 'function' ? t('card.updated_source') : 'Updated source') : (typeof t === 'function' ? t('card.previous_source') : 'Previous source');
    return `<div class="doc-chip"><span>${sideLabel}</span><strong>${escapeHtml(documentName)} · ${escapeHtml(doc.date)} · ${escapeHtml(doc.version)}</strong>${link}</div>`;
  }).join('');

  if (!data.changes.length) {
    changeRoot.innerHTML = `<div class="empty">${typeof t === 'function' ? t('status.no_changes') : 'No text changes were detected in the extracted passages.'}</div>`;
  } else {
    const summary = data.policy_summary
      ? `<section class="policy-summary"><div class="summary-eyebrow">${typeof t === 'function' ? t('summary.eyebrow') : 'AI POLICY SUMMARY'}</div><p>${escapeHtml(data.policy_summary)}</p></section>`
      : data.ai_status === 'pending'
        ? `<section class="analysis-unavailable"><strong>${typeof t === 'function' ? t('summary.ready_title') : 'Verified comparison ready'}</strong><p>${typeof t === 'function' ? t('summary.generating_text') : 'AI explanations are being generated separately. You can review the verified passages below while they run.'}</p></section>`
        : data.ai_status === 'not_requested'
          ? `<section class="analysis-unavailable"><strong>${typeof t === 'function' ? t('summary.ready_title') : 'Verified comparison ready'}</strong><p>${typeof t === 'function' ? t('summary.not_requested_text') : 'AI explanations are available on individual provisions. Use the Explain with AI button on any change card below.'}</p></section>`
        : `<section class="analysis-unavailable"><strong>${typeof t === 'function' ? t('ai.status_unavailable') : 'AI explanations unavailable'}</strong><p>${escapeHtml(data.ai_error || 'The application could not produce an AI explanation.')} ${typeof t === 'function' ? t('banner.ai_offline_detail') : 'The verified evidence comparison is still available below.'}</p></section>`;

    const important = (data.important_changes || []).map(group => {
      const affected = Array.isArray(group.who_may_be_affected) ? group.who_may_be_affected.join(', ') : group.who_may_be_affected;
      const prevLabel = typeof t === 'function' ? t('card.earlier_provision') : 'Previous provision';
      const updLabel = typeof t === 'function' ? t('card.updated_provision') : 'Updated provision';
      const evidence = group.changes.map(c => `<div class="group-evidence"><div class="evidence-change-label">${escapeHtml(c.type)}</div><div class="quotes">${quoteCard(c.previous, prevLabel)}${quoteCard(c.updated, updLabel)}</div></div>`).join('');
      const substanceLabel = group.substantive_change
        ? (typeof t === 'function' ? t('summary.substantive') : 'Substantive change')
        : (typeof t === 'function' ? t('summary.editorial') : 'Editorial or formatting change');
      const list = values => (values || []).map(value => `<li>${escapeHtml(value)}</li>`).join('');
      const impacts = group.substantive_change ? `<section class="impact-explanation">
        <h3>${typeof t === 'function' ? t('summary.practical_implications') : 'Possible practical implications'}</h3>
        <p>${escapeHtml(group.simple_explanation)}</p>
        ${group.potentially_affected_groups?.length
          ? `<p><b>${typeof t === 'function' ? t('summary.affected_groups') : 'Potentially affected groups:'}</b> ${escapeHtml(group.potentially_affected_groups.join(', '))}</p>`
          : `<p><b>${typeof t === 'function' ? t('summary.affected_groups') : 'Potentially affected groups:'}</b> ${typeof t === 'function' ? t('summary.groups_not_identified') : 'The cited passages do not identify this.'}</p>`}
        ${group.practical_implications?.length ? `<p><b>${typeof t === 'function' ? t('summary.practical_implications') : 'Possible implications'}:</b></p><ul>${list(group.practical_implications)}</ul>` : ''}
        ${group.what_changed_for_them?.length ? `<p><b>${typeof t === 'function' ? t('summary.what_may_change') : 'What may change for them:'}</b></p><ul>${list(group.what_changed_for_them)}</ul>` : ''}
        <p class="impact-limit"><b>${typeof t === 'function' ? t('summary.limitations') : 'Limitations:'}</b> ${escapeHtml(group.limitations)}</p>
        <small>${typeof t === 'function' ? t('summary.ai_disclaimer') : 'AI interpretation; source passages below are verified document text.'}</small>
      </section>` : '';

      return `<article class="important-change">
        <div class="change-head">
          <div class="change-title">
            <span class="badge ${group.priority.toLowerCase()}">${escapeHtml(typeof tPriority === 'function' ? tPriority(group.priority) : group.priority)}</span>
            ${escapeHtml(typeof tDynamic === 'function' ? tDynamic(group.title) : group.title)}
          </div>
          <span class="type">${escapeHtml(typeof tCategory === 'function' ? tCategory(group.change_category) : group.change_category || 'Other')} · ${escapeHtml(substanceLabel)}</span>
        </div>
        <div class="policy-fields">
          <p><b>${typeof t === 'function' ? t('summary.what_changed_q') : 'What changed?'}</b> ${escapeHtml(typeof tDynamic === 'function' ? tDynamic(group.what_changed) : group.what_changed)}</p>
          <p><b>${typeof t === 'function' ? t('summary.detected_type') : 'Detected change type'}</b> ${escapeHtml(group.change_type || 'Not classified')}</p>
          <p><b>${typeof t === 'function' ? t('summary.previous_rule') : 'Previous rule (AI explanation)'}</b> ${escapeHtml(group.previous_rule)}</p>
          <p><b>${typeof t === 'function' ? t('summary.updated_rule') : 'Updated rule (AI explanation)'}</b> ${escapeHtml(group.updated_rule)}</p>
          <p><b>${typeof t === 'function' ? t('summary.why_matters') : 'Why this may matter'}</b> ${escapeHtml(group.why_it_matters)}</p>
          ${affected ? `<p><b>${typeof t === 'function' ? t('summary.who_affected') : 'Who may be affected?'}</b> ${escapeHtml(affected)}</p>` : ''}
          <p class="priority-reason"><b>${typeof t === 'function' ? t('summary.priority_reason') : 'Priority reason:'}</b> ${escapeHtml(typeof tDynamic === 'function' ? tDynamic(group.priority_reason) : group.priority_reason)}</p>
          <p><b>${typeof t === 'function' ? t('summary.confidence') : 'Confidence:'}</b> ${Math.round(group.confidence * 100)}%</p>
          ${group.unknown_information ? `<p><b>${typeof t === 'function' ? t('summary.not_determined') : 'Not determined from these passages:'}</b> ${escapeHtml(group.unknown_information)}</p>` : ''}
        </div>
        ${impacts}
        ${evidence}
      </article>`;
    }).join('');

    // Enrich all detected changes with assigned policy category, audiences, and title
    const enrichedChanges = data.changes.map((c, index) => {
      const category = resolvePolicyCategory(c);
      const title = resolveChangeTitle(c, category);
      const audiences = resolveChangeAudiences(c);
      const priorityRaw = (c.priority || 'Low').toLowerCase();
      const normPriority = priorityRaw === 'high' ? 'High' : priorityRaw === 'medium' ? 'Medium' : 'Low';
      return {...c, index, category, title, audiences, normPriority};
    });

    // Dynamic priority counts calculated strictly from actual comparison results
    const highChanges = enrichedChanges.filter(c => c.normPriority === 'High');
    const mediumChanges = enrichedChanges.filter(c => c.normPriority === 'Medium');
    const lowChanges = enrichedChanges.filter(c => c.normPriority === 'Low');
    const highCount = highChanges.length;
    const mediumCount = mediumChanges.length;
    const lowCount = lowChanges.length;

    // Audience counts
    const bizCount = enrichedChanges.filter(c => c.audiences.includes('Businesses')).length;
    const stuCount = enrichedChanges.filter(c => c.audiences.includes('Students')).length;
    const citCount = enrichedChanges.filter(c => c.audiences.includes('General Citizens')).length;

    // Distinct categories present for filters
    const categoryCounts = {};
    for (const c of enrichedChanges) {
      categoryCounts[c.category] = (categoryCounts[c.category] || 0) + 1;
    }
    const categoriesList = Object.keys(categoryCounts).sort();

    // Render individual change card
    function renderChangeCard(c) {
      const earlierLink = c.previous ? sourceLink(c.previous) : null;
      const updatedLink = c.updated ? sourceLink(c.updated) : null;
      const audienceBadges = (c.audiences && c.audiences.length)
        ? c.audiences.map(a => `<span class="badge-audience">${escapeHtml(typeof tAudience === 'function' ? tAudience(a) : a)}</span>`).join('')
        : '';
      const substantiveLabel = c.substantive_change
        ? (typeof t === 'function' ? t('summary.substantive') : 'Substantive')
        : (typeof t === 'function' ? t('summary.editorial') : 'Editorial');
      const prevProvLabel = typeof t === 'function' ? t('card.earlier_provision') : 'Earlier Provision:';
      const updProvLabel = typeof t === 'function' ? t('card.updated_provision') : 'Updated Provision:';

      return `<article class="change-card" data-category="${escapeHtml(c.category)}" data-priority="${escapeHtml(c.normPriority)}" data-audiences="${escapeHtml(c.audiences.join(','))}" id="change-card-${c.index}">
        <div class="change-card-header">
          <div>
            <div class="change-card-badges">
              <span class="badge ${c.normPriority.toLowerCase()}">${escapeHtml(typeof tPriority === 'function' ? tPriority(c.normPriority) : c.normPriority)}</span>
              <span class="badge-category">${escapeHtml(typeof tCategory === 'function' ? tCategory(c.category) : c.category)}</span>
              ${audienceBadges}
              <span class="badge-classification">${escapeHtml(substantiveLabel)}</span>
              <span class="type">${escapeHtml(c.type)}</span>
            </div>
            <h3 class="change-card-title">${escapeHtml(typeof tDynamic === 'function' ? tDynamic(c.title) : c.title)}</h3>
          </div>
          ${c.repetition_count > 1 ? `<div class="repeat-note">${typeof t === 'function' ? t('card.repeated_in', {count: c.repetition_count}) : `Repeated in ${c.repetition_count} verified provisions`}</div>` : ''}
        </div>

        <div class="change-explanation-box">
          <div class="explanation-row">
            <span class="explanation-label">${typeof t === 'function' ? t('card.what_changed') : 'What Changed:'}</span>
            <span>${escapeHtml(typeof tDynamic === 'function' ? tDynamic(c.summary) : c.summary)}</span>
          </div>
          ${c.priority_reason ? `
            <div class="explanation-row rationale">
              <span class="explanation-label">${typeof t === 'function' ? t('card.review_rationale') : 'Review Rationale:'}</span>
              <span>${escapeHtml(typeof tDynamic === 'function' ? tDynamic(c.priority_reason) : c.priority_reason)}</span>
            </div>
          ` : ''}
        </div>

        <div class="explain-ai-row">
          <button type="button" class="btn-explain-ai" data-change-index="${c.index}" aria-label="Explain this provision with AI">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
            <span class="btn-explain-text">${typeof t === 'function' ? t('card.explain_ai') : 'Explain with AI'}</span>
          </button>
          <div class="explain-ai-output" id="explain-ai-output-${c.index}" hidden></div>
        </div>

        <div class="page-source-strip">
          <div class="source-ref-side earlier">
            <span class="source-ref-tag">${prevProvLabel}</span>
            ${c.previous ? `
              <span class="source-ref-page">${typeof t === 'function' ? t('card.page') : 'Page'} ${c.previous.page} (${escapeHtml(c.previous.document || (typeof t === 'function' ? t('card.previous_source') : 'Earlier Gazette'))})</span>
              <a class="source-ref-link" href="${escapeHtml(earlierLink)}" target="_blank" rel="noopener">${typeof t === 'function' ? t('card.view_source_pdf') : 'View Source PDF ↗'}</a>
            ` : `<span class="source-note">${typeof t === 'function' ? t('card.no_earlier_passage') : 'No corresponding passage in earlier version'}</span>`}
          </div>
          <div class="source-ref-divider">→</div>
          <div class="source-ref-side updated">
            <span class="source-ref-tag">${updProvLabel}</span>
            ${c.updated ? `
              <span class="source-ref-page">${typeof t === 'function' ? t('card.page') : 'Page'} ${c.updated.page} (${escapeHtml(c.updated.document || (typeof t === 'function' ? t('card.updated_source') : 'Updated Gazette'))})</span>
              <a class="source-ref-link" href="${escapeHtml(updatedLink)}" target="_blank" rel="noopener">${typeof t === 'function' ? t('card.view_source_pdf') : 'View Source PDF ↗'}</a>
            ` : `<span class="source-note">${typeof t === 'function' ? t('card.no_updated_passage') : 'No corresponding passage in updated version'}</span>`}
          </div>
        </div>

        <div class="quotes">
          ${quoteCard(c.previous, prevProvLabel)}
          ${quoteCard(c.updated, updProvLabel)}
        </div>
      </article>`;
    }

    // Render priority group container (only when it contains at least one change)
    function renderPriorityGroup(priority, title, desc, changes) {
      if (!changes.length) return '';
      const cardsHtml = changes.map(renderChangeCard).join('');
      const priLabel = typeof tPriority === 'function' ? tPriority(priority) : `${priority} Priority`;
      return `<section class="priority-group" id="priority-group-${priority.toLowerCase()}" data-group-priority="${priority}">
        <div class="priority-group-header">
          <div class="priority-group-header-left">
            <span class="badge ${priority.toLowerCase()}">${escapeHtml(priLabel)}</span>
            <h3 class="priority-group-title">${escapeHtml(title)}</h3>
            <span class="priority-group-count">${changes.length}</span>
          </div>
          <p class="priority-group-desc">${escapeHtml(desc)}</p>
        </div>
        <div class="priority-group-cards">
          ${cardsHtml}
        </div>
      </section>`;
    }

    const substantiveCount = enrichedChanges.filter(c => c.substantive_change).length;
    const editorialCount = enrichedChanges.length - substantiveCount;

    // Single unified Policy Changes & Priority Analysis section
    const unifiedSectionMarkup = `<section class="policy-changes-section" id="policy-changes-analysis">
      <div class="policy-changes-header">
        <div class="section-eyebrow">${typeof t === 'function' ? t('audit.eyebrow') : 'VERIFIED EVIDENCE AUDIT'}</div>
        <h2 class="section-title">${typeof t === 'function' ? t('audit.section_title') : 'Policy Changes & Priority Analysis'}</h2>
        <p class="section-desc">${enrichedChanges.length} verified changes detected: ${substantiveCount} classified for substantive review and ${editorialCount} classified as editorial or formatting. Every excerpt is verified against cited PDF page text.</p>
      </div>

      <div class="priority-summary-cards">
        <div class="priority-summary-card high" data-priority="High" role="button" tabindex="0" title="Click to filter by High Priority (click again to clear)">
          <div class="priority-card-top">
            <span class="badge high">${typeof tPriority === 'function' ? tPriority('High') : 'High Priority'}</span>
            <span class="priority-stat-count">${highCount}</span>
          </div>
          <div class="priority-card-name">${typeof t === 'function' ? t('audit.high_title') : 'Critical Policy Rules'}</div>
          <p class="priority-card-description">${typeof t === 'function' ? t('audit.high_desc') : 'Substantive modifications to eligibility, financial ceilings, deadlines, or binding obligations.'}</p>
          <div class="priority-card-action">${typeof t === 'function' ? t('audit.filter_by_priority', {priority: typeof tPriorityShort === 'function' ? tPriorityShort('High') : 'High'}) : 'Filter by High Priority ↓'}</div>
        </div>

        <div class="priority-summary-card medium" data-priority="Medium" role="button" tabindex="0" title="Click to filter by Medium Priority (click again to clear)">
          <div class="priority-card-top">
            <span class="badge medium">${typeof tPriority === 'function' ? tPriority('Medium') : 'Medium Priority'}</span>
            <span class="priority-stat-count">${mediumCount}</span>
          </div>
          <div class="priority-card-name">${typeof t === 'function' ? t('audit.medium_title') : 'Process & Implementation'}</div>
          <p class="priority-card-description">${typeof t === 'function' ? t('audit.medium_desc') : 'Procedural adjustments, application workflows, responsibilities, or document requirements.'}</p>
          <div class="priority-card-action">${typeof t === 'function' ? t('audit.filter_by_priority', {priority: typeof tPriorityShort === 'function' ? tPriorityShort('Medium') : 'Medium'}) : 'Filter by Medium Priority ↓'}</div>
        </div>

        <div class="priority-summary-card low" data-priority="Low" role="button" tabindex="0" title="Click to filter by Low Priority (click again to clear)">
          <div class="priority-card-top">
            <span class="badge low">${typeof tPriority === 'function' ? tPriority('Low') : 'Low Priority'}</span>
            <span class="priority-stat-count">${lowCount}</span>
          </div>
          <div class="priority-card-name">${typeof t === 'function' ? t('audit.low_title') : 'Editorial & Formatting'}</div>
          <p class="priority-card-description">${typeof t === 'function' ? t('audit.low_desc') : 'Non-substantive wording changes, formatting adjustments, or heading alignments.'}</p>
          <div class="priority-card-action">${typeof t === 'function' ? t('audit.filter_by_priority', {priority: typeof tPriorityShort === 'function' ? tPriorityShort('Low') : 'Low'}) : 'Filter by Low Priority ↓'}</div>
        </div>
      </div>

      <div class="filter-controls-card">
        <!-- Audience Filters -->
        <div class="filter-row audience-filter-row">
          <div class="filter-row-header">
            <span class="filter-toolbar-title">${typeof t === 'function' ? t('filter.audience_label') : 'Filter by Audience:'}</span>
            <span class="filter-sub-note">${typeof t === 'function' ? t('filter.audience_note') : 'Derived directly from cited policy text'}</span>
          </div>
          <div class="filter-chips audience-chips" role="tablist">
            <button type="button" class="btn-audience-filter active" data-audience="ALL">
              ${typeof t === 'function' ? t('filter.all_audiences') : 'All Audiences'} <span class="filter-count-badge" id="aud-count-all">(${enrichedChanges.length})</span>
            </button>
            <button type="button" class="btn-audience-filter" data-audience="Businesses">
              ${typeof t === 'function' ? t('filter.businesses') : 'Businesses'} <span class="filter-count-badge" id="aud-count-businesses">(${bizCount})</span>
            </button>
            <button type="button" class="btn-audience-filter" data-audience="Students">
              ${typeof t === 'function' ? t('filter.students') : 'Students'} <span class="filter-count-badge" id="aud-count-students">(${stuCount})</span>
            </button>
            <button type="button" class="btn-audience-filter" data-audience="General Citizens">
              ${typeof t === 'function' ? t('filter.citizens') : 'General Citizens'} <span class="filter-count-badge" id="aud-count-citizens">(${citCount})</span>
            </button>
          </div>
        </div>

        <!-- Category Filters -->
        <div class="filter-row category-filter-row">
          <div class="filter-row-header">
            <span class="filter-toolbar-title">${typeof t === 'function' ? t('filter.category_label') : 'Filter by Policy Category:'}</span>
            <div class="filter-status-group">
              <span class="filter-status-text" id="filter-status-summary">${typeof t === 'function' ? t('status.filter_showing_all', {count: enrichedChanges.length}) : `Showing all ${enrichedChanges.length} changes`}</span>
              <button type="button" class="btn-reset-filters" id="btn-reset-filters" hidden>${typeof t === 'function' ? t('filter.reset') : 'Reset all filters ✕'}</button>
            </div>
          </div>
          <div class="filter-chips category-chips" role="tablist">
            <button type="button" class="btn-category-filter active" data-category="ALL">
              ${typeof t === 'function' ? t('filter.all_categories') : 'All Categories'} <span class="filter-count-badge" id="cat-count-all">(${enrichedChanges.length})</span>
            </button>
            ${categoriesList.map(cat => `
              <button type="button" class="btn-category-filter" data-category="${escapeHtml(cat)}">
                ${escapeHtml(typeof tCategory === 'function' ? tCategory(cat) : cat)} <span class="filter-count-badge">(${categoryCounts[cat]})</span>
              </button>
            `).join('')}
          </div>
        </div>
      </div>

      <div class="priority-groups-container">
        ${renderPriorityGroup('High', typeof t === 'function' ? t('group.high_title') : 'Critical & Operational Policy Changes', typeof t === 'function' ? t('group.high_desc') : 'Substantive revisions impacting rights, limits, criteria, or obligations.', highChanges)}
        ${renderPriorityGroup('Medium', typeof t === 'function' ? t('group.medium_title') : 'Process & Procedural Adjustments', typeof t === 'function' ? t('group.medium_desc') : 'Operational changes to workflows, documentation, or governance.', mediumChanges)}
        ${renderPriorityGroup('Low', typeof t === 'function' ? t('group.low_title') : 'Editorial & Formatting Updates', typeof t === 'function' ? t('group.low_desc') : 'Non-substantive adjustments, numbering, heading alignments, or minor clarifications.', lowChanges)}
        <div class="no-filter-matches" id="no-filter-matches" hidden>${typeof t === 'function' ? t('filter.no_matches') : 'No policy changes match the combined filters. Try clearing or resetting filters.'}</div>
      </div>
    </section>`;

    changeRoot.innerHTML = `${summary}${important ? `<section class="important-section"><h2>${typeof t === 'function' ? t('summary.important_title') : 'Important policy changes'}</h2><div class="important-list">${important}</div></section>` : ''}${unifiedSectionMarkup}`;

    // Claim validation badge attachment
    const summaryText = changeRoot.querySelector('.policy-summary p');
    if (summaryText && data.policy_summary_status) summaryText.before(claimBadgeNode(data.policy_summary_status));
    for (const [index, card] of [...changeRoot.querySelectorAll('.important-change')].entries()) {
      const group = (data.important_changes || [])[index];
      const statuses = group?.claim_validation || {};
      const policyRows = [...card.querySelectorAll('.policy-fields p')];
      const attach = (row, status) => { if (row && status) row.append(' ', claimBadgeNode(status)); };
      attach(policyRows.find(row => row.querySelector('b')?.textContent.includes('?')), statuses.change_summary);
      attach(policyRows.find(row => row.querySelector('b')?.textContent.toLowerCase().includes('previous')), statuses.old_rule);
      attach(policyRows.find(row => row.querySelector('b')?.textContent.toLowerCase().includes('updated')), statuses.new_rule);
      attach(policyRows.find(row => row.querySelector('b')?.textContent.toLowerCase().includes('matter')), statuses.why_it_matters);
      attach(policyRows.find(row => row.querySelector('b')?.textContent.toLowerCase().includes('affected')), statuses.who_may_be_affected);
      attach(policyRows.find(row => row.querySelector('b')?.textContent.toLowerCase().includes('determined')), statuses.unknown_information);
      const impactSection = card.querySelector('.impact-explanation');
      if (impactSection) {
        const impactParagraph = impactSection.querySelector('p');
        attach(impactParagraph, statuses.simple_explanation);
        for (const [field, listIndex] of [['practical_implications', 0], ['what_changed_for_them', 1]]) {
          const listItems = impactSection.querySelectorAll('ul')[listIndex]?.querySelectorAll('li') || [];
          listItems.forEach((item, itemIndex) => attach(item, statuses[field]?.[itemIndex]));
        }
      }
    }

    // Unified interactive filtering state
    let activePriority = 'ALL';
    let activeAudience = 'ALL';
    let activeCategory = 'ALL';

    const priorityCards = changeRoot.querySelectorAll('.priority-summary-card');
    const audienceButtons = changeRoot.querySelectorAll('.btn-audience-filter');
    const categoryButtons = changeRoot.querySelectorAll('.btn-category-filter');
    const filterStatus = changeRoot.querySelector('#filter-status-summary');
    const resetBtn = changeRoot.querySelector('#btn-reset-filters');
    const noMatchesEl = changeRoot.querySelector('#no-filter-matches');
    const changeCards = changeRoot.querySelectorAll('.change-card');
    const priorityGroups = changeRoot.querySelectorAll('.priority-group');

    function updateFilters() {
      // 1. Update button & card active/selected visual states
      priorityCards.forEach(card => {
        const isSel = card.dataset.priority === activePriority;
        card.classList.toggle('selected', isSel);
        card.classList.toggle('active', isSel);
        card.setAttribute('aria-pressed', isSel ? 'true' : 'false');
        const actionEl = card.querySelector('.priority-card-action');
        if (actionEl) {
          actionEl.textContent = isSel
            ? (typeof t === 'function' ? t('audit.filter_active') : 'Active filter (click to clear) ✕')
            : (typeof t === 'function' ? t('audit.filter_by_priority', {priority: typeof tPriorityShort === 'function' ? tPriorityShort(card.dataset.priority) : card.dataset.priority}) : `Filter by ${card.dataset.priority} Priority ↓`);
        }
      });

      audienceButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.audience === activeAudience));
      categoryButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.category === activeCategory));

      // 2. Visible change cards (combined intersection of priority, category, audience)
      let visibleCount = 0;
      changeCards.forEach(card => {
        const cardAudiences = (card.dataset.audiences || '').split(',').filter(Boolean);
        const matchPri = activePriority === 'ALL' || card.dataset.priority === activePriority;
        const matchCat = activeCategory === 'ALL' || card.dataset.category === activeCategory;
        const matchAud = activeAudience === 'ALL' || cardAudiences.includes(activeAudience);
        const visible = matchPri && matchCat && matchAud;
        card.hidden = !visible;
        if (visible) visibleCount++;
      });

      // 3. Update priority groups in detailed area
      priorityGroups.forEach(group => {
        const groupPriority = group.dataset.groupPriority;
        const groupCards = group.querySelectorAll('.change-card');
        const groupVisible = [...groupCards].filter(c => !c.hidden).length;
        const matchActivePri = activePriority === 'ALL' || groupPriority === activePriority;
        group.hidden = !matchActivePri || groupVisible === 0;
        const countPill = group.querySelector('.priority-group-count');
        if (countPill) countPill.textContent = `${groupVisible}`;
      });

      // 4. Update Priority Summary Card counts
      ['High', 'Medium', 'Low'].forEach(p => {
        const count = enrichedChanges.filter(c =>
          c.normPriority === p &&
          (activeAudience === 'ALL' || c.audiences.includes(activeAudience)) &&
          (activeCategory === 'ALL' || c.category === activeCategory)
        ).length;
        const countEl = changeRoot.querySelector(`.priority-summary-card[data-priority="${p}"] .priority-stat-count`);
        if (countEl) countEl.textContent = count;
      });

      // 5. Update Audience button counts
      audienceButtons.forEach(btn => {
        const aud = btn.dataset.audience;
        const badge = btn.querySelector('.filter-count-badge');
        if (aud === 'ALL') {
          const allAudCount = enrichedChanges.filter(c =>
            (activePriority === 'ALL' || c.normPriority === activePriority) &&
            (activeCategory === 'ALL' || c.category === activeCategory)
          ).length;
          if (badge) badge.textContent = `(${allAudCount})`;
        } else {
          const count = enrichedChanges.filter(c =>
            c.audiences.includes(aud) &&
            (activePriority === 'ALL' || c.normPriority === activePriority) &&
            (activeCategory === 'ALL' || c.category === activeCategory)
          ).length;
          if (badge) badge.textContent = `(${count})`;
        }
      });

      // 6. Update Category button counts
      categoryButtons.forEach(btn => {
        const cat = btn.dataset.category;
        const badge = btn.querySelector('.filter-count-badge');
        if (cat === 'ALL') {
          const allCatCount = enrichedChanges.filter(c =>
            (activePriority === 'ALL' || c.normPriority === activePriority) &&
            (activeAudience === 'ALL' || c.audiences.includes(activeAudience))
          ).length;
          if (badge) badge.textContent = `(${allCatCount})`;
          btn.hidden = false;
        } else {
          const count = enrichedChanges.filter(c =>
            c.category === cat &&
            (activePriority === 'ALL' || c.normPriority === activePriority) &&
            (activeAudience === 'ALL' || c.audiences.includes(activeAudience))
          ).length;
          if (badge) badge.textContent = `(${count})`;
          btn.hidden = count === 0 && activeCategory !== cat;
        }
      });

      // 7. Status text & reset button
      const hasActiveFilters = (activePriority !== 'ALL' || activeCategory !== 'ALL' || activeAudience !== 'ALL');
      if (resetBtn) resetBtn.hidden = !hasActiveFilters;

      if (filterStatus) {
        if (!hasActiveFilters) {
          filterStatus.textContent = typeof t === 'function' ? t('status.filter_showing_all', {count: enrichedChanges.length}) : `Showing all ${enrichedChanges.length} changes`;
        } else {
          const parts = [];
          if (activePriority !== 'ALL') parts.push(typeof tPriority === 'function' ? tPriority(activePriority) : `${activePriority} Priority`);
          if (activeAudience !== 'ALL') parts.push(typeof tAudience === 'function' ? tAudience(activeAudience) : activeAudience);
          if (activeCategory !== 'ALL') parts.push(typeof tCategory === 'function' ? tCategory(activeCategory) : activeCategory);
          filterStatus.textContent = typeof t === 'function'
            ? t('status.filter_showing_subset', {visible: visibleCount, total: enrichedChanges.length, filters: parts.join(' · ')})
            : `Showing ${visibleCount} of ${enrichedChanges.length} changes (${parts.join(' · ')})`;
        }
      }

      if (noMatchesEl) noMatchesEl.hidden = visibleCount > 0;
    }

    // Priority card click handlers
    priorityCards.forEach(card => {
      card.addEventListener('click', () => {
        const clickedPriority = card.dataset.priority;
        activePriority = (activePriority === clickedPriority) ? 'ALL' : clickedPriority;
        updateFilters();
        if (activePriority !== 'ALL') {
          const targetGroup = changeRoot.querySelector(`#priority-group-${activePriority.toLowerCase()}`);
          if (targetGroup && !targetGroup.hidden) {
            targetGroup.scrollIntoView({behavior: 'smooth', block: 'start'});
          }
        }
      });
    });

    // Audience button click handlers
    audienceButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const clickedAudience = btn.dataset.audience;
        activeAudience = (activeAudience === clickedAudience) ? 'ALL' : clickedAudience;
        updateFilters();
      });
    });

    // Category button click handlers
    categoryButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const clickedCategory = btn.dataset.category;
        activeCategory = (activeCategory === clickedCategory) ? 'ALL' : clickedCategory;
        updateFilters();
      });
    });

    // Reset all filters button handler
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        activePriority = 'ALL';
        activeAudience = 'ALL';
        activeCategory = 'ALL';
        updateFilters();
      });
    }

    // Attach Explain with AI handlers to individual change cards
    changeRoot.querySelectorAll('.btn-explain-ai').forEach(btn => {
      btn.addEventListener('click', async () => {
        const changeIndex = parseInt(btn.dataset.changeIndex, 10);
        if (isNaN(changeIndex) || !visibleComparison?.comparison_id) return;
        if (btn.disabled) return;

        const outputContainer = changeRoot.querySelector(`#explain-ai-output-${changeIndex}`);
        const textSpan = btn.querySelector('.btn-explain-text');
        const originalText = textSpan ? textSpan.textContent : 'Explain with AI';

        // Set loading state and prevent duplicate clicks
        btn.disabled = true;
        if (textSpan) textSpan.textContent = typeof t === 'function' ? t('card.explaining_ai') : 'Explaining with AI…';
        if (outputContainer) {
          outputContainer.hidden = false;
          outputContainer.className = 'explain-ai-output loading';
          outputContainer.innerHTML = `<span class="explain-spinner" aria-hidden="true"></span> ${typeof t === 'function' ? t('card.explaining_ai') : 'Explaining with AI…'}`;
        }

        try {
          const res = await fetch('/api/explain-provision', {
            method: 'POST',
            headers: {'Content-Type': 'application/json', 'Accept': 'application/json'},
            body: JSON.stringify({
              comparison_id: visibleComparison.comparison_id,
              change_index: changeIndex,
              language: typeof getLanguage === 'function' ? getLanguage() : 'en'
            }),
            signal: AbortSignal.timeout(30000)
          });
          const data = await res.json();
          if (!res.ok || !data.ok) {
            throw new Error(data.error || 'Failed to explain this provision.');
          }

          if (outputContainer) {
            outputContainer.className = 'explain-ai-output success';
            const badge = claimBadge(data.claim_status);
            const label = typeof t === 'function' ? t('card.ai_explanation_label') : 'AI Simple Explanation (Interpretation, not official text):';
            outputContainer.innerHTML = `
              <div class="explain-header">
                <strong>${escapeHtml(label)}</strong>
                ${badge}
              </div>
              <p class="explain-text">${escapeHtml(data.explanation)}</p>
            `;
          }
          if (textSpan) textSpan.textContent = typeof t === 'function' ? t('card.explain_ai') : 'Explain with AI';
          btn.disabled = false;
        } catch (err) {
          if (outputContainer) {
            outputContainer.className = 'explain-ai-output error';
            outputContainer.innerHTML = `
              <span class="explain-error-msg">${escapeHtml(err.message)}</span>
            `;
          }
          if (textSpan) textSpan.textContent = originalText;
          btn.disabled = false;
        }
      });
    });
  }

  const extraMsg = data.ai_status === 'pending'
    ? (typeof t === 'function' ? ' ' + t('summary.generating_text') : ' AI explanations are processing separately.')
    : '';
  statusEl.textContent = typeof t === 'function'
    ? t('status.showing_changes', {count: data.changes.length, plural: data.changes.length === 1 ? '' : 's', extra: extraMsg})
    : `${data.changes.length} detected change${data.changes.length === 1 ? '' : 's'} shown. Each displayed excerpt was checked against its cited PDF page.${extraMsg}`;

  console.info(`CivicLens timing stage=browser_result_rendering elapsed_ms=${(performance.now() - renderStarted).toFixed(1)} changes=${data.changes.length}`);
  if (data.ai_status === 'pending' && data.analysis_id) pollAnalysis(data.analysis_id, pollSequence);
}


if (aiHealthButton) {
  aiHealthButton.addEventListener('click', async () => {
    aiHealthButton.disabled = true;
    if (aiHealthResult) aiHealthResult.textContent = 'Testing AI connection (up to 15 seconds)…';
    try {
      const response = await fetch(new URL('/api/ai/health', window.location.href), {method: 'POST', headers: {'Accept': 'application/json'}, signal: AbortSignal.timeout(20000)});
      const result = await response.json();
      if (!response.ok || !result.ok) {
        const status = result.http_status ? ` HTTP ${result.http_status}.` : '';
        const detail = result.detail ? ` ${result.detail}` : '';
        throw new Error(`${result.error || 'AI connection check failed.'}${status}${detail}`);
      }
      const providerName = result.provider ? (result.provider.charAt(0).toUpperCase() + result.provider.slice(1)) : 'AI';
      if (aiHealthResult) aiHealthResult.textContent = `${providerName} connection successful — Model: ${result.model}.`;
    } catch (err) {
      if (aiHealthResult) {
        aiHealthResult.textContent = err instanceof TypeError
          ? `Could not reach the CivicLens server at ${window.location.origin}. Start python app.py and open http://127.0.0.1:8000.`
          : err.message;
      }
    } finally { aiHealthButton.disabled = false; }
  });
}

async function pollAnalysis(analysisId, sequence, startedAt = performance.now(), delay = 1200) {
  if (performance.now() - startedAt >= MAX_ANALYSIS_POLL_MS) {
    if (sequence === analysisPollSequence && visibleComparison) {
      renderComparison({...visibleComparison, ai_status: 'unavailable', ai_error: 'AI explanations took longer than expected. The verified comparison is ready; try the AI connection check and compare again later.', analysis_id: null});
    }
    return;
  }
  await new Promise(resolve => setTimeout(resolve, delay));
  if (sequence !== analysisPollSequence || !visibleComparison) return;
  const controller = new AbortController();
  const requestTimeout = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(`/api/analysis/${encodeURIComponent(analysisId)}`, {signal: controller.signal});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'AI analysis status could not be loaded.');
    if (sequence !== analysisPollSequence) return;
    if (result.ai_status === 'pending') return pollAnalysis(analysisId, sequence, startedAt, Math.min(5000, Math.round(delay * 1.5)));
    renderComparison({...visibleComparison, ...result, analysis_id: null});
  } catch (err) {
    if (sequence === analysisPollSequence && visibleComparison) {
      const message = controller.signal.aborted ? 'Checking AI analysis timed out. The verified comparison is ready; check the AI connection and try again later.' : err.message;
      renderComparison({...visibleComparison, ai_status: 'unavailable', ai_error: message, analysis_id: null});
    }
  } finally { clearTimeout(requestTimeout); }
}

uploadForm.addEventListener('submit', async event => {
  event.preventDefault();
  uploadButton.disabled = true; compareButton.disabled = true;
  statusEl.textContent = typeof t === 'function' ? t('status.uploading') : 'Checking both PDFs and extracting page text…';
  changeRoot.setAttribute('aria-busy', 'true');
  try {
    const response = await fetch('/api/compare-upload', {method: 'POST', body: new FormData(uploadForm)});
    const data = await response.json();
    if (!response.ok) {
      const msg = data.error || 'The PDF comparison failed.';
      statusEl.textContent = msg;
      const failedTitle = typeof t === 'function' ? t('status.upload_failed') : 'Upload failed';
      changeRoot.innerHTML = `<div class="upload-error-card" role="alert">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
        <div>
          <strong>${failedTitle}</strong>
          <p>${escapeHtml(msg)}</p>
        </div>
      </div>`;
      return;
    }
    renderComparison(data);
  } catch (err) {
    statusEl.textContent = err.message;
    const failedTitle = typeof t === 'function' ? t('status.upload_failed') : 'Upload failed';
    changeRoot.innerHTML = `<div class="upload-error-card" role="alert">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
      <div>
        <strong>${failedTitle}</strong>
        <p>${escapeHtml(err.message)}</p>
      </div>
    </div>`;
  }
  finally { uploadButton.disabled = false; compareButton.disabled = false; changeRoot.setAttribute('aria-busy', 'false'); }
});

async function init() {
  try {
    const res = await fetch('/api/samples'); if (!res.ok) throw new Error('Could not load sample documents.');
    samples = await res.json(); oldSelect.innerHTML = samples.map(option).join(''); newSelect.innerHTML = samples.map(option).join('');
    oldSelect.value = samples[0].id; newSelect.value = samples[1].id; await loadComparison();
  } catch (err) { statusEl.textContent = err.message; }
}

compareButton.addEventListener('click', loadComparison);
init();

// Re-render when global language changes
window.addEventListener('civiclens:languagechange', () => {
  if (visibleComparison) {
    renderComparison(visibleComparison);
  }
});

// Dashboard UI interactions
const sidebarToggle = document.querySelector('#sidebar-toggle');
const sidebar = document.querySelector('#sidebar');
if (sidebarToggle && sidebar) {
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
  });
  document.addEventListener('click', (event) => {
    if (sidebar.classList.contains('open') && !sidebar.contains(event.target) && !sidebarToggle.contains(event.target)) {
      sidebar.classList.remove('open');
    }
  });
}

document.querySelectorAll('.sidebar-nav .nav-item').forEach(item => {
  item.addEventListener('click', () => {
    document.querySelectorAll('.sidebar-nav .nav-item').forEach(el => el.classList.remove('active'));
    item.classList.add('active');
    if (sidebar && sidebar.classList.contains('open')) {
      sidebar.classList.remove('open');
    }
  });
});

document.querySelectorAll('.file-input').forEach(input => {
  input.addEventListener('change', () => {
    const box = input.closest('.dropzone-box');
    const prompt = box?.querySelector('.dropzone-prompt');
    if (prompt && input.files && input.files[0]) {
      prompt.textContent = input.files[0].name;
      prompt.classList.add('has-file');
    }
  });
});
