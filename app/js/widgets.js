// Chapter widgets are compact permalinks into the full lab. `data-widget`
// selects the opening view; `data-state` may contain URL-hash key/value pairs.
import { encodeHash, makeDefaults, decodeHash } from './state.js';

const VIEW_BY_WIDGET = { curve: 'sheet', sheet: 'sheet', density: 'density', spectra: 'spectra', pdf: 'pdf', gradient: 'gradient', published: 'published' };
for (const host of document.querySelectorAll('.qd-widget')) {
  const state = { ...makeDefaults(), ...decodeHash(host.dataset.state || '') };
  state.view = VIEW_BY_WIDGET[host.dataset.widget] || state.view;
  const card = document.createElement('section'); card.className = 'lab-widget-card';
  const text = document.createElement('div');
  const eyebrow = document.createElement('span'); eyebrow.textContent = 'Interactive laboratory'; eyebrow.className = 'lab-widget-kicker'; text.append(eyebrow);
  const title = document.createElement('strong'); title.textContent = ({sheet:'Follow the phase-space sheet',density:'Inspect exact density deposition',spectra:'Compare power spectra',pdf:'Compare density PDFs',gradient:'Check differentiability',published:'Replay the published result'})[state.view] || 'Explore the sheet'; text.append(title);
  const note = document.createElement('span'); note.textContent = 'Open this experiment with the same controls and URL state.'; text.append(note);
  const link = document.createElement('a'); link.href = `../app/index.html#${encodeHash(state)}`; link.textContent = 'Open experiment ↗';
  card.append(text, link); host.replaceChildren(card);
}
