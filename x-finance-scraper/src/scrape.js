/**
 * Pobiera posty z X (Twitter) na temat firmy/hasla powiazanego z finansami
 * z ostatnich N godzin (domyslnie 72h), dzialajac przez przegladarke
 * (Playwright + prawdziwy DOM x.com), a NIE przez oficjalne API.
 *
 * Wymaga wczesniejszego jednorazowego logowania: `npm run login`
 * (zapisuje sesje do auth/storageState.json).
 *
 * Uzycie:
 *   node src/scrape.js
 *   node src/scrape.js --query "Google" --hours 72
 *   node src/scrape.js --query "Google" --keywords "finance,earnings,stock" --lang en
 *   node src/scrape.js --headful          (pokaz okno przegladarki, jesli jest ekran)
 *   node src/scrape.js --strict           (odrzucaj posty bez slowa kluczowego z finansow)
 */
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const ROOT = path.join(__dirname, '..');
const STORAGE_STATE_PATH = path.join(ROOT, 'auth', 'storageState.json');
const OUTPUT_DIR = path.join(ROOT, 'output');
const DEFAULT_CONFIG_PATH = path.join(ROOT, 'config', 'default.json');

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) {
      args[key] = true;
    } else {
      args[key] = next;
      i++;
    }
  }
  return args;
}

function loadConfig(args) {
  const base = JSON.parse(fs.readFileSync(DEFAULT_CONFIG_PATH, 'utf8'));
  const cfg = { ...base };

  if (args.query) cfg.query = String(args.query);
  if (args.keywords) cfg.keywords = String(args.keywords).split(',').map((s) => s.trim()).filter(Boolean);
  if (args.hours) cfg.hoursBack = Number(args.hours);
  if (args.lang) cfg.lang = String(args.lang);
  if (args['max-tweets']) cfg.maxTweets = Number(args['max-tweets']);
  if (args['max-scrolls']) cfg.maxScrolls = Number(args['max-scrolls']);
  if (args.strict) cfg.strictKeywordFilter = true;
  if (args['no-exclude-replies']) cfg.excludeReplies = false;

  cfg.headless = !args.headful;
  return cfg;
}

function buildSearchQuery(cfg) {
  const parts = [];
  parts.push(cfg.query);
  if (Array.isArray(cfg.keywords) && cfg.keywords.length > 0) {
    const orClause = cfg.keywords.map((k) => (k.includes(' ') ? `"${k}"` : k)).join(' OR ');
    parts.push(`(${orClause})`);
  }
  if (cfg.excludeReplies) parts.push('-filter:replies');
  if (cfg.lang) parts.push(`lang:${cfg.lang}`);
  return parts.join(' ');
}

async function extractTweets(page) {
  return page.$$eval('article[data-testid="tweet"]', (articles) => {
    return articles.map((article) => {
      const timeEl = article.querySelector('time');
      const datetime = timeEl ? timeEl.getAttribute('datetime') : null;
      const linkEl = timeEl ? timeEl.closest('a') : null;
      const url = linkEl ? linkEl.href : null;

      const textEl = article.querySelector('[data-testid="tweetText"]');
      const text = textEl ? textEl.innerText.replace(/\s+/g, ' ').trim() : '';

      let handle = null;
      let displayName = null;
      const userNameBlock = article.querySelector('[data-testid="User-Name"]');
      if (userNameBlock) {
        const links = userNameBlock.querySelectorAll('a[href^="/"]');
        for (const a of links) {
          const href = a.getAttribute('href') || '';
          if (/^\/[A-Za-z0-9_]+$/.test(href)) {
            handle = href.slice(1);
            break;
          }
        }
        displayName = (userNameBlock.innerText.split('\n')[0] || '').trim() || null;
      }

      const metrics = {};
      ['reply', 'retweet', 'like'].forEach((key) => {
        const el = article.querySelector(`[data-testid="${key}"]`);
        metrics[key] = el ? (el.getAttribute('aria-label') || el.textContent || '').trim() : null;
      });

      return { url, datetime, text, handle, displayName, metrics };
    });
  });
}

function toCsv(rows) {
  const headers = ['datetime', 'handle', 'displayName', 'text', 'url', 'reply', 'retweet', 'like'];
  const escape = (v) => {
    const s = v === null || v === undefined ? '' : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [headers.join(',')];
  for (const r of rows) {
    lines.push(
      headers
        .map((h) => (h === 'reply' || h === 'retweet' || h === 'like' ? r.metrics?.[h] : r[h]))
        .map(escape)
        .join(',')
    );
  }
  return lines.join('\n');
}

function matchesKeywords(text, cfg) {
  const lower = text.toLowerCase();
  const queryHit = lower.includes(cfg.query.toLowerCase());
  if (!cfg.strictKeywordFilter) return true;
  const kwHit = cfg.keywords.some((k) => lower.includes(k.toLowerCase()));
  return queryHit && kwHit;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const cfg = loadConfig(args);

  if (!fs.existsSync(STORAGE_STATE_PATH)) {
    console.error(
      `Brak pliku sesji: ${STORAGE_STATE_PATH}\n` +
        'Najpierw zaloguj sie jednorazowo lokalnie: npm run login'
    );
    process.exit(1);
  }
  if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  const cutoffMs = Date.now() - cfg.hoursBack * 60 * 60 * 1000;
  const searchQuery = buildSearchQuery(cfg);
  const url = `https://x.com/search?q=${encodeURIComponent(searchQuery)}&src=typed_query&f=live`;

  console.log(`Zapytanie wyszukiwania: ${searchQuery}`);
  console.log(`Okno czasowe: ostatnie ${cfg.hoursBack}h (od ${new Date(cutoffMs).toISOString()})`);

  const browser = await chromium.launch({ headless: cfg.headless });
  const context = await browser.newContext({
    storageState: STORAGE_STATE_PATH,
    viewport: { width: 1280, height: 1000 },
    userAgent:
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
  });
  const page = await context.newPage();

  await page.goto(url, { waitUntil: 'domcontentloaded' });

  if (page.url().includes('/login') || page.url().includes('/i/flow/login')) {
    console.error('Sesja wygasla lub jest nieprawidlowa - zaloguj sie ponownie: npm run login');
    await browser.close();
    process.exit(1);
  }

  try {
    await page.waitForSelector('article[data-testid="tweet"]', { timeout: 20000 });
  } catch {
    console.warn('Nie znaleziono zadnych postow dla tego zapytania (lub strona nie zaladowala sie poprawnie).');
  }

  const seen = new Map();
  let staleScrolls = 0;

  for (let i = 0; i < cfg.maxScrolls; i++) {
    const batch = await extractTweets(page);

    for (const t of batch) {
      if (!t.url || !t.datetime) continue;
      if (seen.has(t.url)) continue;
      if (!matchesKeywords(t.text, cfg)) continue;
      seen.set(t.url, t);
    }

    const times = batch.map((t) => t.datetime).filter(Boolean).map((d) => new Date(d).getTime());
    const oldestInBatch = times.length ? Math.min(...times) : null;

    if (oldestInBatch !== null && oldestInBatch < cutoffMs) {
      staleScrolls++;
    } else {
      staleScrolls = 0;
    }

    console.log(`Scroll ${i + 1}/${cfg.maxScrolls} - zebrano lacznie: ${seen.size}`);

    if (staleScrolls >= 3) {
      console.log('Kolejne posty sa juz starsze niz zadane okno czasowe - konczenie.');
      break;
    }
    if (seen.size >= cfg.maxTweets) {
      console.log('Osiagnieto limit maxTweets - konczenie.');
      break;
    }

    await page.mouse.wheel(0, 2200);
    await page.waitForTimeout(1200 + Math.random() * 1300);
  }

  await browser.close();

  const results = [...seen.values()]
    .filter((t) => new Date(t.datetime).getTime() >= cutoffMs)
    .sort((a, b) => new Date(b.datetime) - new Date(a.datetime));

  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const jsonPath = path.join(OUTPUT_DIR, `x-posts-${stamp}.json`);
  const csvPath = path.join(OUTPUT_DIR, `x-posts-${stamp}.csv`);

  fs.writeFileSync(jsonPath, JSON.stringify(results, null, 2), 'utf8');
  fs.writeFileSync(csvPath, toCsv(results), 'utf8');

  console.log(`\nZnaleziono ${results.length} postow z ostatnich ${cfg.hoursBack}h.`);
  console.log(`Zapisano do:\n  ${jsonPath}\n  ${csvPath}`);
}

main().catch((err) => {
  console.error('Blad scrapera:', err);
  process.exit(1);
});
