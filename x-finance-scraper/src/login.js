/**
 * Jednorazowe logowanie do X (Twitter) w prawdziwej, widocznej przegladarce.
 * Uruchom TEN skrypt na wlasnym komputerze (potrzebny ekran) - nie w kontenerze bez GUI.
 *
 * Uzycie:
 *   npm run login
 *
 * Po zalogowaniu sie recznie w otwartym oknie przegladarki, wroc do terminala
 * i nacisnij Enter - sesja (ciasteczka) zostanie zapisana do auth/storageState.json
 * i bedzie uzywana przez src/scrape.js, wiec logowanie wystarczy zrobic raz
 * (dopoki sesja nie wygasnie).
 */
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');
const readline = require('readline');

const AUTH_DIR = path.join(__dirname, '..', 'auth');
const STORAGE_STATE_PATH = path.join(AUTH_DIR, 'storageState.json');

function waitForEnter(message) {
  return new Promise((resolve) => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.question(message, () => {
      rl.close();
      resolve();
    });
  });
}

(async () => {
  if (!fs.existsSync(AUTH_DIR)) {
    fs.mkdirSync(AUTH_DIR, { recursive: true });
  }

  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
  });
  const page = await context.newPage();

  await page.goto('https://x.com/login', { waitUntil: 'domcontentloaded' });

  console.log('\nOtworzyla sie przegladarka - zaloguj sie recznie na swoje konto X.');
  console.log('Jesli pojawi sie prosba o kod 2FA, tez ja uzupelnij.');
  await waitForEnter('\nGdy zobaczysz swoj glowny feed (strona home), wroc tutaj i nacisnij Enter...\n');

  await context.storageState({ path: STORAGE_STATE_PATH });
  console.log(`Zapisano sesje logowania do: ${STORAGE_STATE_PATH}`);
  console.log('Mozesz teraz uruchamiac: npm run scrape');

  await browser.close();
})().catch((err) => {
  console.error('Blad podczas logowania:', err);
  process.exit(1);
});
