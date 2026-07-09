"""
Jednorazowe logowanie do X (Twitter) w prawdziwej, widocznej przegladarce.
Uruchom TEN skrypt na wlasnym komputerze (potrzebny ekran) - nie w kontenerze bez GUI.

Uzycie:
    python login.py

Po zalogowaniu sie recznie w otwartym oknie przegladarki, wroc do terminala
i nacisnij Enter - sesja (ciasteczka) zostanie zapisana do auth/storage_state.json
i bedzie uzywana przez scrape.py, wiec logowanie wystarczy zrobic raz
(dopoki sesja nie wygasnie).
"""
import os

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.abspath(__file__))
AUTH_DIR = os.path.join(ROOT, "auth")
STORAGE_STATE_PATH = os.path.join(AUTH_DIR, "storage_state.json")


def main():
    os.makedirs(AUTH_DIR, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        page.goto("https://x.com/login", wait_until="domcontentloaded")

        print("\nOtworzyla sie przegladarka - zaloguj sie recznie na swoje konto X.")
        print("Jesli pojawi sie prosba o kod 2FA, tez ja uzupelnij.")
        input("\nGdy zobaczysz swoj glowny feed (strona home), wroc tutaj i nacisnij Enter...\n")

        context.storage_state(path=STORAGE_STATE_PATH)
        print(f"Zapisano sesje logowania do: {STORAGE_STATE_PATH}")
        print("Mozesz teraz uruchamiac: python scrape.py")

        browser.close()


if __name__ == "__main__":
    main()
