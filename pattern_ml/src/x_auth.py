"""Logowanie do X (Twitter) przez Playwright, z ponownym użyciem sesji (cookies).

Dane logowania NIGDY nie są zapisywane w kodzie ani w repo - pobierane są
wyłącznie ze zmiennych środowiskowych X_USERNAME / X_PASSWORD. Po udanym
logowaniu sesja (cookies) jest zapisywana lokalnie w .auth/x_state.json,
żeby kolejne uruchomienia nie musiały logować się od nowa (logowanie z nowego
adresu IP/kontenera często wywołuje dodatkową weryfikację X, więc im rzadziej,
tym lepiej).
"""

import os
import time
from pathlib import Path

from playwright.sync_api import (
    BrowserContext,
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)

AUTH_DIR = Path(__file__).resolve().parent.parent / ".auth"
STORAGE_STATE_PATH = AUTH_DIR / "x_state.json"
LOGIN_SCREENSHOT_PATH = AUTH_DIR / "login_failure.png"

HOME_MARKER = 'a[data-testid="AppTabBar_Home_Link"]'


class LoginError(RuntimeError):
    pass


def is_session_valid(page: Page) -> bool:
    """Sprawdza, czy zapisana sesja nadal jest zalogowana (bez próby logowania)."""
    try:
        page.goto("https://x.com/home", wait_until="domcontentloaded", timeout=20000)
        page.wait_for_selector(HOME_MARKER, timeout=8000)
        return True
    except (PlaywrightTimeoutError, PlaywrightError):
        return False


def login(page: Page) -> None:
    """Loguje się do X danymi z X_USERNAME / X_PASSWORD. Rzuca LoginError, jeśli
    napotka krok, którego nie da się rozwiązać automatycznie (captcha, 2FA,
    weryfikacja telefonu)."""
    username = os.environ.get("X_USERNAME")
    password = os.environ.get("X_PASSWORD")
    if not username or not password:
        raise LoginError(
            "Brak danych logowania: ustaw zmienne środowiskowe X_USERNAME i X_PASSWORD."
        )

    try:
        page.goto("https://x.com/i/flow/login", wait_until="domcontentloaded")
        _fill_and_next(page, username)

        # X czasem prosi o ponowne potwierdzenie username/telefonu (krok anty-botowy)
        if _step_present(page, 'input[data-testid="ocfEnterTextTextInput"]'):
            _fill_and_next(page, username)

        page.wait_for_selector('input[name="password"]', timeout=15000)
        page.fill('input[name="password"]', password)
        page.click('[data-testid="LoginForm_Login_Button"]')

        page.wait_for_selector(HOME_MARKER, timeout=20000)
    except PlaywrightTimeoutError as exc:
        AUTH_DIR.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(LOGIN_SCREENSHOT_PATH))
        raise LoginError(
            "Logowanie nie powiodło się automatycznie - X prawdopodobnie wymaga "
            "dodatkowej weryfikacji (captcha, kod SMS, potwierdzenie e-mail). "
            f"Zrzut ekranu zapisano w {LOGIN_SCREENSHOT_PATH}. Rozważ zalogowanie się "
            "ręcznie w przeglądarce i wgranie pliku sesji (storage_state) zamiast "
            "logowania automatycznego."
        ) from exc
    except PlaywrightError as exc:
        raise LoginError(f"Nie udało się połączyć z X: {exc}") from exc


def _fill_and_next(page: Page, value: str) -> None:
    field = page.locator('input[autocomplete="username"], input[name="text"]').first
    field.wait_for(timeout=15000)
    field.fill(value)
    page.get_by_role("button", name="Next").click()
    time.sleep(1.5)


def _step_present(page: Page, selector: str) -> bool:
    try:
        page.wait_for_selector(selector, timeout=4000)
        return True
    except PlaywrightTimeoutError:
        return False


def ensure_logged_in(context: BrowserContext, page: Page) -> None:
    """Próbuje wznowić zapisaną sesję; jeśli nieważna lub jej brak, loguje się od nowa."""
    if STORAGE_STATE_PATH.exists() and is_session_valid(page):
        return

    login(page)
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    context.storage_state(path=str(STORAGE_STATE_PATH))
