"""Deterministic sandbox tools for tool-use tasks.

`example_args` on each tool tells the offline mock model how to call it;
real models pick their own args from the prompt.
"""

import ast as _ast
import operator as _op

_WEATHER = {
    "Tokyo": "15°C, partly cloudy",
    "London": "11°C, light rain",
    "Cairo": "31°C, clear skies",
}

_FX_TO_USD = {"USD": 1.0, "JPY": 1 / 157.0, "EUR": 1.08, "INR": 1 / 83.5}


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return _WEATHER.get(city, f"No weather data available for {city}")


get_weather.example_args = {"city": "Tokyo"}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies (e.g. 100 USD to JPY)."""
    try:
        usd = amount * _FX_TO_USD[from_currency.upper()]
        result = usd / _FX_TO_USD[to_currency.upper()]
    except KeyError as e:
        return f"ERROR: unsupported currency {e}"
    return f"{amount:g} {from_currency.upper()} = {result:,.0f} {to_currency.upper()}"


convert_currency.example_args = {"amount": 100, "from_currency": "USD", "to_currency": "JPY"}


def get_stock_price(ticker: str) -> str:
    """Get the latest stock price for a ticker symbol."""
    prices = {"ACME": 172.40, "GLOBEX": 54.10, "INITECH": 12.75}
    if ticker.upper() in prices:
        return f"{ticker.upper()}: ${prices[ticker.upper()]:.2f}"
    return f"ERROR: unknown ticker {ticker}"


get_stock_price.example_args = {"ticker": "ACME"}


_CALC_OPS = {
    _ast.Add: _op.add,
    _ast.Sub: _op.sub,
    _ast.Mult: _op.mul,
    _ast.Div: _op.truediv,
    _ast.Mod: _op.mod,
    _ast.Pow: _op.pow,
    _ast.USub: _op.neg,
}


def _calc_eval(node):
    if isinstance(node, _ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, _ast.BinOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_calc_eval(node.left), _calc_eval(node.right))
    if isinstance(node, _ast.UnaryOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_calc_eval(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (numbers and + - * / % ** only)."""
    try:
        result = _calc_eval(_ast.parse(expression, mode="eval").body)
    except Exception as e:
        return f"ERROR: {e}"
    return f"{expression} = {result:g}"


calculator.example_args = {"expression": "84.50 * 0.17"}


_TRANSLATIONS = {
    ("hello", "spanish"): "hola",
    ("thank you", "french"): "merci",
    ("goodbye", "german"): "auf wiedersehen",
}


def translate(text: str, to_lang: str) -> str:
    """Translate a short phrase into another language."""
    key = (text.strip().lower(), to_lang.strip().lower())
    if key in _TRANSLATIONS:
        return f"'{text}' in {to_lang} is '{_TRANSLATIONS[key]}'"
    return f"ERROR: no translation for '{text}' -> {to_lang}"


translate.example_args = {"text": "hello", "to_lang": "spanish"}


_DEFINITIONS = {
    "ephemeral": "lasting for a very short time",
    "ubiquitous": "present, appearing, or found everywhere",
}


def get_definition(word: str) -> str:
    """Look up the dictionary definition of a word."""
    w = word.strip().lower()
    return f"{w}: {_DEFINITIONS[w]}" if w in _DEFINITIONS else f"ERROR: no entry for '{word}'"


get_definition.example_args = {"word": "ephemeral"}


_POPULATIONS = {"Tokyo": "about 14 million people (city proper)", "Canberra": "about 460,000"}


def get_population(city: str) -> str:
    """Get the population of a city."""
    return _POPULATIONS.get(city, f"ERROR: no population data for {city}")


get_population.example_args = {"city": "Tokyo"}


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to} with subject '{subject}'."


send_email.example_args = {
    "to": "alice@example.com",
    "subject": "Report ready",
    "body": "The report is ready for review.",
}


_CONTACTS = {"bob": "Bob <bob@example.com>", "alice": "Alice <alice@example.com>"}


def get_contact(name: str) -> str:
    """Look up a person's email address in the contact list."""
    return _CONTACTS.get(name.strip().lower(), f"ERROR: no contact named {name}")


get_contact.example_args = {"name": "Bob"}


_DISTANCES = {
    frozenset(("London", "Tokyo")): "9,560 km",
    frozenset(("London", "Cairo")): "3,510 km",
}


def get_distance(city_a: str, city_b: str) -> str:
    """Get the approximate distance between two cities."""
    key = frozenset((city_a, city_b))
    if key in _DISTANCES:
        return f"{city_a} to {city_b}: {_DISTANCES[key]}"
    return f"ERROR: no distance data for {city_a} -> {city_b}"


get_distance.example_args = {"city_a": "London", "city_b": "Tokyo"}
