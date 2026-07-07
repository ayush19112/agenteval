"""Sandbox tools for multi-step dependent workflows."""


def get_flight_price(from_city: str, to_city: str) -> str:
    """Get the price of a one-way flight between two cities, in USD."""
    routes = {("London", "Tokyo"): 850, ("Tokyo", "London"): 890, ("London", "Cairo"): 320}
    price = routes.get((from_city, to_city))
    if price is None:
        return f"ERROR: no route data for {from_city} -> {to_city}"
    return f"The flight from {from_city} to {to_city} costs {price} USD."


get_flight_price.example_args = {"from_city": "London", "to_city": "Tokyo"}


_FILES = {
    "notes.txt": "meeting on monday about the launch",
    "report.txt": "the launch is on track with three risks",
    "data.csv": "id,value\n1,2",
}


def list_files() -> str:
    """List the files in the project directory."""
    return ", ".join(_FILES)


list_files.example_args = {}


def read_file(name: str) -> str:
    """Read the contents of a file by name."""
    if name not in _FILES:
        return f"ERROR: no such file '{name}'"
    return _FILES[name]


read_file.example_args = {"name": "notes.txt"}


def check_inventory(item: str) -> str:
    """Check how many units of an item are in stock."""
    stock = {"widget": 42, "gadget": 0}
    if item not in stock:
        return f"ERROR: unknown item '{item}'"
    return f"{item}: {stock[item]} units in stock"


check_inventory.example_args = {"item": "widget"}


def place_order(item: str, quantity: int) -> str:
    """Place an order for a quantity of an item. Returns the order id."""
    return f"Order ORD-7 placed for {quantity} x {item}."


place_order.example_args = {"item": "widget", "quantity": 3}


def get_order_status(order_id: str) -> str:
    """Get the current status of an order by its id."""
    return f"{order_id}: confirmed" if order_id == "ORD-7" else f"ERROR: unknown order {order_id}"


get_order_status.example_args = {"order_id": "ORD-7"}


def search_products(query: str) -> str:
    """Search the product catalog."""
    if "widget" in query.lower():
        return "Results: widget-pro ($25), widget-mini ($10)"
    return f"No products found for '{query}'"


search_products.example_args = {"query": "widget"}


def add_to_cart(item: str, quantity: int) -> str:
    """Add a quantity of an item to the shopping cart."""
    prices = {"widget-pro": 25, "widget-mini": 10}
    if item not in prices:
        return f"ERROR: unknown item '{item}'"
    return f"Added {quantity} x {item} to cart (subtotal ${prices[item] * quantity})."


add_to_cart.example_args = {"item": "widget-mini", "quantity": 3}


def checkout() -> str:
    """Place the order for everything in the cart."""
    return "Order placed: confirmation CHK-9."


checkout.example_args = {}


def get_balance(account: str) -> str:
    """Get the balance of a bank account ('checking' or 'savings')."""
    balances = {"checking": "$1,250.00", "savings": "$8,300.00"}
    if account not in balances:
        return f"ERROR: no account {account}"
    return f"{account}: {balances[account]}"


get_balance.example_args = {"account": "checking"}


def transfer(from_account: str, to_account: str, amount: float) -> str:
    """Transfer money between accounts."""
    return (
        f"Transferred ${amount:,.2f} from {from_account} to {to_account}. "
        "New balances — checking: $1,050.00, savings: $8,500.00."
    )


transfer.example_args = {"from_account": "checking", "to_account": "savings", "amount": 200}


def get_calendar(day: str) -> str:
    """List the calendar events for a day."""
    schedule = {"friday": "9:00 standup; 14:00 design review"}
    return schedule.get(day.lower(), f"No events on {day}")


get_calendar.example_args = {"day": "friday"}


def add_calendar(event: str, day: str, time: str) -> str:
    """Add an event to the calendar."""
    return f"Added '{event}' on {day} at {time}."


add_calendar.example_args = {"event": "1:1 with Bob", "day": "friday", "time": "15:00"}
