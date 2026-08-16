"""
RETRIEVAL — run a query and show the results nicely.

Takes the results from the vector store (score + item index), looks up the
matching items, and prints them with a TYPE icon, PAGE number, and a readable
preview so you can see WHAT matched and WHERE it came from.
"""

ICON = {"text": "📝 TEXT ", "table": "📊 TABLE", "image": "🖼️  IMAGE"}


def format_result(rank, score, item):
    """Return the printable lines for one retrieved item."""
    where = item.get("source", "?")
    if item.get("page"):
        where += f", page {item['page']}"
    head = f"{rank}. {ICON[item['type']]}  (score {score:0.3f}, {where})"
    body = item["content"]
    if item["type"] == "table":
        preview = body                      # show the whole small table
    else:
        preview = body[:220] + ("…" if len(body) > 220 else "")
    indented = "\n".join(f"        {line}" for line in preview.splitlines())
    return f"{head}\n{indented}"


def print_results(query, hits, items):
    """hits is a list of (score, item_index) from the vector store."""
    print(f'\n🔎 Query: "{query}"')
    print("─" * 72)
    for rank, (score, i) in enumerate(hits, 1):
        print(format_result(rank, score, items[i]))
    print()
