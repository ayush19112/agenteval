"""Knowledge-base lookup tool for multi-hop reasoning tasks."""

_FACTS = {
    "best_picture_2020": (
        "The Best Picture winner at the 2020 Oscars was Parasite, directed by Bong Joon-ho."
    ),
    "bong_joon_ho_films": (
        "Bong Joon-ho also directed Snowpiercer, Okja, and Memories of Murder."
    ),
    "capital_of_australia": "The capital of Australia is Canberra (not Sydney).",
    "canberra_population": "Canberra has a population of about 460,000.",
    "tallest_mountain": "Mount Everest is the tallest mountain at 8,849 meters.",
}


def lookup_fact(topic: str) -> str:
    """Look up a fact by topic key. Available topics: best_picture_2020,
    bong_joon_ho_films, capital_of_australia, canberra_population, tallest_mountain."""
    return _FACTS.get(topic, f"No fact found for topic '{topic}'")


lookup_fact.example_args = {"topic": "best_picture_2020"}
