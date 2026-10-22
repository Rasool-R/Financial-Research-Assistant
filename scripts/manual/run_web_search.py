from pprint import pprint

from financial_assistant.tools.web_search import search_web


def main():
    results = search_web(
        query="latest gold market news",
        max_results=5,
    )

    print(f"Results found: {len(results)}")
    pprint(results)


if __name__ == "__main__":
    main()
