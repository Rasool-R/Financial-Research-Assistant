from pprint import pprint

from financial_assistant.nodes.market_agent import market_agent


def main():
    result = market_agent(
        {
            "symbol": "XAU/USD",
            "interval": "1h",
        }
    )

    print("\n--- Market Summary ---")
    pprint(result["market_summary"])

    print("\n--- AI Market Analysis ---")
    print(result["market_analysis"])


if __name__ == "__main__":
    main()
