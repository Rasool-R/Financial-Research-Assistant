import json

from financial_assistant.tools.market_summary import get_market_summary


def main():
    for symbol in ["XAU/USD", "EUR/USD"]:
        summary = get_market_summary(
            symbol=symbol,
            interval="1h",
            outputsize=100,
        )

        print(f"\n--- {symbol} ---")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
