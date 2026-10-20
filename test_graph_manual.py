from financial_assistant.graph import financial_graph


def main():
    result = financial_graph.invoke(
        {
            "user_query": "Analyze the current market conditions.",
            "symbol": "XAU/USD",
            "interval": "1h",
        }
    )

    print("\n--- Market Summary ---")
    print(result["market_summary"])

    print("\n--- Market Analysis ---")
    print(result["market_analysis"])


if __name__ == "__main__":
    main()
