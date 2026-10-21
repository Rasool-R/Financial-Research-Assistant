from financial_assistant.graph import financial_graph


def main():
    result = financial_graph.invoke(
        {
            "user_query": (
                "Analyze the technical conditions of gold "
                "and investigate relevant recent market news."
            ),
            "symbol": "XAU/USD",
            "interval": "1h",
        }
    )

    print("\n--- Final Financial Research Report ---")
    print(result["final_report"])


if __name__ == "__main__":
    main()
