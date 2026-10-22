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

    print("\n--- Review ---")
    print(result["review_feedback"])
    print(f"\nRevisions: {result.get('revision_count', 0)}")

    if result.get("errors"):
        print("\n--- Non-fatal errors ---")
        for error in result["errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
