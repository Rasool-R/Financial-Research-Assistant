from financial_assistant.config import get_model


def main():
    model = get_model()
    response = model.invoke(
        "Reply with exactly: API connection successful"
    )

    print(response.content)


if __name__ == "__main__":
    main()