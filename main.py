"""Ask a market question in plain language.

    python main.py                                  # interactive
    python main.py "قیمت دو ساعت آینده طلا چطور خواهد بود؟"
    python main.py "will gold go up in the next 2 hours?" --lang en
"""

from financial_assistant.app import main

if __name__ == "__main__":
    raise SystemExit(main())
