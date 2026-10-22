class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    """Returns the given replies in order and records every call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return FakeResponse(self.replies.pop(0))
