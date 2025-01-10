def get_role(message):
    """
    Safely get the role from a dict or a pydantic model.
    Adjust if your code uses message.role vs message["role"].
    """
    # If it's a Pydantic ChatCompletionMessage with an attribute:
    if hasattr(message, "role"):
        return message.role
    # If it's a dict
    if isinstance(message, dict):
        return message.get("role", "assistant")
    # Fallback
    return "assistant"


def get_tool_call_id(message) -> str:
    """
    If this is a tool message, return its ID or None.
    """
    if hasattr(message, "tool_call_id"):
        return message.tool_call_id
    if isinstance(message, dict):
        return message.get("tool_call_id")
    return None


def prune_messages_with_tool_pairs(messages, max_count: int) -> list:
    """
    Keep:
      - All leading 'system' messages.
      - Then from the remainder, find the *last* `max_count` user/assistant messages,
        and keep everything from the earliest of those user/assistant messages
        all the way to the end of the conversation.

    Because we keep a contiguous slice of the conversation, any tool calls or
    tool responses after that earliest user/assistant are also retained,
    ensuring we don't break the chain of 'assistant' => 'tool' => 'assistant'.
    """

    # 1) Gather leading system messages.
    system_msgs = []
    i = 0
    while i < len(messages) and get_role(messages[i]) == "system":
        system_msgs.append(messages[i])
        i += 1

    # 2) The remainder is everything after system messages:
    remainder = messages[i:]
    if not remainder:
        return system_msgs  # No non-system messages

    # 3) Identify indices of user/assistant messages within remainder
    user_asst_indices = []
    for idx, msg in enumerate(remainder):
        r = get_role(msg)
        if r in ("user", "assistant"):
            user_asst_indices.append(idx)

    # 4) If we have <= max_count user/assistant messages, keep everything
    if len(user_asst_indices) <= max_count:
        return messages  # nothing to prune

    # 5) Otherwise, find the earliest user/assistant we want to keep:
    #    e.g. the start of the last `max_count` user/assistant messages
    earliest_needed_index = user_asst_indices[-max_count]

    # 6) Keep that slice from earliest_needed_index to the end
    final_slice = remainder[earliest_needed_index:]

    # Rebuild final list: system messages + final slice
    return system_msgs + final_slice