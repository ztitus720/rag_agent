_sessions = {}


def get_history(session_id: str):
    return _sessions.get(session_id, [])


def add_message(session_id: str, role: str, content: str):
    if session_id not in _sessions:
        _sessions[session_id] = []

    _sessions[session_id].append(
        {
            "role": role,
            "content": content,
        }
    )


def clear_history(session_id: str):
    _sessions.pop(session_id, None)