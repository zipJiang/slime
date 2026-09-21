"""Trace-derived search repetition metrics; repeats are evidence, not proof of waste."""

from __future__ import annotations

import json
from typing import Any

from step_controller.generation import GenerateResult, PolicyFormat
from step_controller.generation.parsing import ToolTurn
from step_controller.harness import RolloutState

from .calls import call_key
from .env import LocBenchState


def analyze_trace(
    result: RolloutState[LocBenchState], format: PolicyFormat[ToolTurn]
) -> dict[str, Any]:
    seen: dict[tuple[str, str], int] = {}
    read_lines: dict[str, set[int]] = {}
    fold = 0
    previous_fold = False
    events = []
    fold_events = []
    invalid_submissions = 0
    no_tool_turns = 0
    for index, turn in enumerate(result.turns):
        if not turn.tokens:
            continue
        action = format.parser.parse(
            GenerateResult(text=format.codec.decode(turn.tokens), tokens=turn.tokens)
        )
        if turn.tag == "fold":
            fold += int(not previous_fold)
            previous_fold = True
            fold_events.append(
                {
                    "turn_index": index,
                    "fold_index": fold,
                    "completion_tokens": len(turn.tokens),
                    "visible_text": action.text,
                    "parsed_tool_calls": len(action.tool_calls),
                    "tool_syntax_in_visible_text": any(
                        marker in action.text
                        for marker in ("[tool call:", "<tool_call>", "<function=")
                    ),
                }
            )
            continue
        previous_fold = False
        # In this environment only submit or forced finalization ends a task.
        # Finalization executes no research calls, even if the model emits one.
        if turn.transition is not None and turn.transition.done:
            continue
        replies = turn.transition.messages if turn.transition is not None else ()
        if any(call.name == "submit" for call in action.tool_calls):
            invalid_submissions += int(
                any(reply.get("content", "").startswith("error:") for reply in replies)
            )
            continue  # submit takes precedence; sibling calls were not executed
        no_tool_turns += int(not action.tool_calls)
        for call_index, call in enumerate(action.tool_calls):
            content = (
                replies[call_index].get("content", "")
                if call_index < len(replies)
                else ""
            )
            key = call_key(call.name, call.arguments)
            error = content.startswith(("error", "Tool "))
            event = {
                "turn_index": index,
                "fold_index": fold,
                "tool": call.name,
                "arguments": call.arguments,
                "repeat": not error and key in seen,
                "repeat_from_before_fold": not error
                and key in seen
                and seen[key] < fold,
                "error": error,
                "read_lines": 0,
                "previously_read_lines": 0,
            }
            if not error:
                if call.name == "read":
                    try:
                        reply = json.loads(content)
                        lines = {
                            int(line.partition(":")[0])
                            for line in reply["text"].splitlines()
                            if line.partition(":")[0].isdigit()
                        }
                        old = read_lines.setdefault(reply["path"], set())
                        event["read_lines"] = len(lines)
                        event["previously_read_lines"] = len(lines & old)
                        old.update(lines)
                    except (KeyError, ValueError, TypeError):
                        pass
                seen.setdefault(key, fold)
            events.append(event)
    return {
        "tool_calls": len(events),
        "exact_repeat_calls": sum(bool(e["repeat"]) for e in events),
        "repeats_from_before_fold": sum(
            bool(e["repeat_from_before_fold"]) for e in events
        ),
        "tool_errors": sum(bool(e["error"]) for e in events),
        "read_lines": sum(int(e["read_lines"]) for e in events),
        "previously_read_lines": sum(int(e["previously_read_lines"]) for e in events),
        "invalid_submissions": invalid_submissions,
        "no_tool_turns": no_tool_turns,
        "fold_replies": len(fold_events),
        "tool_shaped_fold_replies": sum(
            bool(e["parsed_tool_calls"] or e["tool_syntax_in_visible_text"])
            for e in fold_events
        ),
        "fold_events": fold_events,
        "events": events,
    }
