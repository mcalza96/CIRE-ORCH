import re

def refactor():
    with open('app/graph/universal/steps.py', 'r') as f:
        content = f.read()

    # Add new utils imports
    content = content.replace(
        "    _append_stage_timing,\n",
        "    state_get_list,\n    state_get_dict,\n    state_get_int,\n    state_get_str,\n    track_node_timing,\n    _append_stage_timing,\n"
    )

    # Dictionary casts
    # raw_partials = state.get("partial_answers")\n    partial_answers_list: list[Any] = raw_partials if isinstance(raw_partials, list) else []
    # into partial_answers_list = state_get_list(state, "partial_answers")
    
    # Simple list replacements
    content = re.sub(
        r'([a-zA-Z0-9_]+)\s*=\s*list\(state\.get\("([^"]+)"\)\s*or\s*\[\]\)',
        r'\1 = state_get_list(state, "\2")',
        content
    )

    # Int replacements
    content = re.sub(
        r'([a-zA-Z0-9_]+)\s*=\s*int\(state\.get\("([^"]+)"\)\s*or\s*0\)',
        r'\1 = state_get_int(state, "\2", 0)',
        content
    )
    
    # Dict replacements
    content = re.sub(
        r'([a-zA-Z0-9_]+)\s*=\s*dict\(state\.get\("([^"]+)"\)\s*or\s*\{\}\)',
        r'\1 = state_get_dict(state, "\2")',
        content
    )

    # String replacements
    content = re.sub(
        r'([a-zA-Z0-9_]+)\s*=\s*str\(state\.get\("([^"]+)"\)\s*or\s*""\)',
        r'\1 = state_get_str(state, "\2", "")',
        content
    )

    # Now the stage_timings manual inserts on return
    nodes = [
        ("planner", "planner_node"),
        ("execute_tool", "execute_tool_node"),
        ("reflect", "reflect_node"),
        ("subquery_aggregate", "aggregate_subqueries_node"),
        ("generator", "generator_node"),
        ("validation", "citation_validate_node")
    ]

    for stage, node_func in nodes:
        # 1. Add decorator before the `async def`
        content = re.sub(
            r"(async def " + node_func + r")\b",
            f"@track_node_timing(\"{stage}\")\n\\1",
            content
        )

        # 2. Remove "t0 = time.perf_counter()" inside this function.
        # Since node funcs are unique, we just remove all t0 = time.perf_counter()
        # but only exactly matching the ones at the start of functions (they are indented by 4 spaces)
        
    content = re.sub(r"^\s*t0 = time\.perf_counter\(\)\n", "", content, flags=re.MULTILINE)

    # 3. Remove manual addition of stage_timings_ms from all return blocks.
    # Pattern to find: "stage_timings_ms": _append_stage_timing(
    #        state, stage="xyz", elapsed_ms=(time.perf_counter() - t0) * 1000.0
    #    )
    # or similar
    
    # We can match `, "stage_timings_ms": _append_stage_timing(...)` entirely
    
    pattern = re.compile(
        r',\s*"stage_timings_ms":\s*_append_stage_timing\([^)]*\)',
        re.MULTILINE | re.DOTALL
    )
    # Wait, the closing paren for _append_stage_timing is )
    # Let's match more carefully
    pattern = re.compile(
        r',\s*"stage_timings_ms":\s*_append_stage_timing\(\s*state,\s*stage="[^"]+",\s*elapsed_ms=\(time\.perf_counter\(\) - t0\) \* 1000\.0\s*\)',
        re.MULTILINE
    )
    content = pattern.sub("", content)

    # Let's also check for cases without the comma before it
    pattern2 = re.compile(
        r'"stage_timings_ms":\s*_append_stage_timing\(\s*state,\s*stage="[^"]+",\s*elapsed_ms=\(time\.perf_counter\(\) - t0\) \* 1000\.0\s*\),?',
        re.MULTILINE
    )
    content = pattern2.sub("", content)

    # Edge cases when elapsed_ms spans lines? The existing code is exactly:
    # "stage_timings_ms": _append_stage_timing(
    #     state, stage="planner", elapsed_ms=(time.perf_counter() - t0) * 1000.0
    # ),

    pattern3 = r',\s*"stage_timings_ms":\s*_append_stage_timing\(\s*state,\s*stage="[^"]+",\s*elapsed_ms=\(time\.perf_counter\(\) - t0\) \* 1000\.0\n\s*\)'
    content = re.sub(pattern3, "", content, flags=re.MULTILINE)
    
    pattern4 = r'"stage_timings_ms":\s*_append_stage_timing\(\s*state,\s*stage="[^"]+",\s*elapsed_ms=\(time\.perf_counter\(\) - t0\) \* 1000\.0\n\s*\),?'
    content = re.sub(pattern4, "", content, flags=re.MULTILINE)

    with open('app/graph/universal/steps.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    refactor()
