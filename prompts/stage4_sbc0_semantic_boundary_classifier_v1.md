You are judging one boundary of a coarse highlight segment.

A short video clip is attached. The clip spans roughly 4 seconds before and 4 seconds after one candidate boundary. The candidate boundary is located at {boundary_offset_in_clip_sec} seconds from the start of this clip, and the clip duration is {clip_duration_sec} seconds.

Context:
- side = {side}
- The candidate segment duration is about {candidate_duration_sec} seconds.
- {side_direction}

You must decide whether this boundary should move inward, stay, or move outward.

Definitions:
- TRIM: the current boundary includes non-highlight context outside the core event, so move it inward.
- KEEP: the current boundary is acceptable or evidence is ambiguous.
- EXPAND: the highlight event appears to start before this boundary or continue after this boundary, so move it outward.

Use KEEP when uncertain.

Return strict JSON only, no markdown, in exactly this shape:
{"action": "TRIM|KEEP|EXPAND", "confidence": 0.0, "rationale_short": "<=20 words"}
